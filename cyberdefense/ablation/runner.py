"""OpTC component ablations of SR-CPPO.
Experiment orchestration and result export."""

import os
import argparse
import numpy as np
import torch
import json
import math
import platform
from pathlib import Path

from .agent import AblationAgent
from .config import (
    Arm,
    CORE_HASH,
    DATASET,
    IMPLEMENTATION_VERSION,
    LABELS,
    METRICS,
    REFERENCE_WEIGHTS,
    SCHEMA,
    SEEDS,
    TRAIN_ARMS,
    condition_tag,
)
from cyberdefense.common.data import data_paths
from cyberdefense.common.runtime import package_digest
from .environment import AblationEnv
from .evaluation import evaluation
from .io import digest_file, digest_object, load_torch, load_weights, write_csv, write_json
from .runtime import logger, make_logger, set_global_seed
from .training import train_seed


def resolve_data(args):
    files = data_paths("optc", args.data_dir, horizon=args.horizon)
    return [Path(p) for p in files]


def runtime_info(device):
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "device": device,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def build_metadata(args, arm, seed, data, runtime):
    training = {
        "schema": SCHEMA,
        "patch_version": IMPLEMENTATION_VERSION,
        "base_source_sha256": CORE_HASH,
        "runner_sha256": package_digest(),
        "dataset": DATASET,
        "variant": arm.name,
        "seed": seed,
        "data_sha256": data,
        "train_eps": args.train_eps,
        "horizon": args.horizon,
        "update_epochs": args.update_epochs,
        "runtime": runtime,
    }
    return {
        **training,
        "training_signature": digest_object(
            {k: v for k, v in training.items() if k != "runner_sha256"}
        ),
        "eval_protocol": args.eval_protocol,
        "eval_eps": args.eval_eps,
        "confidence_cap": args.confidence_cap,
        "evaluation_signature": digest_object(
            {
                "data": data,
                "protocol": args.eval_protocol,
                "eval_eps": args.eval_eps,
                "horizon": args.horizon,
                "confidence_cap": args.confidence_cap,
            }
        ),
        "weights_native": arm.weights,
        "weights_reference": REFERENCE_WEIGHTS,
        "interpretation": "conditional total pipeline effect; native feedback and selection retained",
        "evaluation_scope": "same arrays as training; frozen networks, stateful controller; no proven temporal holdout",
    }


def external_paths(args, seed, role):
    if args.full_manifest:
        manifest = json.loads(Path(args.full_manifest).read_text(encoding="utf-8-sig"))
        entry = manifest.get("checkpoints", manifest).get(str(seed), {})
        value = entry.get(role)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = Path(args.full_manifest).resolve().parent / path
            return path
    if args.full_checkpoint_dir:
        directory = Path(args.full_checkpoint_dir)
        named = (
            directory
            / f"optc_v721_{('best_safe' if role == 'best_safe' else 'final')}_seed{seed}.pt"
        )
        if named.is_file():
            return named
        if role == "best_safe":
            generic = directory / "optc_v721_best_safe.pt"
            if generic.is_file():
                checkpoint = load_torch(generic)
                if int(checkpoint.get("seed", -1)) == seed:
                    return generic
    return None


def refresh_incompatible_outputs(run_dir, prefix):
    for p in list(run_dir.glob(f"{prefix}*")):
        if p.is_file():
            p.unlink()


def run_one(args, name, seed, files, hashes, runtime):
    arm = Arm(name)
    run_dir = (
        Path(args.output)
        / DATASET
        / f"{args.eval_protocol}_{condition_tag(args)}"
        / name
        / f"s{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"OpTC_{name}_s{seed}"
    make_logger(run_dir / f"{prefix}.log")
    metadata = build_metadata(args, arm, seed, hashes, runtime)
    metadata_path = run_dir / f"{prefix}_config.json"
    result_path = run_dir / f"{prefix}_result.json"
    if (
        name not in ("no_best_safe", "no_criticality")
        and result_path.exists()
        and (not args.reevaluate)
    ):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "complete":
            saved_path = Path(result["checkpoint_path"])
            if saved_path.is_file() and digest_file(saved_path) == result["checkpoint_sha256"]:
                logger.info("[%s] Validated completed result; skipping", prefix)
                return result
    if name in ("no_best_safe", "no_criticality"):
        needs_purge = False
        if metadata_path.exists():
            prev_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if prev_meta.get("patch_version") != IMPLEMENTATION_VERSION:
                needs_purge = True
        if result_path.exists():
            prev_res = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                prev_res.get("status") != "complete"
                or prev_res.get("metadata", {}).get("patch_version") != IMPLEMENTATION_VERSION
            ):
                needs_purge = True
        if needs_purge:
            logger.info("[%s] Refreshing incompatible output files for %s", prefix, name)
            refresh_incompatible_outputs(run_dir, prefix)
    if result_path.exists() and (not args.reevaluate):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "complete":
            saved_path = Path(result["checkpoint_path"])
            if saved_path.is_file() and digest_file(saved_path) == result["checkpoint_sha256"]:
                logger.info("[%s] Validated completed result; skipping", prefix)
                return result
    write_json(metadata_path, metadata)
    env = AblationEnv(*files, arm=arm, horizon=args.horizon)
    set_global_seed(seed)
    agent = AblationAgent(arm, run_dir, prefix, metadata, seed=seed, device=runtime["device"])
    counters = {}
    source_run_dir = run_dir
    if args.confidence_cap is not None:
        source_run_dir = (
            Path(args.output) / DATASET / f"{args.eval_protocol}_nominal" / name / f"s{seed}"
        )
    if name in ("full_reference", "deployment_no_fallback"):
        role = "best_safe"
        checkpoint_path = external_paths(args, seed, role)
        if checkpoint_path is None or not checkpoint_path.is_file():
            result = {
                "dataset": DATASET,
                "variant": name,
                "seed": seed,
                "status": "unavailable",
                "reason": f"Missing original Full {role} checkpoint. No Full retraining or substitution performed.",
                "metadata": metadata,
            }
            write_json(result_path, result)
            logger.warning(result["reason"])
            return result
        checkpoint = load_weights(agent, checkpoint_path, expected_role=None, external=True)
        source = str(checkpoint_path.resolve())
        role_selected = "external_" + role
        checkpoint_hash = digest_file(checkpoint_path)
        counters = checkpoint.get("counters", {})
    elif name == "no_best_safe":
        ext_final = external_paths(args, seed, "final")
        if ext_final is not None and ext_final.is_file():
            checkpoint = load_weights(agent, ext_final, expected_role="final", external=True)
            source = str(ext_final.resolve())
            role_selected = "external_final"
            checkpoint_hash = digest_file(ext_final)
            counters = checkpoint.get("counters", {})
        elif args.evaluate_only:
            checkpoint_path = source_run_dir / f"{prefix}_selected.pt"
            checkpoint = load_weights(agent, checkpoint_path)
            role_selected = checkpoint.get("selected_role", "best_snapshot_fallback")
            counters = checkpoint.get("counters", {})
            source, checkpoint_hash = (str(checkpoint_path.resolve()), digest_file(checkpoint_path))
        else:
            role_selected, counters = train_seed(env, agent, args)
            checkpoint_path = run_dir / f"{prefix}_selected.pt"
            source, checkpoint_hash = (str(checkpoint_path.resolve()), digest_file(checkpoint_path))
    else:
        if args.evaluate_only:
            checkpoint_path = source_run_dir / f"{prefix}_selected.pt"
            checkpoint = load_weights(agent, checkpoint_path)
            role_selected = checkpoint.get("selected_role", "selected")
            counters = checkpoint.get("counters", {})
        else:
            role_selected, counters = train_seed(env, agent, args)
            checkpoint_path = run_dir / f"{prefix}_selected.pt"
        source, checkpoint_hash = (str(checkpoint_path.resolve()), digest_file(checkpoint_path))
    metrics = evaluation(env, agent, args, "selected")
    if name in TRAIN_ARMS + ("no_decayed_no_anti_windup",):
        final_path = source_run_dir / f"{prefix}_final.pt"
        if final_path.is_file():
            load_weights(agent, final_path, expected_role="final")
            final_metrics = evaluation(env, agent, args, "final")
            final_result = {
                "dataset": DATASET,
                "variant": name,
                "seed": seed,
                "endpoint": "final",
                **final_metrics,
            }
            write_json(run_dir / f"{prefix}_final_result.json", final_result)
    complete = not counters or counters.get("failed_updates", 0) == 0
    result = {
        "dataset": DATASET,
        "variant": name,
        "seed": seed,
        "status": "complete" if complete else "invalid",
        "endpoint": "selected",
        "checkpoint_role": role_selected,
        "checkpoint_path": source,
        "checkpoint_sha256": checkpoint_hash,
        "training_counters": counters,
        "metadata": metadata,
        **metrics,
    }
    write_json(result_path, result)
    write_csv(
        run_dir / f"{prefix}_raw.csv",
        [{k: v for k, v in result.items() if k not in ("metadata", "training_counters")}],
    )
    logger.info(
        "[%s] COMPLETE: Reward=%.4f R_J=%.4f Vmin_A=%.4f FDR=%.4f%% AIR=%.4f%%",
        prefix,
        metrics["Avg_Reward"],
        metrics["R_J"],
        metrics["Vmin_A"],
        metrics["FDR_pct"],
        metrics["AIR_pct"],
    )
    return result


def summaries(args):
    root = Path(args.output) / DATASET / f"{args.eval_protocol}_{condition_tag(args)}"
    root.mkdir(parents=True, exist_ok=True)
    results = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(root.glob("*/s*/OpTC_*_result.json"))
    ]
    results = [r for r in results if "metadata" in r]
    rows, long, status, paired, finals = ([], [], [], [], [])
    expected_seeds = SEEDS
    visible_names = [
        name
        for name in LABELS
        if name in args.variants or any((r["variant"] == name for r in results))
    ]
    selected = {
        name: [r for r in results if r["variant"] == name and r["seed"] in expected_seeds]
        for name in visible_names
    }
    protocol_tuples = {
        (
            r["metadata"].get("eval_protocol"),
            r["metadata"].get("eval_eps"),
            r["metadata"].get("horizon"),
            r["metadata"].get("confidence_cap"),
        )
        for group in selected.values()
        for r in group
        if r["status"] == "complete"
    }
    if len(protocol_tuples) > 1:
        raise ValueError("Cannot pool different evaluation protocols/horizons in one table")
    order = visible_names
    table = [
        "| Model | Avg. Reward ↑ | R_J ↓ | Vmin_A ↑ | FDR (%) ↓ | AIR (%) ↑ |",
        "|---|---:|---:|---:|---:|---:|",
        "| SLA reference | — | < 13.25 | ≥ 0.68 | approximately 0 | — |",
    ]
    for name in order:
        group = selected[name]
        good = [r for r in group if r["status"] == "complete"]
        by_seed = {r["seed"]: r for r in good}
        full_n = set(by_seed) == set(expected_seeds)
        status.append(
            {
                "dataset": DATASET,
                "variant": name,
                "expected_seeds": " ".join(map(str, expected_seeds)),
                "completed_seeds": " ".join(map(str, sorted(by_seed))),
                "n": len(by_seed),
                "status": "complete_5seeds" if full_n else "incomplete_or_nonstandard",
                "reason": "; ".join(
                    (r.get("reason", "") for r in group if r["status"] != "complete")
                ),
            }
        )
        formatted = []
        for metric in METRICS:
            values = np.asarray(
                [by_seed[s][metric] for s in expected_seeds if s in by_seed], dtype=float
            )
            if values.size:
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else None
                half = 2.7764451051977987 * sd / math.sqrt(5) if full_n else None
                long.append(
                    {
                        "dataset": DATASET,
                        "variant": name,
                        "metric": metric,
                        "n": len(values),
                        "mean": mean,
                        "sample_sd": sd,
                        "ci95_low": mean - half if half is not None else None,
                        "ci95_high": mean + half if half is not None else None,
                        "ci_method": (
                            "Student-t df=4, untruncated"
                            if full_n
                            else "not computed for incomplete/nonstandard group"
                        ),
                    }
                )
                formatted.append(f"{mean:.3f} ± {sd:.3f}" if full_n else f"NA (n={len(values)}/5)")
            else:
                formatted.append("NA (n=0/5)")
        table.append("| " + LABELS[name] + " | " + " | ".join(formatted[:5]) + " |")
        rows.extend(
            (
                {k: v for k, v in r.items() if k not in ("metadata", "training_counters")}
                for r in good
            )
        )
        full_by_seed = {
            r["seed"]: r for r in selected.get("full_reference", []) if r["status"] == "complete"
        }
        if full_n and set(full_by_seed) == set(expected_seeds) and (name != "full_reference"):
            for metric in METRICS:
                diffs = np.asarray(
                    [by_seed[s][metric] - full_by_seed[s][metric] for s in expected_seeds]
                )
                half = 2.7764451051977987 * float(diffs.std(ddof=1)) / math.sqrt(5)
                paired.append(
                    {
                        "dataset": DATASET,
                        "variant": name,
                        "metric": metric,
                        "n_pairs": 5,
                        "difference": "ablation minus Full",
                        "mean_difference": float(diffs.mean()),
                        "sample_sd_difference": float(diffs.std(ddof=1)),
                        "ci95_low": float(diffs.mean()) - half,
                        "ci95_high": float(diffs.mean()) + half,
                        "note": "descriptive small-n t interval; no significance claim",
                    }
                )
    for p in sorted(root.glob("*/s*/OpTC_*_final_result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["variant"] in visible_names and r["seed"] in expected_seeds:
            finals.append(r)
    tag = f"OpTC_ablation_{args.eval_protocol}_{condition_tag(args)}"
    write_csv(root / f"{tag}_raw.csv", rows)
    write_csv(root / f"{tag}_summary.csv", long)
    write_csv(root / f"{tag}_status.csv", status)
    write_csv(root / f"{tag}_paired_differences.csv", paired)
    write_csv(root / f"{tag}_final_endpoint_raw.csv", finals)
    table.extend(
        [
            "",
            "Each complete row is mean ± sample SD across five training seeds: 42, 43, 45, 46, 47.",
            "NA is not a five-seed result. Reference weights=(0.20,0.90); survivability scale=[0,1.10].",
            "FDR preserves the original benign a2-episode-rate definition. AIR counts successful interception events / active attack steps.",
            "Vmin_A is the malicious-episode mean of episode minima. R_J is evaluated on the same episodes as mixture CVaR.",
            "Networks are frozen; fallback has sequential rolling state. Evaluation uses the same telemetry arrays, not a verified temporal holdout.",
            "Ablations estimate conditional pipeline contributions; no additive or interaction claim without a factorial analysis.",
        ]
    )
    (root / f"{tag}_paper_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Integrated five-seed OpTC ablations; never retrains Full"
    )
    parser.add_argument("--variants", nargs="+", choices=list(LABELS), default=list(TRAIN_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--train-eps", type=int, default=1500)
    parser.add_argument("--eval-eps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--update-epochs", type=int, default=18)
    parser.add_argument("--data-dir")
    parser.add_argument("--output", default="outputs/ablation")
    parser.add_argument("--full-checkpoint-dir")
    parser.add_argument(
        "--full-manifest",
        help="JSON with checkpoint paths per seed; relative paths resolve from manifest",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--eval-protocol", choices=("indexed", "legacy"), default="indexed")
    parser.add_argument(
        "--confidence-cap",
        type=float,
        default=None,
        help="evaluation-only pressure; separate output condition",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any((s not in SEEDS for s in args.seeds))
    ):
        parser.error("Seeds must be a unique subset of 42 43 45 46 47; seed44 is excluded")
    if (
        min(
            args.train_eps,
            args.eval_eps,
            args.horizon,
            args.update_epochs,
            args.save_every,
            args.log_every,
        )
        <= 0
    ):
        parser.error("Episode/epoch/horizon/interval settings must be positive")
    if args.confidence_cap is not None and (not 0 <= args.confidence_cap <= 1):
        parser.error("--confidence-cap must be in [0,1]")
    if args.confidence_cap is not None and (not (args.evaluate_only or args.summarize_only)):
        parser.error("Pressure tests reuse trained checkpoints: set --evaluate-only")
    args.variants = list(dict.fromkeys(args.variants))
    if args.full_checkpoint_dir or args.full_manifest:
        args.variants = list(dict.fromkeys(["full_reference"] + args.variants))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summarize_only:
        summaries(args)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA requested but unavailable; install CUDA-enabled PyTorch or select --device cpu"
        )
    runtime = runtime_info(device)
    files = resolve_data(args)
    hashes = {p.name: digest_file(p) for p in files}
    base = Path(args.output) / DATASET / f"{args.eval_protocol}_{condition_tag(args)}"
    base.mkdir(parents=True, exist_ok=True)
    write_json(base / "OpTC_ablation_runtime.json", runtime)
    inventory = []
    for seed in args.seeds:
        for role in ("best_safe", "final"):
            path = external_paths(args, seed, role)
            inventory.append(
                {
                    "dataset": DATASET,
                    "variant": "full_reference",
                    "seed": seed,
                    "role": role,
                    "path": str(path) if path else "",
                    "exists": bool(path and path.is_file()),
                }
            )
    write_csv(base / "OpTC_full_checkpoint_inventory.csv", inventory)
    for name in args.variants:
        for seed in args.seeds:
            try:
                run_one(args, name, seed, files, hashes, runtime)
            except Exception:
                summaries(args)
                raise
            summaries(args)
