"""Unified entry point for SR-CPPO and the five comparison methods."""

import argparse
import importlib
import json
import os
import platform
from pathlib import Path

METHODS = ("sr-cppo", "ppo", "cpo", "cppo-pid", "cvar-ppo", "rule-based")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="sr-cppo")
    parser.add_argument("--dataset", choices=("optc", "dapt2020"), default="optc")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--train-eps", type=int, default=1500)
    parser.add_argument("--eval-eps", type=int, default=200)
    parser.add_argument("--data-dir", help="Directory containing the selected dataset feature pair")
    parser.add_argument("--output", help="Output directory; default: outputs/<dataset>/<method>")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="auto selects CUDA when available; cpu disables CUDA for this process",
    )
    parser.add_argument("--threads", type=int, help="Optional PyTorch CPU thread count")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="SR-CPPO frozen evaluation without deployment fallback",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="SR-CPPO: reuse native best-safe checkpoints in --output",
    )
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        parser.error("--seeds must contain unique integers")
    if min(args.train_eps, args.eval_eps) < 1 or (args.threads is not None and args.threads < 1):
        parser.error("Episode counts and --threads must be positive")
    if args.method != "sr-cppo" and (args.no_fallback or args.eval_only):
        parser.error("--no-fallback and --eval-only apply to --method sr-cppo")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from .common.data import data_paths, select_dataset
    from .common.runtime import configure_logging, output_directory, package_digest

    if args.threads:
        torch.set_num_threads(args.threads)
    select_dataset(args.dataset, args.data_dir)
    files = data_paths()
    out = Path(args.output or f"outputs/{args.dataset}/{args.method}").expanduser().resolve()
    module = importlib.import_module(f'cyberdefense.methods.{args.method.replace("-", "_")}.runner')
    metadata = {
        **vars(args),
        "data_files": list(files),
        "output": str(out),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device_used": "cuda" if torch.cuda.is_available() else "cpu",
        "implementation_sha256": package_digest(),
    }
    with output_directory(out):
        configure_logging(out)
        (out / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.method == "sr-cppo":
            module.run_experiments(
                seeds=tuple(args.seeds),
                total_eps=args.train_eps,
                eval_eps=args.eval_eps,
                enable_fallback_in_eval=not args.no_fallback,
                eval_only=args.eval_only,
            )
        else:
            module.run_multi_seed_experiments(
                seeds=tuple(args.seeds), total_eps=args.train_eps, eval_eps=args.eval_eps
            )
