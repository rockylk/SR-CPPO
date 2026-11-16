"""OpTC component ablations of SR-CPPO.
Episode collection and training schedule."""

import csv
import numpy as np
import torch
import time

from .config import DATASET, SCHEMA
from .io import (
    append_csv,
    atomic_torch_save,
    resume_agent,
    save_resume,
    select_training_checkpoint,
    write_csv,
    write_json,
)
from .runtime import logger


def train_seed(env, agent, args):
    counters = {"successful_updates": 0, "failed_updates": 0, "optimizer_steps": 0, "rollbacks": 0}
    last_path = agent.run_dir / f"{agent.prefix}_last_resume.pt"
    start = 0
    if last_path.exists():
        if not args.resume:
            raise FileExistsError(f"Existing training state: {last_path}; use --resume")
        start, counters = resume_agent(agent, last_path)
        logger.info("[%s] Resuming at episode %d", agent.prefix, start)

    def counted_step(optimizer, call_args, call_kwargs):
        counters["optimizer_steps"] += 1

    hook = agent.opt.register_step_post_hook(counted_step)
    trace_path = agent.run_dir / f"{agent.prefix}_train.csv"
    if start and trace_path.exists():
        with trace_path.open(encoding="utf-8-sig", newline="") as f:
            old_rows = list(csv.DictReader(f))
        kept = [r for r in old_rows if int(r["episode"]) < start]
        if len(kept) != len(old_rows):
            backup = trace_path.with_suffix(".uncommitted.csv")
            backup.write_bytes(trace_path.read_bytes())
            if kept:
                write_csv(trace_path, kept)
            else:
                trace_path.unlink()
    try:
        for ep in range(start, args.train_eps):
            started = time.perf_counter()
            buf = {
                k: []
                for k in (
                    "obs",
                    "beliefs",
                    "phase_logits",
                    "attack_prob",
                    "a_type",
                    "a_mu",
                    "logp_old",
                    "rewards",
                    "costs",
                    "dones",
                )
            }
            labels = []
            o, phase = env.reset()
            b = agent.encoder.init_belief(1, agent.device)
            gate = agent.episode_count >= agent.gate_warmup_eps
            fallback = agent.risk_fallback_active()
            buf["a1_fallback_active"] = bool(fallback)
            counts = [0, 0, 0]
            done = False
            while not done:
                with torch.no_grad():
                    b = agent.encoder(
                        torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0), b
                    )
                    pl, ap = (agent.encoder.phase_logits(b), agent.encoder.attack_prob(b))
                    at, am, lp, _ = agent.actor.act(
                        b, pl, ap, gate_active=gate, a1_fallback_active=fallback
                    )
                a_type, a_mu = (at.item(), am.item())
                o2, r, c, done, phase2, info = env.step(a_type, a_mu)
                for key, value in (
                    ("obs", o.copy()),
                    ("beliefs", b.detach().cpu().numpy()[0]),
                    ("phase_logits", pl.detach().cpu().numpy()[0]),
                    ("attack_prob", float(ap.item())),
                    ("a_type", a_type),
                    ("a_mu", a_mu),
                    ("logp_old", lp.item()),
                    ("rewards", r),
                    ("costs", c),
                    ("dones", float(done)),
                ):
                    buf[key].append(value)
                labels.append(phase)
                counts[a_type] += 1
                o, phase = (o2, phase2)
            rollout_sec = time.perf_counter() - started
            R, C = (float(np.sum(buf["rewards"])), float(np.sum(buf["costs"])))
            risk, vmin = (info["V_drop_ep"], info["V_min_ep"])
            a2_ratio = counts[2] / sum(counts)
            agent.recent_R.append(R)
            agent.recent_C.append(C)
            regime = "a" if env.is_malicious else "b"
            getattr(agent, "recent_C_" + regime).append(C)
            getattr(agent, "recent_Vmin_" + regime).append(vmin)
            getattr(agent, "recent_a2_" + regime).append(a2_ratio)
            getattr(agent, "cost_buf_mal" if env.is_malicious else "cost_buf_benign").append(risk)
            getattr(agent, "cost_recent_mal" if env.is_malicious else "cost_recent_benign").append(
                risk
            )
            update_start = time.perf_counter()
            try:
                stats = agent.update(buf, labels, epochs=args.update_epochs)
                finite = torch.stack(
                    [
                        torch.isfinite(p).all()
                        for group in agent.opt.param_groups
                        for p in group["params"]
                    ]
                ).all()
                if not bool(finite):
                    raise FloatingPointError("Non-finite model parameters after PPO update")
            except Exception as exc:
                counters["failed_updates"] += 1
                write_json(
                    agent.run_dir / f"{agent.prefix}_failure.json",
                    {
                        "dataset": DATASET,
                        "variant": agent.arm.name,
                        "seed": agent.seed,
                        "episode": ep,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "counters": counters,
                        "resume_from_next_episode": start,
                        "note": "Run aborted. Never label a partially applied update as complete.",
                    },
                )
                raise
            counters["successful_updates"] += 1
            agent.episode_count += 1
            rollback = False
            if len(agent.recent_R) >= 30:
                avg_R = float(np.mean(agent.recent_R))
                vb = float(np.mean(agent.recent_Vmin_b)) if agent.recent_Vmin_b else 1.1
                va = float(np.mean(agent.recent_Vmin_a)) if agent.recent_Vmin_a else 1.1
                a2b = float(np.mean(agent.recent_a2_b)) if agent.recent_a2_b else 0.0
                a2a = float(np.mean(agent.recent_a2_a)) if agent.recent_a2_a else 0.0
                if not agent.arm.best_safe:
                    if avg_R > agent.best_R_avg:
                        agent.best_R_avg = avg_R
                        agent._save_snapshot()
                        agent.snapshot_ep = ep
                else:
                    if (
                        avg_R > agent.best_R_avg
                        and vb >= 0.5 * env.V_min_target
                        and (va >= 0.5 * env.V_min_target)
                    ):
                        agent.best_R_avg = avg_R
                        agent._save_snapshot()
                        agent.snapshot_ep = ep
                    if agent.best_R_avg > 0:
                        agent.best_R_avg *= 0.9985
                    if agent.arm.criticality_aware:
                        safe_ok = (
                            stats["cvar_j"] < agent.c_limit_joint
                            and stats["lam"] < 5.0
                            and (va >= 0.68)
                            and (a2b <= 0.02)
                        )
                        score = (
                            avg_R
                            + 0.1 * max(0.0, agent.c_limit_joint - stats["cvar_j"])
                            + 2.0 * max(0.0, va - 0.68)
                        )
                    else:
                        safe_ok = stats["cvar_j"] < agent.c_limit_joint and stats["lam"] < 5.0
                        score = avg_R + 0.1 * max(0.0, agent.c_limit_joint - stats["cvar_j"])
                    if safe_ok and score > agent.best_safe_score:
                        agent.best_safe_score, agent.best_safe_ep = (score, ep)
                        agent._save_safe_checkpoint(
                            ep,
                            {
                                "avg_R": avg_R,
                                "avg_Vmin_b": vb,
                                "avg_Vmin_a": va,
                                "avg_a2_b": a2b,
                                "avg_a2_a": a2a,
                                **{
                                    k: stats[k]
                                    for k in ("cvar_b", "cvar_a", "cvar_j", "lam", "lam_raw")
                                },
                                "R_ep": R,
                                "C_ep": C,
                                "V_drop_ep": risk,
                                "V_min_ep": vmin,
                            },
                        )
                    cooled = ep - agent.last_rollback_ep >= agent.rollback_cooldown
                    big_drop = (
                        agent.best_R_avg > 5.0
                        and avg_R < agent.best_R_avg - agent.underperf_threshold
                    )
                    if big_drop and cooled:
                        agent.underperf_streak += 1
                        if agent.underperf_streak >= 2:
                            agent._load_snapshot()
                            agent.last_rollback_ep, agent.underperf_streak = (ep, 0)
                            agent.best_R_avg = max(avg_R, agent.best_R_avg - 5.0)
                            agent.pid.clear(hard=True)
                            agent.sat_streak = 0
                            agent.clear_cost_buffers()
                            rollback = True
                            counters["rollbacks"] += 1
                    else:
                        agent.underperf_streak = max(0, agent.underperf_streak - 1)
            row = {
                "dataset": DATASET,
                "variant": agent.arm.name,
                "seed": agent.seed,
                "episode": ep,
                "native_reward": R,
                "native_cost": C,
                "native_risk": risk,
                "native_vmin": vmin,
                "reference_reward": env.reference_reward_sum,
                "reference_cost": env.reference_cost_sum,
                "reference_risk": info["reference_risk"],
                "reference_vmin": info["reference_vmin"],
                "malicious": int(env.is_malicious),
                "steps": sum(counts),
                "a0_ratio": counts[0] / sum(counts),
                "a1_ratio": counts[1] / sum(counts),
                "a2_ratio": a2_ratio,
                "fallback": int(fallback),
                "attack_steps": info["ep_attack_steps"],
                "intercepts": info["ep_intercepts"],
                "lambda_effective": stats["lam"],
                "lambda_raw_before_sanity": stats["lam_raw"],
                "lambda_after_sanity_and_rollback": agent.pid.lam,
                "integral": agent.pid.integral,
                "sanity_action": stats["sanity"],
                "rollback": int(rollback),
                "conditional_block_opportunity": agent.pid.diagnostics.get(
                    "conditional_block_opportunity", 0
                ),
                "integral_clip_opportunity": agent.pid.diagnostics.get(
                    "integral_clip_opportunity", 0
                ),
                "near_saturation_before": agent.pid.diagnostics.get("near_saturation_before", 0),
                "kl": stats["kl"],
                "entropy": stats["ent"],
                "best_safe_ep": agent.best_safe_ep,
                "successful_updates": counters["successful_updates"],
                "optimizer_steps": counters["optimizer_steps"],
                "rollout_seconds": rollout_sec,
                "update_seconds": time.perf_counter() - update_start,
            }
            append_csv(trace_path, row)
            if ep % args.log_every == 0 or rollback or ep == args.train_eps - 1:
                logger.info(
                    "[%s] ep=%d/%d R=%.3f risk=%.3f lambda=%.3f fb=%d sampling=%.2fs update=%.2fs",
                    agent.prefix,
                    ep + 1,
                    args.train_eps,
                    R,
                    risk,
                    agent.pid.lam,
                    fallback,
                    rollout_sec,
                    row["update_seconds"],
                )
            if (ep + 1) % args.save_every == 0 or ep == args.train_eps - 1:
                save_resume(agent, last_path, ep + 1, counters)
                start = ep + 1
    finally:
        hook.remove()
    final = agent._state_dict_for_ckpt()
    final.update(
        schema=SCHEMA,
        role="final",
        metadata=agent.metadata,
        saved_episode=args.train_eps - 1,
        safe_metrics=agent.best_safe_metrics,
        counters=counters,
    )
    atomic_torch_save(final, agent.run_dir / f"{agent.prefix}_final.pt")
    selected_role = select_training_checkpoint(agent)
    selected = agent._state_dict_for_ckpt()
    selected.update(
        schema=SCHEMA,
        role="selected",
        selected_role=selected_role,
        metadata=agent.metadata,
        safe_metrics=agent.best_safe_metrics,
        counters=counters,
    )
    atomic_torch_save(selected, agent.run_dir / f"{agent.prefix}_selected.pt")
    return (selected_role, counters)
