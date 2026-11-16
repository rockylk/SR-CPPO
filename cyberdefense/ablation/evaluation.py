"""OpTC component ablations of SR-CPPO.
Frozen-policy rollout evaluation and episode-level metrics."""

import numpy as np
import torch
from collections import deque

from .config import DATASET, condition_tag
from .io import digest_object, write_csv
from .risk import compute_pooled_cvar_exact, compute_unweighted_cvar_exact
from .runtime import set_global_seed


def evaluation(env, agent, args, endpoint):
    set_global_seed(agent.seed)
    agent.encoder.eval()
    agent.actor.eval()
    buffers = {k: deque(maxlen=80) for k in ("b", "m")}
    recent_m = deque(maxlen=10)
    episodes, event_rows = ([], [])
    schedule_records = []
    for ep in range(args.eval_eps):
        env.schedule = (agent.seed, ep) if args.eval_protocol == "indexed" else None
        o, _ = env.reset()
        scenario = {
            "episode": ep,
            "malicious": bool(env.is_malicious),
            "start_idx": int(env.start_idx),
            "initial_phase": int(env.attack_phase),
        }
        schedule_records.append(scenario)
        fallback = False
        reasons = (False, False, False)
        if agent.arm.fallback and recent_m:
            cb, cm = (
                compute_unweighted_cvar_exact(buffers[k], alpha=agent.alpha) for k in ("b", "m")
            )
            rj = (1 - env.attack_rate) * cb + env.attack_rate * cm
            reasons = (
                np.mean(recent_m) > 0.45 * env.c_limit_attack
                or np.max(recent_m) > 0.75 * env.c_limit_attack,
                cm > 0.7 * env.c_limit_attack,
                rj > 0.8 * env.c_limit_eff,
            )
            fallback = bool(any(reasons))
        b = agent.encoder.init_belief(1, agent.device)
        counts = [0, 0, 0]
        low_conf_steps = a1_open_steps = 0
        done = False
        with torch.no_grad():
            while not done:
                b = agent.encoder(
                    torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0), b
                )
                pl, ap = (agent.encoder.phase_logits(b), agent.encoder.attack_prob(b))
                if args.confidence_cap is not None:
                    ap = ap.clamp(max=args.confidence_cap)
                low_conf_steps += int(ap.item() < agent.actor.gate_low)
                a1_open_steps += int(
                    ap.item()
                    >= (agent.actor.gate_low_fallback if fallback else agent.actor.gate_low)
                )
                at, am, _, _ = agent.actor.act(
                    b, pl, ap, gate_active=True, a1_fallback_active=fallback
                )
                counts[at.item()] += 1
                o, _, _, done, _, info = env.step(at.item(), am.item())
        regime = "m" if env.is_malicious else "b"
        buffers[regime].append(info["V_drop_ep"])
        if env.is_malicious:
            recent_m.append(info["V_drop_ep"])
        row = {
            "dataset": DATASET,
            "variant": agent.arm.name,
            "seed": agent.seed,
            "endpoint": endpoint,
            "eval_protocol": args.eval_protocol,
            "condition": condition_tag(args),
            **scenario,
            "steps": sum(counts),
            "native_reward": env.native_reward_sum,
            "reference_reward": env.reference_reward_sum,
            "native_risk": info["V_drop_ep"],
            "reference_risk": info["reference_risk"],
            "native_vmin": info["V_min_ep"],
            "reference_vmin": info["reference_vmin"],
            "a0_ratio": counts[0] / sum(counts),
            "a1_ratio": counts[1] / sum(counts),
            "a2_ratio": counts[2] / sum(counts),
            "fallback": int(fallback),
            "fb_recent": int(reasons[0]),
            "fb_mal_cvar": int(reasons[1]),
            "fb_joint": int(reasons[2]),
            "low_conf_steps": low_conf_steps,
            "a1_open_steps": a1_open_steps,
            "attack_steps": info["ep_attack_steps"],
            "intercepts": info["ep_intercepts"],
        }
        episodes.append(row)
        if args.save_step_traces:
            event_rows.extend(
                (
                    {
                        "dataset": DATASET,
                        "variant": agent.arm.name,
                        "seed": agent.seed,
                        "endpoint": endpoint,
                        "episode": ep,
                        **event,
                    }
                    for event in env.trace
                )
            )
    env.schedule = None

    def aggregate(prefix):
        benign = [r for r in episodes if not r["malicious"]]
        malicious = [r for r in episodes if r["malicious"]]
        if not benign or not malicious:
            raise ValueError("Evaluation requires both regimes; increase --eval-eps")
        cb = compute_unweighted_cvar_exact([r[prefix + "risk"] for r in benign], agent.alpha)
        cm = compute_unweighted_cvar_exact([r[prefix + "risk"] for r in malicious], agent.alpha)
        mix = compute_pooled_cvar_exact(
            [r[prefix + "risk"] for r in benign],
            [r[prefix + "risk"] for r in malicious],
            agent.attack_rate,
            agent.alpha,
        )
        attack_steps = sum((r["attack_steps"] for r in malicious))
        if attack_steps == 0:
            raise ValueError("No active attack steps; AIR cannot be estimated")
        return {
            "Avg_Reward": float(np.mean([r[prefix + "reward"] for r in episodes])),
            "R_J": (1 - agent.attack_rate) * cb + agent.attack_rate * cm,
            "CVaR_B": cb,
            "CVaR_M": cm,
            "CVaR_mix": mix,
            "Vmin_A": float(np.mean([r[prefix + "vmin"] for r in malicious])),
            "Vmin_B": float(np.mean([r[prefix + "vmin"] for r in benign])),
            "FDR_pct": 100.0 * float(np.mean([r["a2_ratio"] for r in benign])),
            "AIR_pct": 100.0 * sum((r["intercepts"] for r in malicious)) / attack_steps,
        }

    reference, native = (aggregate("reference_"), aggregate("native_"))
    if reference["R_J"] > reference["CVaR_mix"] + 1e-08:
        raise AssertionError("Common-reference stratified risk exceeds mixture CVaR")
    write_csv(agent.run_dir / f"{agent.prefix}_{endpoint}_episodes.csv", episodes)
    if event_rows:
        write_csv(agent.run_dir / f"{agent.prefix}_{endpoint}_steps.csv", event_rows)
    return {
        **reference,
        **{"native_" + k: v for k, v in native.items()},
        "eval_episodes": len(episodes),
        "benign_episodes": sum((not r["malicious"] for r in episodes)),
        "malicious_episodes": sum((r["malicious"] for r in episodes)),
        "fallback_episodes": sum((r["fallback"] for r in episodes)),
        "scenario_sha256": digest_object(schedule_records),
        "schedule_rule": (
            "SeedSequence(seed,episode,step,event); two step-event slots"
            if args.eval_protocol == "indexed"
            else "original global NumPy stream"
        ),
        "best_safe_ep": agent.best_safe_ep,
    }
