"""SR-CPPO recurrent policy, distributional cost critic and PID controller.
Episode collection and training schedule."""

import numpy as np
import torch
from collections import deque

from .agent import SR_CPPO_v721
from .evaluation import evaluate_frozen_policy
from .runtime import logger, set_global_seed


def train_and_eval_single_seed_v721(
    env, seed=42, total_eps=1500, eval_eps=200, enable_fallback_in_eval=True, eval_only=False
):
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SR_CPPO_v721(
        obs_dim=16,
        c_limit_benign=env.c_limit_benign,
        c_limit_attack=env.c_limit_attack,
        attack_rate=env.attack_rate,
        gate_low=0.3,
        gate_high=0.7,
        gate_low_fallback=0.12,
        gate_warmup_eps=50,
        pid_start_eps=60,
        pid_ramp_eps=60,
        seed=seed,
        device=device,
    )
    if not eval_only:
        logger.info(
            f"\n=============================================================\n[OpTC V7.2.1] Starting Independent Training Run | Seed={seed} | TrainEps={total_eps} | EvalEps={eval_eps} | Device={device}\n============================================================="
        )
        recent_ep_intercepts = deque(maxlen=30)
        recent_ep_attack_steps = deque(maxlen=30)
        for ep in range(total_eps):
            buf = {
                k: []
                for k in [
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
                ]
            }
            phase_labels = []
            o, phase = env.reset()
            b = agent.encoder.init_belief(1, agent.device)
            done = False
            info = {}
            gate_active_now = agent.episode_count >= agent.gate_warmup_eps
            a1_fallback_now = agent.risk_fallback_active()
            buf["a1_fallback_active"] = bool(a1_fallback_now)
            a0_count = 0
            a1_count = 0
            a2_count = 0
            step_count = 0
            while not done:
                o_t = torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    b = agent.encoder(o_t, b)
                    pl = agent.encoder.phase_logits(b)
                    ap = agent.encoder.attack_prob(b)
                    a_type, a_mu, logp, _ = agent.actor.act(
                        b, pl, ap, gate_active=gate_active_now, a1_fallback_active=a1_fallback_now
                    )
                o2, r, c, done, phase2, info = env.step(a_type.item(), a_mu.item())
                buf["obs"].append(o.copy())
                buf["beliefs"].append(b.detach().cpu().numpy()[0])
                buf["phase_logits"].append(pl.detach().cpu().numpy()[0])
                buf["attack_prob"].append(float(ap.item()))
                buf["a_type"].append(a_type.item())
                buf["a_mu"].append(a_mu.item())
                buf["logp_old"].append(logp.item())
                buf["rewards"].append(r)
                buf["costs"].append(c)
                buf["dones"].append(float(done))
                phase_labels.append(phase)
                step_count += 1
                if a_type.item() == 0:
                    a0_count += 1
                elif a_type.item() == 1:
                    a1_count += 1
                else:
                    a2_count += 1
                o = o2
                phase = phase2
            R_ep = float(np.sum(buf["rewards"]))
            C_ep = float(np.sum(buf["costs"]))
            V_drop_ep = float(info.get("V_drop_ep", 0.0))
            V_min_ep = float(info.get("V_min_ep", 1.1))
            ep_attack_steps = int(info.get("ep_attack_steps", 0))
            ep_intercepts = int(info.get("ep_intercepts", 0))
            recent_ep_attack_steps.append(ep_attack_steps)
            recent_ep_intercepts.append(ep_intercepts)
            a0_ratio = a0_count / max(1, step_count)
            a1_ratio = a1_count / max(1, step_count)
            a2_ratio = a2_count / max(1, step_count)
            agent.recent_R.append(R_ep)
            agent.recent_C.append(C_ep)
            if env.is_malicious:
                agent.recent_C_a.append(C_ep)
                agent.recent_Vmin_a.append(V_min_ep)
                agent.cost_buf_mal.append(V_drop_ep)
                agent.cost_recent_mal.append(V_drop_ep)
                agent.recent_a2_a.append(a2_ratio)
            else:
                agent.recent_C_b.append(C_ep)
                agent.recent_Vmin_b.append(V_min_ep)
                agent.cost_buf_benign.append(V_drop_ep)
                agent.cost_recent_benign.append(V_drop_ep)
                agent.recent_a2_b.append(a2_ratio)
            try:
                stats = agent.update(buf, phase_labels)
            except Exception as e:
                logger.info(f"[Seed {seed} | Ep {ep}] Update Skip: {e}")
                agent.episode_count += 1
                continue
            agent.episode_count += 1
            rollback_fired = False
            if len(agent.recent_R) >= 30:
                avg_R = float(np.mean(agent.recent_R))
                avg_Vmin_b = float(np.mean(agent.recent_Vmin_b)) if agent.recent_Vmin_b else 1.1
                avg_Vmin_a = float(np.mean(agent.recent_Vmin_a)) if agent.recent_Vmin_a else 1.1
                avg_a2_b = float(np.mean(agent.recent_a2_b)) if agent.recent_a2_b else 0.0
                avg_a2_a = float(np.mean(agent.recent_a2_a)) if agent.recent_a2_a else 0.0
                sla_b = avg_Vmin_b >= 0.5 * env.V_min_target
                sla_a = avg_Vmin_a >= 0.5 * env.V_min_target
                if avg_R > agent.best_R_avg and sla_b and sla_a:
                    agent.best_R_avg = avg_R
                    agent._save_snapshot()
                if agent.best_R_avg > 0.0:
                    agent.best_R_avg *= 0.9985
                safe_ok = (
                    stats["cvar_j"] < agent.c_limit_joint
                    and stats["lam"] < 5.0
                    and (avg_Vmin_a >= 0.68)
                    and (avg_a2_b <= 0.02)
                )
                safety_margin = max(0.0, agent.c_limit_joint - stats["cvar_j"])
                safe_score = avg_R + 0.1 * safety_margin + 2.0 * max(0.0, avg_Vmin_a - 0.68)
                if safe_ok and safe_score > agent.best_safe_score:
                    agent.best_safe_score = safe_score
                    agent.best_safe_ep = ep
                    agent._save_safe_checkpoint(
                        ep,
                        {
                            "avg_R": avg_R,
                            "avg_Vmin_b": avg_Vmin_b,
                            "avg_Vmin_a": avg_Vmin_a,
                            "avg_a2_b": avg_a2_b,
                            "avg_a2_a": avg_a2_a,
                            "cvar_b": stats["cvar_b"],
                            "cvar_a": stats["cvar_a"],
                            "cvar_j": stats["cvar_j"],
                            "lam": stats["lam"],
                            "lam_raw": stats["lam_raw"],
                            "R_ep": R_ep,
                            "C_ep": C_ep,
                            "V_drop_ep": V_drop_ep,
                            "V_min_ep": V_min_ep,
                        },
                    )
                cooled = ep - agent.last_rollback_ep >= agent.rollback_cooldown
                big_drop = (
                    agent.best_R_avg > 5.0 and avg_R < agent.best_R_avg - agent.underperf_threshold
                )
                if big_drop and cooled:
                    agent.underperf_streak += 1
                    if agent.underperf_streak >= 2:
                        logger.info(
                            f"[Seed {seed} | Ep {ep}] *** Rollback to best snapshot (avgR {avg_R:.2f} << best {agent.best_R_avg:.2f}) ***"
                        )
                        agent._load_snapshot()
                        agent.last_rollback_ep = ep
                        agent.underperf_streak = 0
                        agent.best_R_avg = max(avg_R, agent.best_R_avg - 5.0)
                        agent.pid.clear(hard=True)
                        agent.sat_streak = 0
                        agent.clear_cost_buffers()
                        rollback_fired = True
                else:
                    agent.underperf_streak = max(0, agent.underperf_streak - 1)
            if ep % 10 == 0 or rollback_fired:
                env_type = "[ATTACK]" if env.is_malicious else "[BENIGN]"
                avg_R = float(np.mean(agent.recent_R)) if agent.recent_R else 0.0
                avg_C_b = float(np.mean(agent.recent_C_b)) if agent.recent_C_b else 0.0
                avg_C_a = float(np.mean(agent.recent_C_a)) if agent.recent_C_a else 0.0
                avg_Vmin_b = float(np.mean(agent.recent_Vmin_b)) if agent.recent_Vmin_b else 1.1
                avg_Vmin_a = float(np.mean(agent.recent_Vmin_a)) if agent.recent_Vmin_a else 1.1
                avg_a2_b = float(np.mean(agent.recent_a2_b)) if agent.recent_a2_b else 0.0
                a_mix = f"{a0_ratio:.2f}/{a1_ratio:.2f}/{a2_ratio:.2f}"
                sanity_tag = "" if stats["sanity"] == "none" else f" sanity={stats['sanity']}"
                total_atk_steps = sum(recent_ep_attack_steps)
                air_percent = (
                    sum(recent_ep_intercepts) / total_atk_steps * 100.0
                    if total_atk_steps > 0
                    else 0.0
                )
                logger.info(
                    f"Seed {seed} | Ep {ep:4d} {env_type:8s} | R={R_ep:6.2f} avgR={avg_R:6.2f} | Vmin(b/a)={avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} Vdrop={V_drop_ep:5.1f} | Cb={avg_C_b:5.1f} Ca={avg_C_a:6.1f} | CVaR(b/a/J)={stats['cvar_b']:4.1f}/{stats['cvar_a']:5.1f}/{stats['cvar_j']:5.2f}(lim={agent.c_limit_joint:.2f}) | λ={stats['lam']:5.2f}/{agent.pid.lam_max:.0f} raw={stats['lam_raw']:5.2f} sat={stats['sat']:2d} | a012={a_mix} bA2avg={avg_a2_b:.3f} | AIR={air_percent:.1f}% gate(on={stats['gate_on']},blk={stats['gate_blk']:.2f},fb={stats['a1_fb']}) | KL={stats['kl']:.4f} ent={stats['ent']:.3f} | phs={stats['phase_loss']:.3f} atk={stats['attack_loss']:.3f} | β={stats['beta']:.2f} ec={stats['ec']:.3f} aw={stats['aux_w']:.2f} | lr={stats['lr']:.2e} bestSafeEp={agent.best_safe_ep}{sanity_tag}"
                )
    ckpt_source = agent.load_best_safe_or_snapshot()
    eval_metrics = evaluate_frozen_policy(
        env, agent, eval_eps=eval_eps, enable_fallback=enable_fallback_in_eval
    )
    eval_metrics["seed"] = seed
    eval_metrics["best_safe_ep"] = agent.best_safe_ep
    eval_metrics["ckpt_source"] = ckpt_source
    fb_str = "WITH_Fallback" if enable_fallback_in_eval else "WO_Fallback"
    prop2_ok = eval_metrics["R_J"] <= eval_metrics["CVaR_mix"] + 1e-09
    logger.info(
        f"\n[Seed {seed} Frozen Evaluation ({fb_str} | {ckpt_source} | bestSafeEp={agent.best_safe_ep})]\n  -> Avg Reward = {eval_metrics['Avg_Reward']:.2f}\n  -> R_J (Test) = {eval_metrics['R_J']:.2f} <= CVaR_mix (Test) = {eval_metrics['CVaR_mix']:.2f} [Risk ordering: {prop2_ok}]\n  -> CVaR_B     = {eval_metrics['CVaR_B']:.2f} (<= 5.0) | CVaR_M = {eval_metrics['CVaR_M']:.2f} (<= 20.0) | R_J_dec (Best-Safe Ckpt) = {eval_metrics['R_J_at_BestSafe_Ckpt']:.2f}\n  -> Vmin_A     = {eval_metrics['Vmin_A']:.2f} (>= 0.68) | Vmin_B = {eval_metrics['Vmin_B']:.2f}\n  -> FDR        = {eval_metrics['FDR_pct']:.2f}% (~0.00%) | AIR    = {eval_metrics['AIR_pct']:.2f}%"
    )
    return eval_metrics
