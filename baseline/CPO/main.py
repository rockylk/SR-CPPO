import torch
import numpy as np
from src.utils.logger import logger
from src.envs.optc_env import *
from src.agents.trainer import *
import os

"""
SR-CPPO V7.2.1: V7.2 Safe-Checkpoint + A1 Fallback + Decayed-CVaR Edition
=========================================================================
Built on V7.2.

V7.2 already shows clear improvement over V7.0:
  - stricter V_min_target = 0.65;
  - true CVaR signal instead of V7.0's blind/zero CVaR pathology;
  - low-disturbance behavior is preserved;
  - attack Vmin remains around 0.69~0.71 in stable windows.

V7.2.1 adds three targeted stability patches:

  E1  Best-Safe Checkpoint
      Do NOT save only by highest avgR.
      Save checkpoint only when:
        - avgR improves;
        - CVaR_j < c_limit_joint;
        - lambda < 5;
        - avg_Vmin_a >= 0.68;
        - benign a2 ratio <= 0.02.
      Saved as: optc_v721_best_safe.pt

  E2  Rare Attack No-Op Fix: A1 Fallback Only
      Problem observed in V7.2:
        rare attack episodes may have gate_blk≈1.0, a012≈1/0/0,
        causing high Vdrop because hard gate blocks a1.
      Fix:
        - DO NOT relax a2;
        - a2 still requires attack_prob >= 0.70;
        - when recent attack risk is high, allow a1 with a lower fallback
          threshold gate_low_fallback=0.12;
        - this is rollout-level and stored in batch, so PPO evaluate uses
          the same support as act().

  E3  Decayed / Recent-Dominant CVaR
      Problem observed in V7.2:
        old extreme benign tail may reappear and push CVaR_b high later.
      Fix:
        - compute main-buffer CVaR and recent-buffer CVaR separately;
        - benign CVaR is dominated by recent buffer;
        - malicious CVaR also uses recent buffer but retains more main memory.
      Physical buffers remain separate; no physical merge.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from collections import deque
from datetime import datetime


# =====================================================
# 0. Logger
# =====================================================
logger = setup_logger()


# =====================================================
# 1. Basic Components
# =====================================================
def train_data_driven_v721(total_eps=1500):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    benign_path = os.path.join(current_dir, "../../optc_ot_sequence_benign2.npy")
    mal_path = os.path.join(current_dir, "../../optc_ot_sequence_malicious2.npy")

    env = OpTC_DataDrivenEnv(
        benign_path,
        mal_path,
        horizon=200,
        attack_rate=0.55,
        V_threshold=0.6,
        V_min_target=0.65,
        idle_bonus=0.03,
    )

    agent = CPO_Baseline(
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
    )

    logger.info(
        f"[CPO Baseline] Engine Started. "
        f"V_thr={env.V_threshold}, V_min_tgt={env.V_min_target}, "
        f"idle_bonus={env.idle_bonus} | "
        f"C_lim(b/a/joint)={env.c_limit_benign:.1f}/{env.c_limit_attack:.1f}/"
        f"{agent.c_limit_joint:.2f} | "
        f"w(b/a)={agent.w_b:.2f}/{agent.w_a:.2f} | "
        f"lr={agent.base_lr} | "
        f"gate=hard(low={agent.actor.gate_low},high={agent.actor.gate_high},"
        f"a1_fb_low={agent.actor.gate_low_fallback}), "
        f"warmup={agent.gate_warmup_eps} | "
        f"ec_floor={agent.ent_coef_floor}, aux_w=0.6(fixed), "
        f"cost_buf={agent.cost_buf_mal.maxlen}+recent={agent.cost_recent_mal.maxlen}, "
        f"safe_ckpt={agent.safe_ckpt_path}"
    )

    from collections import deque
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

        # E2: rollout-level A1 fallback flag.
        # This flag is stored in batch and reused in update() for PPO consistency.
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
                    b,
                    pl,
                    ap,
                    gate_active=gate_active_now,
                    a1_fallback_active=a1_fallback_now,
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
        V_min_ep = float(info.get("V_min_ep", 1.0))
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
            import traceback
            logger.info(f"[Ep {ep}] Update Skip: {e}\n{traceback.format_exc()}")
            agent.episode_count += 1
            continue

        agent.episode_count += 1

        rollback_fired = False

        if len(agent.recent_R) >= 30:
            avg_R = float(np.mean(agent.recent_R))

            avg_Vmin_b = float(np.mean(agent.recent_Vmin_b)) if agent.recent_Vmin_b else 1.0
            avg_Vmin_a = float(np.mean(agent.recent_Vmin_a)) if agent.recent_Vmin_a else 1.0

            avg_a2_b = float(np.mean(agent.recent_a2_b)) if agent.recent_a2_b else 0.0
            avg_a2_a = float(np.mean(agent.recent_a2_a)) if agent.recent_a2_a else 0.0

            sla_b = avg_Vmin_b >= 0.5 * env.V_min_target
            sla_a = avg_Vmin_a >= 0.5 * env.V_min_target

            if avg_R > agent.best_R_avg and sla_b and sla_a:
                agent.best_R_avg = avg_R
                agent._save_snapshot()

            if agent.best_R_avg > 0.0:
                agent.best_R_avg *= 0.9985

            # E1: best-safe checkpoint.
            safe_ok = (
                stats["cvar_j"] < agent.c_limit_joint
                and stats["lam"] < 5.0
                and avg_Vmin_a >= 0.68
                and avg_a2_b <= 0.02
            )

            # Score mainly follows avgR, but slightly rewards stronger safety margins.
            safety_margin = max(0.0, agent.c_limit_joint - stats["cvar_j"])
            safe_score = avg_R + 0.10 * safety_margin + 2.0 * max(0.0, avg_Vmin_a - 0.68)

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

            cooled = (ep - agent.last_rollback_ep) >= agent.rollback_cooldown
            big_drop = (agent.best_R_avg > 5.0) and (
                avg_R < agent.best_R_avg - agent.underperf_threshold
            )

            if big_drop and cooled:
                agent.underperf_streak += 1

                if agent.underperf_streak >= 2:
                    logger.info(
                        f"[Ep {ep}] *** Rollback to best snapshot "
                        f"(avgR {avg_R:.2f} << best {agent.best_R_avg:.2f}) ***"
                    )

                    agent._load_snapshot()
                    agent.last_rollback_ep = ep
                    agent.underperf_streak = 0
                    agent.best_R_avg = max(avg_R, agent.best_R_avg - 5.0)

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
            avg_Vmin_b = float(np.mean(agent.recent_Vmin_b)) if agent.recent_Vmin_b else 1.0
            avg_Vmin_a = float(np.mean(agent.recent_Vmin_a)) if agent.recent_Vmin_a else 1.0
            avg_a2_b = float(np.mean(agent.recent_a2_b)) if agent.recent_a2_b else 0.0

            a_mix = f"{a0_ratio:.2f}/{a1_ratio:.2f}/{a2_ratio:.2f}"

            sanity_tag = "" if stats["sanity"] == "none" else f" sanity={stats['sanity']}"

            total_atk_steps = sum(recent_ep_attack_steps)
            air_percent = (sum(recent_ep_intercepts) / total_atk_steps * 100.0) if total_atk_steps > 0 else 0.0

            logger.info(
                f"Ep {ep:4d} {env_type:8s} | "
                f"R={R_ep:6.2f} avgR={avg_R:6.2f} | "
                f"Vmin(b/a)={avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} "
                f"Vdrop={V_drop_ep:5.1f} | "
                f"Cb={avg_C_b:5.1f} Ca={avg_C_a:6.1f} | "
                f"CVaR(b/a/J)="
                f"{stats['cvar_b']:4.1f}/{stats['cvar_a']:5.1f}/{stats['cvar_j']:5.2f}"
                f"(lim={agent.c_limit_joint:.2f}) | "
                f"ν={stats['lam']:5.2f}/∞ "
                f"raw={stats['lam_raw']:5.2f} sat={stats['sat']:2d} | "
                f"a012={a_mix} bA2avg={avg_a2_b:.3f} | "
                f"AIR={air_percent:.1f}% "
                f"gate(on={stats['gate_on']},blk={stats['gate_blk']:.2f},fb={stats['a1_fb']}) | "
                f"KL={stats['kl']:.4f} ent={stats['ent']:.3f} | "
                f"phs={stats['phase_loss']:.3f} atk={stats['attack_loss']:.3f} | "
                f"β={stats['beta']:.2f} ec={stats['ec']:.3f} aw={stats['aux_w']:.2f} | "
                f"lr={stats['lr']:.2e} bestSafeEp={agent.best_safe_ep}{sanity_tag}"
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_data_driven_v721(total_eps=1500)