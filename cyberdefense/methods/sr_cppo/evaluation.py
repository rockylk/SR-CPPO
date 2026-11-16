"""SR-CPPO recurrent policy, distributional cost critic and PID controller.
Frozen-policy rollout evaluation and episode-level metrics."""

import numpy as np
import torch
from collections import deque

from .risk import compute_pooled_cvar_exact, compute_unweighted_cvar_exact
from .runtime import set_global_seed


def evaluate_frozen_policy(env, agent, eval_eps=200, enable_fallback=True):
    set_global_seed(agent.seed)
    agent.encoder.eval()
    agent.actor.eval()
    eval_buf_b = deque(maxlen=80)
    eval_buf_m = deque(maxlen=80)
    eval_rec_b = deque(maxlen=10)
    eval_rec_m = deque(maxlen=10)

    def check_eval_a1_fallback():
        if not enable_fallback or len(eval_rec_m) == 0:
            return False
        rec_m = np.asarray(eval_rec_m, dtype=np.float64)
        cond_recent = (
            np.mean(rec_m) > 0.45 * env.c_limit_attack or np.max(rec_m) > 0.75 * env.c_limit_attack
        )
        cvar_m_cur = compute_unweighted_cvar_exact(eval_buf_m, alpha=agent.alpha)
        cvar_b_cur = compute_unweighted_cvar_exact(eval_buf_b, alpha=agent.alpha)
        rj_cur = (1.0 - env.attack_rate) * cvar_b_cur + env.attack_rate * cvar_m_cur
        cond_cvar_a = cvar_m_cur > 0.7 * env.c_limit_attack
        cond_joint = rj_cur > 0.8 * env.c_limit_eff
        return bool(cond_recent or cond_cvar_a or cond_joint)

    ep_rewards = []
    d_benign = []
    d_mal = []
    vmin_benign = []
    vmin_mal = []
    benign_a2_ratios = []
    total_atk_steps = 0
    total_intercepts = 0
    with torch.no_grad():
        for _ in range(eval_eps):
            o, _ = env.reset()
            b = agent.encoder.init_belief(1, agent.device)
            done = False
            info = {}
            ep_r = 0.0
            a2_count = 0
            step_count = 0
            a1_fb_now = check_eval_a1_fallback()
            while not done:
                o_t = torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0)
                b = agent.encoder(o_t, b)
                pl = agent.encoder.phase_logits(b)
                ap = agent.encoder.attack_prob(b)
                a_type, a_mu, _, _ = agent.actor.act(
                    b, pl, ap, gate_active=True, a1_fallback_active=a1_fb_now
                )
                o, r, _, done, _, info = env.step(a_type.item(), a_mu.item())
                ep_r += r
                step_count += 1
                if a_type.item() == 2:
                    a2_count += 1
            v_drop_ep = float(info.get("V_drop_ep", 0.0))
            v_min_ep = float(info.get("V_min_ep", 1.1))
            atk_steps = int(info.get("ep_attack_steps", 0))
            intercepts = int(info.get("ep_intercepts", 0))
            ep_rewards.append(ep_r)
            if env.is_malicious:
                d_mal.append(v_drop_ep)
                eval_buf_m.append(v_drop_ep)
                eval_rec_m.append(v_drop_ep)
                vmin_mal.append(v_min_ep)
                total_atk_steps += atk_steps
                total_intercepts += intercepts
            else:
                d_benign.append(v_drop_ep)
                eval_buf_b.append(v_drop_ep)
                eval_rec_b.append(v_drop_ep)
                vmin_benign.append(v_min_ep)
                benign_a2_ratios.append(a2_count / max(1, step_count))
    cvar_b_eval = compute_unweighted_cvar_exact(d_benign, alpha=agent.alpha)
    cvar_m_eval = compute_unweighted_cvar_exact(d_mal, alpha=agent.alpha)
    rj_eval = (1.0 - agent.attack_rate) * cvar_b_eval + agent.attack_rate * cvar_m_eval
    cvar_mix_eval = compute_pooled_cvar_exact(
        d_benign, d_mal, p_m=agent.attack_rate, alpha=agent.alpha
    )
    rj_ckpt_saved = float(agent.best_safe_metrics.get("cvar_j", rj_eval))
    avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    vmin_a = float(np.mean(vmin_mal)) if vmin_mal else 1.1
    vmin_b = float(np.mean(vmin_benign)) if vmin_benign else 1.1
    fdr_pct = float(np.mean(benign_a2_ratios) * 100.0) if benign_a2_ratios else 0.0
    air_pct = (
        float(total_intercepts) / float(total_atk_steps) * 100.0 if total_atk_steps > 0 else 0.0
    )
    return {
        "Avg_Reward": avg_reward,
        "R_J": rj_eval,
        "Vmin_A": vmin_a,
        "FDR_pct": fdr_pct,
        "AIR_pct": air_pct,
        "CVaR_B": cvar_b_eval,
        "CVaR_M": cvar_m_eval,
        "CVaR_mix": cvar_mix_eval,
        "Vmin_B": vmin_b,
        "R_J_at_BestSafe_Ckpt": rj_ckpt_saved,
    }
