"""Static threshold response heuristic. Background: Stakhanova et al. (2007), A taxonomy of intrusion response systems.
Frozen-policy rollout evaluation and episode-level metrics."""

import numpy as np
import torch

from .risk import compute_pooled_cvar_exact, compute_unweighted_cvar_exact


def evaluate_frozen_baseline(env, agent, eval_eps=200, alpha=0.85):
    agent.set_eval_mode()
    ep_rewards = []
    d_benign = []
    d_mal = []
    d_pooled = []
    vmin_benign = []
    vmin_mal = []
    benign_a2_ratios = []
    total_atk_steps = 0
    total_intercepts = 0
    with torch.no_grad():
        for _ in range(eval_eps):
            o, _ = env.reset()
            b = agent.init_belief()
            done = False
            info = {}
            ep_r = 0.0
            a2_count = 0
            step_count = 0
            while not done:
                a_type, a_mu, _, b = agent.step_action(o, b)
                o, r, _, done, _, info = env.step(a_type, a_mu)
                ep_r += r
                step_count += 1
                if a_type == 2:
                    a2_count += 1
            v_drop_ep = float(info.get("V_drop_ep", 0.0))
            v_min_ep = float(info.get("V_min_ep", 1.1))
            atk_steps = int(info.get("ep_attack_steps", 0))
            intercepts = int(info.get("ep_intercepts", 0))
            ep_rewards.append(ep_r)
            d_pooled.append(v_drop_ep)
            if env.is_malicious:
                d_mal.append(v_drop_ep)
                vmin_mal.append(v_min_ep)
                total_atk_steps += atk_steps
                total_intercepts += intercepts
            else:
                d_benign.append(v_drop_ep)
                vmin_benign.append(v_min_ep)
                benign_a2_ratios.append(a2_count / max(1, step_count))
    cvar_b_eval = compute_unweighted_cvar_exact(d_benign, alpha=alpha)
    cvar_m_eval = compute_unweighted_cvar_exact(d_mal, alpha=alpha)
    rj_eval = (1.0 - env.attack_rate) * cvar_b_eval + env.attack_rate * cvar_m_eval
    cvar_mix_eval = compute_pooled_cvar_exact(d_benign, d_mal, p_m=env.attack_rate, alpha=alpha)
    avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    vmin_a = float(np.mean(vmin_mal)) if vmin_mal else float("nan")
    vmin_b = float(np.mean(vmin_benign)) if vmin_benign else float("nan")
    fdr_pct = float(np.mean(benign_a2_ratios) * 100.0) if benign_a2_ratios else float("nan")
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
        "R_J_unweighted": rj_eval,
        "CVaR_mix": cvar_mix_eval,
        "Vmin_B": vmin_b,
    }
