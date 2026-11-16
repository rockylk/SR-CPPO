"""CPPO-PID. Reference: Stooke et al. (2020), Responsive Safety in Reinforcement Learning by PID Lagrangian Methods, ICML.
Episode collection and training schedule."""

import os
import numpy as np
import torch
from collections import deque

from .agent import StrictPIDResponsivePPO
from .evaluation import evaluate_frozen_baseline, is_false_defense_action
from .risk import compute_unweighted_cvar_exact
from .runtime import logger, set_global_seed


def train_single_seed_cppo_pid(env, seed=42, total_eps=1500, eval_eps=200):
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = StrictPIDResponsivePPO(cost_limit=env.c_limit_joint, seed=seed, device=device)
    logger.info(
        f"\n[CPPO-PID Stooke 2020 | Seed={seed}] Started (TrainEps={total_eps}, EvalEps={eval_eps}, CostLimit={agent.cost_limit:.2f}, Device={device})"
    )
    recent_R = deque(maxlen=50)
    recent_cost = deque(maxlen=50)
    cost_buf_b = deque(maxlen=80)
    cost_buf_m = deque(maxlen=80)
    recent_Vmin_a = deque(maxlen=30)
    recent_fdr_b = deque(maxlen=30)
    recent_atk = deque(maxlen=30)
    recent_int = deque(maxlen=30)
    batch_trajectories = []
    for ep in range(total_eps):
        o, _ = env.reset()
        b = agent.init_belief()
        traj = {
            "obs": [],
            "a_types": [],
            "a_mus": [],
            "log_probs": [],
            "rewards": [],
            "c_shaped": [],
            "dones": [],
            "v_drop_ep": 0.0,
        }
        done = False
        false_def_count, step_count = (0, 0)
        info = {}
        while not done:
            a_type, a_mu, logp, b = agent.step_action(o, b)
            o2, r, _, done, _, info = env.step(a_type, a_mu)
            traj["obs"].append(o.copy())
            traj["a_types"].append(a_type)
            traj["a_mus"].append(a_mu)
            traj["log_probs"].append(logp)
            traj["rewards"].append(r)
            traj["c_shaped"].append(info["c_shaped"])
            traj["dones"].append(float(done))
            step_count += 1
            if not env.is_malicious and is_false_defense_action(a_type, a_mu):
                false_def_count += 1
            o = o2
        R_ep = float(np.sum(traj["rewards"]))
        v_drop_ep = float(info.get("V_drop_ep", 0.0))
        v_min_ep = float(info.get("V_min_ep", 1.1))
        traj["v_drop_ep"] = v_drop_ep
        recent_R.append(R_ep)
        recent_cost.append(v_drop_ep)
        recent_atk.append(int(info.get("ep_attack_steps", 0)))
        recent_int.append(int(info.get("ep_intercepts", 0)))
        if env.is_malicious:
            cost_buf_m.append(v_drop_ep)
            recent_Vmin_a.append(v_min_ep)
        else:
            cost_buf_b.append(v_drop_ep)
            recent_fdr_b.append(false_def_count / max(1, step_count))
        batch_trajectories.append(traj)
        if len(batch_trajectories) >= 5 or ep == total_eps - 1:
            agent.update(batch_trajectories, epochs=8)
            batch_trajectories = []
        if ep >= int(0.6 * total_eps) and len(recent_R) >= 30:
            avg_R = float(np.mean(recent_R))
            avg_C = float(np.mean(recent_cost))
            if avg_C <= agent.cost_limit * 1.15 and avg_R > agent.best_score:
                agent.best_score = avg_R
                agent.save_best()
        if ep % 50 == 0:
            cvar_b = compute_unweighted_cvar_exact(cost_buf_b, alpha=0.85)
            cvar_m = compute_unweighted_cvar_exact(cost_buf_m, alpha=0.85)
            rj = (1.0 - env.attack_rate) * cvar_b + env.attack_rate * cvar_m
            air = sum(recent_int) / max(1, sum(recent_atk)) * 100.0
            vmin_a = float(np.mean(recent_Vmin_a)) if recent_Vmin_a else 1.1
            fdr = float(np.mean(recent_fdr_b) * 100.0) if recent_fdr_b else 0.0
            logger.info(
                f"CPPO-PID Seed {seed} | Ep {ep:4d} | avgR={np.mean(recent_R):6.2f} | R_J={rj:5.2f} | lam={agent.lam:5.2f} | Vmin_A={vmin_a:.2f} | FDR={fdr:5.2f}% | AIR={air:5.2f}%"
            )
    if agent.best_state is None:
        agent.save_best()
    agent.load_best()
    ckpt_path = os.path.join(os.getcwd(), f"cppo_pid_best_seed{seed}.pt")
    torch.save(agent.best_state, ckpt_path)
    eval_metrics = evaluate_frozen_baseline(env, agent, eval_eps=eval_eps)
    eval_metrics["seed"] = seed
    logger.info(
        f"\n[CPPO-PID Seed {seed} Frozen Evaluation]\n  -> Avg Reward = {eval_metrics['Avg_Reward']:.2f} | R_J = {eval_metrics['R_J']:.2f} (CVaR_B={eval_metrics['CVaR_B']:.2f}, CVaR_M={eval_metrics['CVaR_M']:.2f}, CVaR_mix={eval_metrics['CVaR_mix']:.2f})\n  -> Vmin_A = {eval_metrics['Vmin_A']:.2f} | Vmin_B = {eval_metrics['Vmin_B']:.2f} | FDR = {eval_metrics['FDR_pct']:.2f}% | AIR = {eval_metrics['AIR_pct']:.2f}%"
    )
    return eval_metrics
