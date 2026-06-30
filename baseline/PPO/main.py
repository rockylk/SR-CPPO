import torch
import numpy as np
from src.utils.logger import logger
from src.envs.optc_env import *
from src.agents.trainer import *
import os

"""
Proximal Policy Optimization (PPO) - STRICT Original Version
Based on: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

Integrated into OpTC V7.2.1 Data-Driven Environment.
Note: This is an UNCONSTRAINED baseline. It optimizes only for Reward.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

# =====================================================
# 0. Logger Setup
# =====================================================
logger = setup_logger()

# =====================================================
# 1. Environment Helpers (Identical to SR-CPPO)
# =====================================================
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """ Fixed GAE: computed strictly inside a single trajectory """
    advantages = np.zeros_like(rewards)
    g = 0.0
    # Pad values with 0 for the terminal state
    vals = np.append(values, 0.0) 
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t+1] - vals[t]
        g = delta + gamma * lam * g
        advantages[t] = g
    return advantages

def train_strict_ppo(total_eps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    env = OpTC_DataDrivenEnv(os.path.join(current_dir, "../../optc_ot_sequence_benign2.npy"), 
                             os.path.join(current_dir, "../../optc_ot_sequence_malicious2.npy"))
    encoder = BeliefEncoder(obs_dim=16, belief_dim=64).to(device)
    
    # UNCONSTRAINED AGENT
    agent = StrictPPOAgent(state_dim=64, device=device)

    logger.info(f"[Strict PPO Schulman 2017] Started. Unconstrained Optimization. Device={device}")

    recent_R = deque(maxlen=30)
    recent_C_a = deque(maxlen=30)
    recent_C_b = deque(maxlen=30)
    recent_Vmin_a = deque(maxlen=30)
    recent_Vmin_b = deque(maxlen=30)
    recent_ep_intercepts = deque(maxlen=30)
    recent_ep_attack_steps = deque(maxlen=30)
    
    # Checkpoint tracking
    best_safe_score = -1e9
    best_safe_ep = -1
    safe_ckpt_path = os.path.join(current_dir, "ppo_strict_best_safe.pt")

    for ep in range(total_eps):
        o, _ = env.reset()
        b = encoder.init_belief(1, device)
        
        traj = {"states": [], "a_types": [], "a_mus": [], "rewards": [], "costs": [], "log_probs": []}
        done = False
        a0_count = a1_count = a2_count = step_count = 0

        while not done:
            o_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                b = encoder(o_t, b)
                b_np = b.cpu().numpy()[0]
                
            a_type, a_mu, logp = agent.act(b_np)
            o2, r, c, done, _, info = env.step(a_type, a_mu)

            traj["states"].append(b_np)
            traj["a_types"].append(a_type)
            traj["a_mus"].append(a_mu)
            traj["log_probs"].append(logp)
            traj["rewards"].append(r)
            traj["costs"].append(c) # Collected for evaluation, NOT used for constraints

            step_count += 1
            if a_type == 0: a0_count += 1
            elif a_type == 1: a1_count += 1
            else: a2_count += 1
            o = o2

        R_ep = float(np.sum(traj["rewards"]))
        C_ep = float(info["V_drop_ep"])
        V_min_ep = info["V_min_ep"]
        
        recent_R.append(R_ep)
        recent_ep_attack_steps.append(info["ep_attack_steps"])
        recent_ep_intercepts.append(info["ep_intercepts"])

        if env.is_malicious:
            recent_C_a.append(C_ep)
            recent_Vmin_a.append(V_min_ep)
        else:
            recent_C_b.append(C_ep)
            recent_Vmin_b.append(V_min_ep)

        # Standard PPO Update (No Constraints)
        stats = agent.update([traj])
        avg_cost = C_ep # Purely for tracking

        # Logging matching standard format for parsing script
        if ep % 10 == 0:
            env_type = "[ATTACK]" if env.is_malicious else "[BENIGN]"
            
            avg_R = float(np.mean(recent_R)) if recent_R else 0.0
            avg_C_b = float(np.mean(recent_C_b)) if recent_C_b else 0.0
            avg_C_a = float(np.mean(recent_C_a)) if recent_C_a else 0.0
            avg_Vmin_b = float(np.mean(recent_Vmin_b)) if recent_Vmin_b else 1.0
            avg_Vmin_a = float(np.mean(recent_Vmin_a)) if recent_Vmin_a else 1.0

            a_mix = f"{a0_count/step_count:.2f}/{a1_count/step_count:.2f}/{a2_count/step_count:.2f}"
            
            # Determine "Safe" Checkpoint (Even though PPO doesn't optimize for it, we evaluate it for the paper)
            a2_ratio = a2_count / max(1, step_count) if not env.is_malicious else 0.0
            safe_ok = (avg_cost < env.c_limit_joint and avg_Vmin_a >= 0.68 and a2_ratio <= 0.02)
            safe_score = avg_R + 0.10 * max(0, env.c_limit_joint - avg_cost) + 2.0 * max(0, avg_Vmin_a - 0.68)
            
            if safe_ok and safe_score > best_safe_score and ep > 20:
                best_safe_score = safe_score
                best_safe_ep = ep
                torch.save(agent.actor.state_dict(), safe_ckpt_path)
                logger.info(f"[Ep {ep}] *** Best-Safe checkpoint saved: path={safe_ckpt_path}, score={safe_score:.2f}, avgR={avg_R:.2f}, CVaR_j={avg_cost:.2f}, lam=0.00, Vmin_a={avg_Vmin_a:.2f}, benign_a2={a2_ratio:.3f} ***")

            total_atk = sum(recent_ep_attack_steps)
            air = (sum(recent_ep_intercepts) / total_atk * 100.0) if total_atk > 0 else 0.0

            logger.info(
                f"Ep {ep:4d} {env_type:8s} | "
                f"R={R_ep:6.2f} avgR={avg_R:6.2f} | "
                f"Vmin(b/a)={avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} "
                f"Vdrop={info['V_drop_ep']:5.1f} | "
                f"Cb={avg_C_b:5.1f} Ca={avg_C_a:6.1f} | "
                f"CVaR(b/a/J)=0.0/0.0/{avg_cost:5.2f}(lim={env.c_limit_joint:.2f}) | "
                f"λ= 0.00/30 raw=0.0 sat= 0 | "
                f"a012={a_mix} bA2avg={a2_ratio:.3f} | "
                f"AIR={air:.1f}% gate(on=0,blk=0.0,fb=0) | "
                f"KL={stats['kl']:.4f} ent={stats['entropy']:.3f} | phs=0.0 atk=0.0 | β=0.0 ec=0.0 aw=0.0 | lr={agent.optimizer.param_groups[0]['lr']:.2e} bestSafeEp={best_safe_ep}"
            )

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_strict_ppo(total_eps=1500)