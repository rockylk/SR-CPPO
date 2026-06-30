import torch
import numpy as np
from src.utils.logger import logger
from src.envs.optc_env import *
from src.agents.trainer import *
import os

"""
CVaR-PPO Original Baseline Implementation
Integrated into OpTC V7.2.1 Data-Driven Environment.

Based on the original paper algorithm provided by the user:
- Standard MLP Actor-Critic
- Step-wise/Trajectory-level Empirical CVaR computation
- Standard Lagrangian dual gradient descent for lambda
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime

# =====================================================
# 0. Logger
# =====================================================
logger = setup_logger()


# =====================================================
# 1. Environment Helpers (POMDP & Survivability)
# =====================================================
def compute_var_cvar(costs, alpha=0.15):
    """
    costs: tensor [batch_size]
    Note: To constrain risk, CVaR must penalize the upper tail (highest costs).
    We use descending=True so the tail grabs the highest values.
    """
    sorted_costs, _ = torch.sort(costs, descending=True)
    idx = int(alpha * len(sorted_costs))
    idx = max(idx, 1)

    var = sorted_costs[idx]
    tail = sorted_costs[:idx]
    cvar = torch.mean(tail)

    return var, cvar


def train_original_cvar_ppo(total_eps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    benign_path = os.path.join(current_dir, "../../optc_ot_sequence_benign2.npy")
    mal_path = os.path.join(current_dir, "../../optc_ot_sequence_malicious2.npy")

    env = OpTC_DataDrivenEnv(benign_path, mal_path)
    
    # Encoder generates 64-dim belief states
    encoder = BeliefEncoder(obs_dim=16, belief_dim=64).to(device)
    
    # The Original Agent
    agent = CVaRPPOAgent(
        state_dim=64, 
        action_dim=3,
        alpha=0.15,
        cvar_threshold=env.c_limit_eff, # 13.25
        device=device
    )

    logger.info(f"[CVaR-PPO Original] Engine Started. Cvar_Threshold={agent.cvar_threshold:.2f} | Alpha={agent.alpha} | Device={device}")

    # Logging buffers
    recent_R = deque(maxlen=30)
    recent_C_a = deque(maxlen=30)
    recent_C_b = deque(maxlen=30)
    recent_Vmin_a = deque(maxlen=30)
    recent_Vmin_b = deque(maxlen=30)
    
    recent_ep_intercepts = deque(maxlen=30)
    recent_ep_attack_steps = deque(maxlen=30)

    for ep in range(total_eps):
        o, _ = env.reset()
        b = encoder.init_belief(1, device)
        
        traj = {
            "states": [], "a_types": [], "a_mus": [],
            "log_probs": [], "rewards": [], "costs": []
        }

        done = False
        a0_count = a1_count = a2_count = step_count = 0

        while not done:
            o_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                b = encoder(o_t, b)
                b_np = b.cpu().numpy()[0]
                
            # Act using Original Agent logic
            a_type, a_mu, logp = agent.act(b_np)
            
            o2, r, c, done, _, info = env.step(a_type, a_mu)

            traj["states"].append(b_np)
            traj["a_types"].append(a_type)
            traj["a_mus"].append(a_mu)
            traj["log_probs"].append(logp)
            traj["rewards"].append(r)
            traj["costs"].append(c)

            step_count += 1
            if a_type == 0: a0_count += 1
            elif a_type == 1: a1_count += 1
            else: a2_count += 1

            o = o2

        # Episode metrics
        R_ep = float(np.sum(traj["rewards"]))
        C_ep = float(np.sum(traj["costs"]))
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

        # Update Agent using Trajectory (Original Logic)
        stats = agent.update([traj])

        # Logging matching standard format for parsing
        if ep % 10 == 0:
            env_type = "[ATTACK]" if env.is_malicious else "[BENIGN]"
            
            avg_R = float(np.mean(recent_R)) if recent_R else 0.0
            avg_C_b = float(np.mean(recent_C_b)) if recent_C_b else 0.0
            avg_C_a = float(np.mean(recent_C_a)) if recent_C_a else 0.0
            avg_Vmin_b = float(np.mean(recent_Vmin_b)) if recent_Vmin_b else 1.0
            avg_Vmin_a = float(np.mean(recent_Vmin_a)) if recent_Vmin_a else 1.0

            a_mix = f"{a0_count/step_count:.2f}/{a1_count/step_count:.2f}/{a2_count/step_count:.2f}"
            
            total_atk = sum(recent_ep_attack_steps)
            air = (sum(recent_ep_intercepts) / total_atk * 100.0) if total_atk > 0 else 0.0

            logger.info(
                f"Ep {ep:4d} {env_type:8s} | "
                f"R={R_ep:6.2f} avgR={avg_R:6.2f} | "
                f"Vmin(b/a)={avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} "
                f"Vdrop={info['V_drop_ep']:5.1f} | "
                f"Cb={avg_C_b:5.1f} Ca={avg_C_a:6.1f} | "
                f"CVaR(b/a/J)=0.0/0.0/{stats['cvar']:5.2f}(lim={agent.cvar_threshold:.2f}) | "
                f"λ={stats['lambda']:5.2f}/100 raw=0.0 sat= 0 | "
                f"a012={a_mix} bA2avg=0.0 | "
                f"AIR={air:.1f}% gate(on=0,blk=0.0,fb=0) | "
                f"KL=0.0 ent=0.0 | phs=0.0 atk=0.0 | β=0.0 ec=0.0 aw=0.0 | lr={agent.optimizer.param_groups[0]['lr']:.2e} bestSafeEp=0"
            )

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_original_cvar_ppo(total_eps=1500)