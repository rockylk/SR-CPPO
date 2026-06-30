import torch
import numpy as np
from src.utils.logger import logger
from src.envs.optc_env import *
from src.agents.trainer import *
import os

"""
Responsive Safety PPO (PID Lagrangian) - Original Strict Version
Based on: "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods" (Stooke et al., 2020)

Integrated into OpTC V7.2.1 Data-Driven Environment.
Includes Critical Stability Fixes for NaN, LogStd Overflow, and Best-Safe Checkpoint Logging.
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
# 0. Logger Setup
# =====================================================
logger = setup_logger()

# =====================================================
# 1. Environment Helpers (POMDP & Survivability)
# =====================================================
class PIDController:
    def __init__(self, kp=1.0, ki=0.05, kd=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        """ λ_t update = PID(error). Output bounded to avoid infinity. """
        self.integral = max(0.0, min(self.integral + error, 1000.0))
        derivative = error - self.prev_error
        self.prev_error = error

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        return max(0.0, min(output, 30.0))

# =========================================================
# 4. Responsive Safety PPO Agent
# =========================================================
def train_strict_pid_ppo(total_eps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    env = OpTC_DataDrivenEnv(os.path.join(current_dir, "../../optc_ot_sequence_benign2.npy"), 
                             os.path.join(current_dir, "../../optc_ot_sequence_malicious2.npy"))
    encoder = BeliefEncoder(obs_dim=16, belief_dim=64).to(device)
    
    agent = StrictPIDResponsivePPO(
        state_dim=64, 
        action_dim=3,
        cost_limit=env.c_limit_joint, 
        device=device
    )

    logger.info(f"[Strict CPPO-PID] Engine Started. Cost_Limit={agent.cost_limit:.2f} | PID(Kp={agent.pid.kp}, Ki={agent.pid.ki}, Kd={agent.pid.kd}) | Device={device}")

    recent_R = deque(maxlen=30)
    recent_C_a = deque(maxlen=30)
    recent_C_b = deque(maxlen=30)
    recent_Vmin_a = deque(maxlen=30)
    recent_Vmin_b = deque(maxlen=30)
    recent_ep_intercepts = deque(maxlen=30)
    recent_ep_attack_steps = deque(maxlen=30)

    # -------------------------------------------------
    # FIX: Add Best-Safe Checkpoint Tracking
    # -------------------------------------------------
    best_safe_score = -1e9
    best_safe_ep = -1
    safe_ckpt_path = os.path.join(current_dir, "cppo_pid_strict_best_safe.pt")
    
    batch_trajectories = []

    for ep in range(total_eps):
        o, _ = env.reset()
        b = encoder.init_belief(1, device)
        
        traj = {
            "states": [], "a_types": [], "a_mus": [],
            "log_probs": [], "rewards": [], "costs": [], "dones": []
        }

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
            traj["costs"].append(c)
            traj["dones"].append(float(done))

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

        # Batch collection logic (Wait for 5 episodes to update to prevent short-trajectory instability)
        batch_trajectories.append(traj)
        avg_cost = C_ep 
        lam = agent.lam
        
        if len(batch_trajectories) >= 5:
            avg_cost, lam = agent.update(batch_trajectories)
            batch_trajectories = [] 

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
            
            # -------------------------------------------------
            # FIX: Determine Safe Checkpoint and Save physically
            # -------------------------------------------------
            a2_ratio = a2_count / max(1, step_count) if not env.is_malicious else 0.0
            safe_ok = (avg_cost < agent.cost_limit and avg_Vmin_a >= 0.68 and a2_ratio <= 0.02)
            safe_score = avg_R + 0.10 * max(0, agent.cost_limit - avg_cost) + 2.0 * max(0, avg_Vmin_a - 0.68)
            
            if safe_ok and safe_score > best_safe_score and ep > 20:
                best_safe_score = safe_score
                best_safe_ep = ep
                # Physical Save!
                torch.save(agent.actor.state_dict(), safe_ckpt_path)
                logger.info(f"[Ep {ep}] *** Best-Safe checkpoint saved: path={safe_ckpt_path}, score={safe_score:.2f}, avgR={avg_R:.2f}, CVaR_j={avg_cost:.2f}, lam={lam:.2f}, Vmin_a={avg_Vmin_a:.2f}, benign_a2={a2_ratio:.3f} ***")

            logger.info(
                f"Ep {ep:4d} {env_type:8s} | "
                f"R={R_ep:6.2f} avgR={avg_R:6.2f} | "
                f"Vmin(b/a)={avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} "
                f"Vdrop={info['V_drop_ep']:5.1f} | "
                f"Cb={avg_C_b:5.1f} Ca={avg_C_a:6.1f} | "
                f"CVaR(b/a/J)=0.0/0.0/{avg_cost:5.2f}(lim={agent.cost_limit:.2f}) | "
                f"λ={lam:5.2f}/30 raw=0.0 sat= 0 | "
                f"a012={a_mix} bA2avg={a2_ratio:.3f} | "
                f"AIR={air:.1f}% gate(on=0,blk=0.0,fb=0) | "
                f"KL=0.0 ent=0.0 | phs=0.0 atk=0.0 | β=0.0 ec=0.0 aw=0.0 | lr={agent.optimizer.param_groups[0]['lr']:.2e} bestSafeEp={best_safe_ep}"
            )

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_strict_pid_ppo(total_eps=1500)