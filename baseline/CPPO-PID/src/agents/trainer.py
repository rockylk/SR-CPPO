import torch
import numpy as np
import time
import os
from collections import deque
from src.utils.logger import logger
from src.utils.math_utils import *
from src.models.networks import *

class StrictPIDResponsivePPO:
    def __init__(self, state_dim, action_dim=3, cost_limit=13.25, gamma=0.99, lr=3e-4, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic_r = Critic(state_dim).to(device)
        self.critic_c = Critic(state_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic_r.parameters()) + list(self.critic_c.parameters()),
            lr=lr
        )

        self.cost_limit = cost_limit
        self.pid = PIDController(kp=1.0, ki=0.05, kd=0.01)
        self.gamma = gamma
        self.lam = 0.0

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            type_probs, mu_mean = self.actor(state)
            type_dist = torch.distributions.Categorical(type_probs)
            a_type = type_dist.sample()
            
            idx = a_type.unsqueeze(-1)
            mu_m = mu_mean.gather(-1, idx).squeeze(-1)
            
            # USE SAFE STD
            mu_std = self.actor.get_mu_std()[a_type]
            mu_dist = torch.distributions.Normal(mu_m, mu_std)
            a_mu = mu_dist.sample().clamp(1e-4, 1.0 - 1e-4)
            
            log_prob = type_dist.log_prob(a_type) + mu_dist.log_prob(a_mu)
            
        return a_type.item(), a_mu.item(), log_prob.item()

    def gae(self, vals, rs, ds, lam_gae=0.95):
        adv = np.zeros_like(rs)
        g = 0.0
        nv = 0.0
        for t in reversed(range(len(rs))):
            delta = rs[t] + self.gamma * nv * (1.0 - ds[t]) - vals[t]
            g = delta + self.gamma * lam_gae * (1.0 - ds[t]) * g
            adv[t] = g
            nv = vals[t]
        return adv

    def update(self, trajectories):
        all_policy_loss = []
        all_val_r_loss = []
        all_val_c_loss = []
        total_cost = 0.0

        for traj in trajectories:
            states = torch.FloatTensor(np.array(traj["states"])).to(self.device)
            a_types = torch.LongTensor(traj["a_types"]).to(self.device)
            a_mus = torch.FloatTensor(traj["a_mus"]).to(self.device)
            old_log_probs = torch.FloatTensor(traj["log_probs"]).to(self.device)
            dones = np.array(traj["dones"])

            rewards = np.array(traj["rewards"], dtype=np.float32)
            costs = np.array(traj["costs"], dtype=np.float32)
            
            total_cost += np.sum(costs)

            with torch.no_grad():
                vR = self.critic_r(states).cpu().numpy()
                vC = self.critic_c(states).cpu().numpy()

            adv_r = self.gae(vR, rewards, dones)
            adv_c = self.gae(vC, costs, dones)

            ret_r = torch.as_tensor(adv_r + vR, dtype=torch.float32, device=self.device)
            ret_c = torch.as_tensor(adv_c + vC, dtype=torch.float32, device=self.device)
            
            adv_eff = adv_r - self.lam * adv_c 
            adv_eff = torch.as_tensor(adv_eff, dtype=torch.float32, device=self.device)
            
            # -------------------------------------------------
            # FIX: Safe Advantage Normalization
            # -------------------------------------------------
            if adv_eff.numel() > 1 and adv_eff.std() > 1e-6:
                adv_eff = (adv_eff - adv_eff.mean()) / (adv_eff.std() + 1e-8)
            else:
                adv_eff = adv_eff - adv_eff.mean()

            type_probs, mu_mean = self.actor(states)
            type_dist = torch.distributions.Categorical(type_probs)
            
            idx = a_types.unsqueeze(-1)
            mu_m = mu_mean.gather(-1, idx).squeeze(-1)
            mu_std = self.actor.get_mu_std()[a_types]
            mu_dist = torch.distributions.Normal(mu_m, mu_std)
            
            log_probs = type_dist.log_prob(a_types) + mu_dist.log_prob(a_mus)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped = torch.clamp(ratio, 0.8, 1.2) * adv_eff

            policy_loss = -torch.min(ratio * adv_eff, clipped).mean()
            val_r_loss = F.mse_loss(self.critic_r(states), ret_r)
            val_c_loss = F.mse_loss(self.critic_c(states), ret_c)

            all_policy_loss.append(policy_loss)
            all_val_r_loss.append(val_r_loss)
            all_val_c_loss.append(val_c_loss)

        avg_cost = total_cost / len(trajectories)
        error = avg_cost - self.cost_limit
        self.lam = self.pid.update(error)

        loss = (
            torch.stack(all_policy_loss).mean() +
            0.5 * torch.stack(all_val_r_loss).mean() +
            0.5 * torch.stack(all_val_c_loss).mean()
        )

        # Catch remaining NaN gracefully
        if not torch.isnan(loss):
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer.step()

        return avg_cost, self.lam


# =====================================================
# 5. Main Training Loop
# =====================================================
