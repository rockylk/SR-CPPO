import torch
import numpy as np
import time
import os
from collections import deque
from src.utils.logger import logger
from src.utils.math_utils import *
from src.models.networks import *

class CVaRPPOAgent:
    def __init__(self, state_dim, action_dim=3, alpha=0.15, lr=3e-4, gamma=0.99, lam_lr=0.01, cvar_threshold=13.25, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.alpha = alpha
        self.cvar_threshold = cvar_threshold
        self.lam = torch.tensor(1.0, requires_grad=False).to(device)
        self.lam_lr = lam_lr
        self.gamma = gamma

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            type_probs, mu_mean = self.actor(state)
            
            type_dist = torch.distributions.Categorical(type_probs)
            a_type = type_dist.sample()
            
            idx = a_type.unsqueeze(-1)
            mu_m = mu_mean.gather(-1, idx).squeeze(-1)
            mu_std = self.actor.mu_logstd[a_type].exp()
            mu_dist = torch.distributions.Normal(mu_m, mu_std)
            a_mu = mu_dist.sample().clamp(1e-4, 1.0 - 1e-4)
            
            log_prob = type_dist.log_prob(a_type) + mu_dist.log_prob(a_mu)
            
        return a_type.item(), a_mu.item(), log_prob.item()

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, trajectories):
        all_policy_loss = []
        all_value_loss = []
        all_costs = []

        for traj in trajectories:
            states = torch.FloatTensor(np.array(traj["states"])).to(self.device)
            a_types = torch.LongTensor(traj["a_types"]).to(self.device)
            a_mus = torch.FloatTensor(traj["a_mus"]).to(self.device)
            old_log_probs = torch.FloatTensor(traj["log_probs"]).to(self.device)

            rewards = traj["rewards"]
            costs = traj["costs"]

            # Returns
            returns = self.compute_returns(rewards)
            cost_returns = self.compute_returns(costs)
            
            values = self.critic(states).squeeze()

            # (Fix applied here): Subtract lambda * cost_returns to make constraint differentiable
            advantages = (returns - self.lam * cost_returns) - values.detach()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO Policy Loss
            type_probs, mu_mean = self.actor(states)
            type_dist = torch.distributions.Categorical(type_probs)
            
            idx = a_types.unsqueeze(-1)
            mu_m = mu_mean.gather(-1, idx).squeeze(-1)
            mu_std = self.actor.mu_logstd[a_types].exp()
            mu_dist = torch.distributions.Normal(mu_m, mu_std)
            
            log_probs = type_dist.log_prob(a_types) + mu_dist.log_prob(a_mus)
            ratio = torch.exp(log_probs - old_log_probs)

            clip_adv = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

            # Value Loss
            value_loss = (values - returns).pow(2).mean()

            all_policy_loss.append(policy_loss)
            all_value_loss.append(value_loss)
            all_costs.append(cost_returns)

        # CVaR Computation (Trajectory-level)
        all_costs = torch.cat(all_costs)
        var, cvar = compute_var_cvar(all_costs, self.alpha)

        # Lagrangian Loss (Original code logic preserved, but effect applied via advantage)
        cvar_constraint = cvar - self.cvar_threshold
        lagrangian_loss = self.lam * cvar_constraint

        total_loss = (
            torch.stack(all_policy_loss).mean()
            + 0.5 * torch.stack(all_value_loss).mean()
            + lagrangian_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Dual Update (IMPORTANT - paper core)
        self.lam = torch.clamp(
            self.lam + self.lam_lr * cvar_constraint.detach(),
            min=0.0,
            max=100.0
        )

        return {
            "policy_loss": torch.stack(all_policy_loss).mean().item(),
            "value_loss": torch.stack(all_value_loss).mean().item(),
            "cvar": cvar.item(),
            "var": var.item(),
            "lambda": self.lam.item()
        }


# =====================================================
# Main Training Loop
# =====================================================
