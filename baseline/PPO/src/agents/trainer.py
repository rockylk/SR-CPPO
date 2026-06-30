import torch
import numpy as np
import time
import os
from collections import deque
from src.utils.logger import logger
from src.utils.math_utils import *
from src.models.networks import *

class StrictPPOAgent:
    def __init__(self, state_dim, action_dim=3, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, ent_coef=0.01, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef

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
        return np.array(returns, dtype=np.float32)

    def update(self, trajectories, epochs=10, batch_size=64):
        all_states, all_a_types, all_a_mus = [], [], []
        all_returns, all_advantages, all_old_logps = [], [], []

        # -------------------------------------------------
        # FIX: Per-Trajectory GAE to prevent cross-contamination
        # -------------------------------------------------
        for traj in trajectories:
            states_np = np.array(traj["states"])
            states_t = torch.FloatTensor(states_np).to(self.device)
            rewards_np = np.array(traj["rewards"])
            
            with torch.no_grad():
                values_np = self.critic(states_t).cpu().numpy()
            
            returns = self.compute_returns(rewards_np)
            advantages = compute_gae(rewards_np, values_np, self.gamma, self.lam)

            all_states.append(states_np)
            all_a_types.append(traj["a_types"])
            all_a_mus.append(traj["a_mus"])
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_old_logps.append(traj["log_probs"])

        # Flatten
        states = torch.FloatTensor(np.concatenate(all_states)).to(self.device)
        a_types = torch.LongTensor(np.concatenate(all_a_types)).to(self.device)
        a_mus = torch.FloatTensor(np.concatenate(all_a_mus)).to(self.device)
        returns = torch.FloatTensor(np.concatenate(all_returns)).to(self.device)
        advantages = torch.FloatTensor(np.concatenate(all_advantages)).to(self.device)
        old_logps = torch.FloatTensor(np.concatenate(all_old_logps)).to(self.device)

        # Advantage Normalization (Crucial PPO Step)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -------------------------------------------------
        # PPO Multi-Epoch Mini-Batch SGD
        # -------------------------------------------------
        dataset_size = states.shape[0]
        final_policy_loss, final_val_loss, final_entropy = 0, 0, 0
        total_batches = 0

        for _ in range(epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                batch_idx = idx[start:start+batch_size]

                b_states = states[batch_idx]
                b_a_types = a_types[batch_idx]
                b_a_mus = a_mus[batch_idx]
                b_old_logp = old_logps[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Policy Objective
                type_probs, mu_mean = self.actor(b_states)
                type_dist = torch.distributions.Categorical(type_probs)
                
                idx_unsqueeze = b_a_types.unsqueeze(-1)
                mu_m = mu_mean.gather(-1, idx_unsqueeze).squeeze(-1)
                mu_std = self.actor.mu_logstd[b_a_types].exp()
                mu_dist = torch.distributions.Normal(mu_m, mu_std)
                
                logp = type_dist.log_prob(b_a_types) + mu_dist.log_prob(b_a_mus)
                entropy = type_dist.entropy().mean() + mu_dist.entropy().mean()

                ratio = torch.exp(logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Objective
                value = self.critic(b_states)
                value_loss = F.mse_loss(value, b_returns)

                # Total Loss
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

                final_policy_loss += policy_loss.item()
                final_val_loss += value_loss.item()
                final_entropy += entropy.item()
                total_batches += 1

        # Diagnostic KL
        with torch.no_grad():
            type_probs, mu_mean = self.actor(states)
            type_dist = torch.distributions.Categorical(type_probs)
            idx_unsqueeze = a_types.unsqueeze(-1)
            mu_m = mu_mean.gather(-1, idx_unsqueeze).squeeze(-1)
            mu_std = self.actor.mu_logstd[a_types].exp()
            mu_dist = torch.distributions.Normal(mu_m, mu_std)
            new_logp = type_dist.log_prob(a_types) + mu_dist.log_prob(a_mus)
            kl = (old_logps - new_logp).mean().item()

        return {
            "policy_loss": final_policy_loss / total_batches,
            "value_loss": final_val_loss / total_batches,
            "entropy": final_entropy / total_batches,
            "kl": kl
        }

# =====================================================
# 4. Main Training Loop
# =====================================================
