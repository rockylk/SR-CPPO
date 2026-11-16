"""PPO. Reference: Schulman et al. (2017), Proximal Policy Optimization Algorithms.
Policy optimization and checkpoint selection."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .models import BeliefEncoder, HybridActor, ScalarCritic, hybrid_log_prob


class StrictPPOAgent:

    def __init__(
        self,
        obs_dim=16,
        state_dim=64,
        action_dim=3,
        lr=0.0002,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        seed=42,
        device="cpu",
    ):
        self.device = device
        self.seed = seed
        self.encoder = BeliefEncoder(obs_dim, state_dim).to(device)
        self.actor = HybridActor(state_dim, action_dim).to(device)
        self.critic = ScalarCritic(state_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.best_state = None
        self.best_R_avg = -1000000000.0

    def init_belief(self):
        return self.encoder.init_belief(1, self.device)

    def set_eval_mode(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()

    def step_action(self, obs_np, b_prev):
        o_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            b_new = self.encoder(o_t, b_prev)
            type_probs, mu_mean = self.actor(b_new)
            type_dist = torch.distributions.Categorical(type_probs)
            a_type = type_dist.sample()
            idx = a_type.unsqueeze(-1)
            mu_m = mu_mean.gather(-1, idx).squeeze(-1)
            mu_std = self.actor.get_mu_std()[a_type]
            mu_dist = torch.distributions.Normal(mu_m, mu_std)
            a_mu = mu_dist.sample().clamp(0.0001, 1.0 - 0.0001)
            log_prob = hybrid_log_prob(type_dist, mu_dist, a_type, a_mu)
        return (a_type.item(), a_mu.item(), log_prob.item(), b_new)

    def gae(self, vals, rs, ds):
        adv = np.zeros_like(rs)
        g = 0.0
        nv = 0.0
        for t in reversed(range(len(rs))):
            delta = rs[t] + self.gamma * nv * (1.0 - ds[t]) - vals[t]
            g = delta + self.gamma * self.lam * (1.0 - ds[t]) * g
            adv[t] = g
            nv = vals[t]
        return adv

    def save_best(self):
        self.best_state = {
            "enc": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
            "act": {k: v.detach().clone() for k, v in self.actor.state_dict().items()},
            "cr": {k: v.detach().clone() for k, v in self.critic.state_dict().items()},
        }

    def load_best(self):
        if self.best_state is not None:
            self.encoder.load_state_dict(self.best_state["enc"])
            self.actor.load_state_dict(self.best_state["act"])
            self.critic.load_state_dict(self.best_state["cr"])

    def update(self, traj, epochs=10):
        obs_t = torch.as_tensor(np.asarray(traj["obs"]), dtype=torch.float32, device=self.device)
        a_types = torch.as_tensor(traj["a_types"], dtype=torch.long, device=self.device)
        a_mus = torch.as_tensor(traj["a_mus"], dtype=torch.float32, device=self.device)
        old_logps = torch.as_tensor(traj["log_probs"], dtype=torch.float32, device=self.device)
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        dones = np.asarray(traj["dones"], dtype=np.float32)
        with torch.no_grad():
            b_seq0 = self.encoder.encode_sequence(obs_t)
            values_np = self.critic(b_seq0).cpu().numpy()
        adv = self.gae(values_np, rewards, dones)
        ret = adv + values_np
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        if adv_t.std() > 1e-08:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-08)
        last_kl, last_ent = (0.0, 0.0)
        for _ in range(epochs):
            b_seq = self.encoder.encode_sequence(obs_t)
            type_probs, mu_mean = self.actor(b_seq)
            type_dist = torch.distributions.Categorical(type_probs)
            idx = a_types.unsqueeze(-1)
            mu_m = mu_mean.gather(-1, idx).squeeze(-1)
            mu_std = self.actor.get_mu_std()[a_types]
            mu_dist = torch.distributions.Normal(mu_m, mu_std)
            logp = hybrid_log_prob(type_dist, mu_dist, a_types, a_mus)
            entropy = type_dist.entropy().mean() + mu_dist.entropy().mean()
            ratio = torch.exp((logp - old_logps).clamp(-20, 20))
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(self.critic(b_seq), ret_t)
            loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 0.5)
            self.optimizer.step()
            last_kl = (old_logps - logp).mean().item()
            last_ent = entropy.item()
        return {"kl": last_kl, "entropy": last_ent}
