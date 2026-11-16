"""CPPO-PID. Reference: Stooke et al. (2020), Responsive Safety in Reinforcement Learning by PID Lagrangian Methods, ICML.
Policy optimization and checkpoint selection."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .models import BeliefEncoder, HybridActor, ScalarCritic, hybrid_log_prob
from .risk import StandardPIDController, undiscounted_cost_returns


class StrictPIDResponsivePPO:

    def __init__(
        self,
        obs_dim=16,
        state_dim=64,
        action_dim=3,
        cost_limit=13.25,
        gamma=0.99,
        lr=0.0002,
        clip_eps=0.2,
        seed=42,
        device="cpu",
    ):
        self.device = device
        self.seed = seed
        self.encoder = BeliefEncoder(obs_dim, state_dim).to(device)
        self.actor = HybridActor(state_dim, action_dim).to(device)
        self.critic_r = ScalarCritic(state_dim).to(device)
        self.critic_c = ScalarCritic(state_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic_r.parameters())
            + list(self.critic_c.parameters()),
            lr=lr,
        )
        self.cost_limit = cost_limit
        self.pid = StandardPIDController(kp=1.0, ki=0.05, kd=0.01, lam_max=30.0)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lam = 0.0
        self.best_state = None
        self.best_score = -1000000000.0

    def init_belief(self):
        return self.encoder.init_belief(1, self.device)

    def set_eval_mode(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic_r.eval()
        self.critic_c.eval()

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

    def save_best(self):
        self.best_state = {
            "enc": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
            "act": {k: v.detach().clone() for k, v in self.actor.state_dict().items()},
            "cr": {k: v.detach().clone() for k, v in self.critic_r.state_dict().items()},
            "cc": {k: v.detach().clone() for k, v in self.critic_c.state_dict().items()},
        }

    def load_best(self):
        if self.best_state is not None:
            self.encoder.load_state_dict(self.best_state["enc"])
            self.actor.load_state_dict(self.best_state["act"])
            self.critic_r.load_state_dict(self.best_state["cr"])
            self.critic_c.load_state_dict(self.best_state["cc"])

    def update(self, trajectories, epochs=10):
        avg_cost = float(np.mean([tr["v_drop_ep"] for tr in trajectories]))
        self.lam = self.pid.update(avg_cost, self.cost_limit)
        prepared = []
        with torch.no_grad():
            for traj in trajectories:
                obs = torch.as_tensor(
                    np.asarray(traj["obs"]), dtype=torch.float32, device=self.device
                )
                states = self.encoder.encode_sequence(obs)
                vr = self.critic_r(states).cpu().numpy()
                vc = self.critic_c(states).cpu().numpy()
                adv_r = self.gae(
                    vr, np.asarray(traj["rewards"], dtype=np.float32), np.asarray(traj["dones"])
                )
                ret_c = undiscounted_cost_returns(traj["c_shaped"])
                prepared.append(
                    (
                        obs,
                        torch.as_tensor(traj["a_types"], dtype=torch.long, device=self.device),
                        torch.as_tensor(traj["a_mus"], dtype=torch.float32, device=self.device),
                        torch.as_tensor(traj["log_probs"], dtype=torch.float32, device=self.device),
                        torch.as_tensor(adv_r, dtype=torch.float32, device=self.device),
                        torch.as_tensor(ret_c - vc, dtype=torch.float32, device=self.device),
                        torch.as_tensor(adv_r + vr, dtype=torch.float32, device=self.device),
                        torch.as_tensor(ret_c, dtype=torch.float32, device=self.device),
                    )
                )
        for _ in range(epochs):
            policy_losses, vr_losses, vc_losses = ([], [], [])
            for obs, at, am, old_lp, ar, ac, rr, rc in prepared:
                states = self.encoder.encode_sequence(obs)
                probs, means = self.actor(states)
                cat = torch.distributions.Categorical(probs)
                normal = torch.distributions.Normal(
                    means.gather(-1, at[:, None]).squeeze(-1), self.actor.get_mu_std()[at]
                )
                lp = hybrid_log_prob(cat, normal, at, am)
                ratio = torch.exp((lp - old_lp).clamp(-20, 20))
                clipped = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
                reward_surr = torch.minimum(ratio * ar, clipped * ar).sum()
                cost_surr = torch.maximum(ratio * ac, clipped * ac).sum()
                policy_losses.append(-(reward_surr - self.lam * cost_surr) / (1.0 + self.lam))
                vr_losses.append(F.mse_loss(self.critic_r(states), rr))
                vc_losses.append(F.mse_loss(self.critic_c(states), rc))
            mean_length = float(np.mean([len(tr["obs"]) for tr in trajectories]))
            loss = torch.stack(policy_losses).mean() / mean_length
            loss = loss + 0.5 * (torch.stack(vr_losses).mean() + torch.stack(vc_losses).mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CPPO-PID loss")
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 0.5)
            self.optimizer.step()
        return (avg_cost, self.lam)
