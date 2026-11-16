"""CPO. Reference: Achiam et al. (2017), Constrained Policy Optimization, ICML.
Policy optimization and checkpoint selection."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .models import BeliefEncoder, HybridActor, ScalarCritic, hybrid_log_prob
from .risk import undiscounted_cost_returns


def conjugate_gradients(Avp_fn, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp_p = Avp_fn(p)
        alpha = rdotr / (torch.dot(p, Avp_p) + 1e-08)
        x += alpha * p
        r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-08)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def solve_cpo_step(x, y, g, b, violation, delta):
    q = max(float(torch.dot(g, x)), 0.0)
    s = max(float(torch.dot(b, y)), 0.0)
    r = float(torch.dot(g, y))
    zero = torch.zeros_like(x)
    if s <= 1e-12:
        if violation > 0.0 or q <= 1e-12:
            return (zero, violation > 0.0)
        return (np.sqrt(2.0 * delta / q) * x, False)
    if violation > np.sqrt(2.0 * delta * s):
        return (-np.sqrt(2.0 * delta / s) * y, True)
    if q <= 1e-12 and violation <= 0.0:
        return (zero, False)
    if q > 1e-12:
        unconstrained = np.sqrt(2.0 * delta / q) * x
        if float(torch.dot(b, unconstrained)) + violation <= 0.0:
            return (unconstrained, False)
    center = -(violation / s) * y
    radius_sq = max(0.0, 2.0 * delta - violation**2 / s)
    tangent_sq = max(0.0, q - r**2 / s)
    if tangent_sq <= 1e-12:
        return (center, False)
    return (center + np.sqrt(radius_sq / tangent_sq) * (x - r / s * y), False)


class StrictCPOAgent:

    def __init__(
        self,
        obs_dim=16,
        state_dim=64,
        action_dim=3,
        cost_limit=13.25,
        kl_bound=0.01,
        gamma=0.99,
        lr=0.0002,
        seed=42,
        device="cpu",
    ):
        self.device = device
        self.seed = seed
        self.encoder = BeliefEncoder(obs_dim, state_dim).to(device)
        self.actor = HybridActor(state_dim, action_dim).to(device)
        self.critic_r = ScalarCritic(state_dim).to(device)
        self.critic_c = ScalarCritic(state_dim).to(device)
        self.opt_enc_critics = optim.Adam(
            list(self.critic_r.parameters()) + list(self.critic_c.parameters()), lr=lr
        )
        self.cost_limit = cost_limit
        self.kl_bound = kl_bound
        self.gamma = gamma
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

    def update(self, trajectories):
        policy_params = list(self.encoder.parameters()) + list(self.actor.parameters())
        obs_batches, type_batches, mu_batches, adv_rs, adv_cs, ret_rs, ret_cs = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        with torch.no_grad():
            for traj in trajectories:
                obs = torch.as_tensor(
                    np.asarray(traj["obs"]), dtype=torch.float32, device=self.device
                )
                states = self.encoder.encode_sequence(obs)
                vr = self.critic_r(states).cpu().numpy()
                vc = self.critic_c(states).cpu().numpy()
                ar = self.gae(
                    vr, np.asarray(traj["rewards"], dtype=np.float32), np.asarray(traj["dones"])
                )
                rc = undiscounted_cost_returns(traj["c_shaped"])
                obs_batches.append(obs)
                type_batches.append(
                    torch.as_tensor(traj["a_types"], dtype=torch.long, device=self.device)
                )
                mu_batches.append(
                    torch.as_tensor(traj["a_mus"], dtype=torch.float32, device=self.device)
                )
                adv_rs.append(torch.as_tensor(ar, dtype=torch.float32, device=self.device))
                adv_cs.append(torch.as_tensor(rc - vc, dtype=torch.float32, device=self.device))
                ret_rs.append(torch.as_tensor(ar + vr, dtype=torch.float32, device=self.device))
                ret_cs.append(torch.as_tensor(rc, dtype=torch.float32, device=self.device))
        at, am = (torch.cat(type_batches), torch.cat(mu_batches))
        ar, ac = (torch.cat(adv_rs), torch.cat(adv_cs))
        if ar.numel() > 1 and ar.std() > 1e-06:
            ar = (ar - ar.mean()) / (ar.std() + 1e-08)
        ep_cost_mean = float(np.mean([tr["v_drop_ep"] for tr in trajectories]))
        c_val = ep_cost_mean - self.cost_limit
        n_episodes = len(trajectories)

        def encode():
            return torch.cat([self.encoder.encode_sequence(obs) for obs in obs_batches])

        with torch.no_grad():
            states_old = encode()
            old_probs, old_means = self.actor(states_old)
            old_std = self.actor.get_mu_std().unsqueeze(0).expand_as(old_means).clone()
            old_cat = torch.distributions.Categorical(old_probs)
            old_normal = torch.distributions.Normal(
                old_means.gather(-1, at[:, None]).squeeze(-1),
                old_std.gather(-1, at[:, None]).squeeze(-1),
            )
            old_lp = hybrid_log_prob(old_cat, old_normal, at, am)

        def surrogates():
            states = encode()
            probs, means = self.actor(states)
            cat = torch.distributions.Categorical(probs)
            normal = torch.distributions.Normal(
                means.gather(-1, at[:, None]).squeeze(-1), self.actor.get_mu_std()[at]
            )
            lp = hybrid_log_prob(cat, normal, at, am)
            ratio = torch.exp((lp - old_lp).clamp(-20, 20))
            return ((ratio * ar).sum() / n_episodes, ((ratio - 1.0) * ac).sum() / n_episodes)

        def kl_fn():
            probs, means = self.actor(encode())
            std = self.actor.get_mu_std().unsqueeze(0).expand_as(means)
            cat_kl = (old_probs * (old_probs.log() - probs.log())).sum(-1)
            norm_kl = (
                torch.log(std / old_std)
                + (old_std.square() + (old_means - means).square()) / (2.0 * std.square())
                - 0.5
            )
            return (cat_kl + (old_probs * norm_kl).sum(-1)).mean()

        def flat_grad(value, create_graph=False):
            gradients = torch.autograd.grad(value, policy_params, create_graph=create_graph)
            return torch.cat([grad.reshape(-1) for grad in gradients])

        reward_old, _ = surrogates()
        g = flat_grad(reward_old).detach()
        _, cost_old = surrogates()
        b = flat_grad(cost_old).detach()

        def fvp(vector):
            kl_grad = flat_grad(kl_fn(), create_graph=True)
            return flat_grad(torch.dot(kl_grad, vector)).detach() + 0.01 * vector

        x = conjugate_gradients(fvp, g)
        y = conjugate_gradients(fvp, b)
        step, recovery = solve_cpo_step(x, y, g, b, c_val, self.kl_bound)
        old_params = torch.cat([p.detach().reshape(-1) for p in policy_params]).clone()

        def assign(vector):
            offset = 0
            with torch.no_grad():
                for param in policy_params:
                    count = param.numel()
                    param.copy_(vector[offset : offset + count].view_as(param))
                    offset += count

        accepted = False
        if torch.isfinite(step).all() and torch.linalg.vector_norm(step) > 1e-12:
            expected_improvement = float(torch.dot(g, step))
            for backtrack in range(10):
                fraction = 0.5**backtrack
                assign(old_params + fraction * step)
                with torch.no_grad():
                    new_r, delta_c = surrogates()
                    kl = float(kl_fn())
                    allowed_c = max(-c_val, 0.0)
                    cost_ok = float(delta_c) <= allowed_c + 1e-07
                reward_ok = (
                    c_val > 0.0
                    or recovery
                    or float(new_r - reward_old) >= 0.1 * fraction * expected_improvement
                )
                if np.isfinite(kl) and kl <= self.kl_bound and cost_ok and reward_ok:
                    accepted = True
                    break
        if not accepted:
            assign(old_params)
        with torch.no_grad():
            states = encode().detach()
        rr, rc = (torch.cat(ret_rs), torch.cat(ret_cs))
        for _ in range(8):
            loss = F.mse_loss(self.critic_r(states), rr) + F.mse_loss(self.critic_c(states), rc)
            self.opt_enc_critics.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.opt_enc_critics.param_groups[0]["params"], 0.5)
            self.opt_enc_critics.step()
        self.last_update = {
            "accepted": accepted,
            "recovery": recovery,
            "kl": float(kl_fn().detach()),
            "cost_mean": ep_cost_mean,
        }
        return ep_cost_mean
