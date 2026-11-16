"""CPPO-PID. Reference: Stooke et al. (2020), Responsive Safety in Reinforcement Learning by PID Lagrangian Methods, ICML.
Recurrent belief encoder, hybrid action policy and value networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hybrid_log_prob(type_dist, mu_dist, a_type, a_mu):
    lower, upper = (0.0001, 1.0 - 0.0001)
    z_low = (lower - mu_dist.loc) / mu_dist.scale
    z_high = (mu_dist.loc - upper) / mu_dist.scale
    log_mu = torch.where(
        a_mu <= lower,
        torch.special.log_ndtr(z_low),
        torch.where(a_mu >= upper, torch.special.log_ndtr(z_high), mu_dist.log_prob(a_mu)),
    )
    return type_dist.log_prob(a_type) + log_mu


class BeliefEncoder(nn.Module):

    def __init__(self, obs_dim=16, belief_dim=64):
        super().__init__()
        self.gru = nn.GRUCell(obs_dim, belief_dim)
        self.belief_dim = belief_dim

    def forward(self, o_t, b_prev):
        return self.gru(o_t, b_prev)

    def encode_sequence(self, obs_tensor):
        T = obs_tensor.shape[0]
        b = torch.zeros(1, self.belief_dim, device=obs_tensor.device)
        beliefs = []
        for t in range(T):
            b = self.gru(obs_tensor[t].unsqueeze(0), b)
            beliefs.append(b.squeeze(0))
        return torch.stack(beliefs, dim=0)

    def init_belief(self, batch, device):
        return torch.zeros(batch, self.belief_dim, device=device)


class HybridActor(nn.Module):

    def __init__(self, state_dim=64, n_types=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
        )
        self.type_head = nn.Linear(128, n_types)
        self.mu_head = nn.Sequential(nn.Linear(128, n_types), nn.Sigmoid())
        self.mu_logstd = nn.Parameter(torch.zeros(n_types) - 0.5)

    def get_mu_std(self):
        return torch.clamp(self.mu_logstd, min=-5.0, max=1.0).exp()

    def forward(self, x):
        h = self.net(x)
        logits = self.type_head(h).clamp(-15.0, 15.0)
        type_probs = F.softmax(logits, dim=-1)
        type_probs = torch.clamp(type_probs, 1e-06, 1.0)
        type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)
        return (type_probs, self.mu_head(h))


class ScalarCritic(nn.Module):

    def __init__(self, state_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
