import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

class BeliefEncoder(nn.Module):
    def __init__(self, obs_dim, belief_dim=64, n_phases=3):
        super().__init__()
        self.gru = nn.GRUCell(obs_dim, belief_dim)
        self.phase_head = nn.Linear(belief_dim, n_phases)
        self.attack_head = nn.Linear(belief_dim, 2)
        self.belief_dim = belief_dim

    def forward(self, o_t, b_prev):
        return self.gru(o_t, b_prev)

    def init_belief(self, batch, device):
        return torch.zeros(batch, self.belief_dim, device=device)

class Actor(nn.Module):
    def __init__(self, state_dim, n_types=3):
        super(Actor, self).__init__()
        # Backbone (Faithful to original)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        # Discrete branch
        self.type_head = nn.Sequential(
            nn.Linear(128, n_types),
            nn.Softmax(dim=-1)
        )
        # Continuous branch (To support Env's a_mu)
        self.mu_head = nn.Sequential(
            nn.Linear(128, n_types),
            nn.Sigmoid()
        )
        self.mu_logstd = nn.Parameter(torch.zeros(n_types) - 0.5)

    def forward(self, x):
        h = self.net(x)
        type_probs = self.type_head(h)
        mu_mean = self.mu_head(h)
        return type_probs, mu_mean


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # Faithful to original
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# Original CVaR Utility (Modified sorting to target WORST costs)
