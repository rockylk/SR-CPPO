import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

class BeliefEncoder(nn.Module):
    def __init__(self, obs_dim, belief_dim=64):
        super().__init__()
        self.gru = nn.GRUCell(obs_dim, belief_dim)
        self.belief_dim = belief_dim

    def forward(self, o_t, b_prev):
        return self.gru(o_t, b_prev)

    def init_belief(self, batch, device):
        return torch.zeros(batch, self.belief_dim, device=device)

class Actor(nn.Module):
    def __init__(self, state_dim, n_types=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.type_head = nn.Sequential(nn.Linear(128, n_types), nn.Softmax(dim=-1))
        self.mu_head = nn.Sequential(nn.Linear(128, n_types), nn.Sigmoid())
        self.mu_logstd = nn.Parameter(torch.zeros(n_types) - 0.5)

    def get_mu_std(self):
        """ Safe std computation to prevent numerical overflow """
        safe_logstd = torch.clamp(self.mu_logstd, min=-20.0, max=2.0)
        return safe_logstd.exp()

    def forward(self, x):
        h = self.net(x)
        type_probs = self.type_head(h)
        # FIX: Ensure probs are strictly positive to avoid NaN in Categorical
        type_probs = torch.clamp(type_probs, 1e-6, 1.0) 
        type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)
        return type_probs, self.mu_head(h)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# =========================================================
# 3. PID Controller (STRICT paper version)
# =========================================================
