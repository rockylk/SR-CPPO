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
        self.n_phases = n_phases

    def forward(self, o_t, b_prev):
        return self.gru(o_t, b_prev)

    def phase_logits(self, b):
        return self.phase_head(b)

    def attack_logits(self, b):
        return self.attack_head(b)

    def attack_prob(self, b):
        return F.softmax(self.attack_logits(b), dim=-1)[..., 1]

    def init_belief(self, batch, device):
        return torch.zeros(batch, self.belief_dim, device=device)


class HybridActor(nn.Module):
    """
    V7.2.1 keeps V7.2 hard action gate but adds A1-only fallback.

    Normal gate:
      a0: always allowed
      a1: allowed iff attack_prob >= 0.30
      a2: allowed iff attack_prob >= 0.70

    E2 A1 fallback:
      When recent attack risk is high:
        a1 threshold is temporarily lowered to gate_low_fallback=0.12.
      Important:
        a2 is NOT relaxed. a2 still requires attack_prob >= 0.70.
    """

    def __init__(
        self,
        belief_dim,
        n_types=3,
        hidden=128,
        n_phases=3,
        logstd_init=-0.5,
        gate_low=0.3,
        gate_high=0.7,
        gate_low_fallback=0.12,
        type_ent_weight=1.5,
        mu_ent_weight=0.5,
    ):
        super().__init__()

        input_dim = belief_dim + n_phases

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.type_head = nn.Linear(hidden, n_types)
        self.mu_mean = nn.Linear(hidden, n_types)
        self.mu_logstd = nn.Parameter(torch.zeros(n_types) + logstd_init)

        self.n_phases = n_phases
        self.gate_low = gate_low
        self.gate_high = gate_high
        self.gate_low_fallback = gate_low_fallback

        self.type_ent_weight = type_ent_weight
        self.mu_ent_weight = mu_ent_weight

    def _input(self, b, phase_logits):
        phase_probs = F.softmax(phase_logits.detach(), dim=-1)
        return torch.cat([b, phase_probs], dim=-1)

    def forward(self, b, phase_logits):
        x = self._input(b, phase_logits)
        h = self.trunk(x)
        type_logits = self.type_head(h).clamp(-15.0, 15.0)
        mu_mean = torch.sigmoid(self.mu_mean(h))
        return type_logits, mu_mean

    def _safe_logstd(self, a_type):
        return self.mu_logstd[a_type].clamp(-1.0, 1.0)

    def _apply_gate(
        self,
        type_logits,
        attack_prob,
        gate_active=True,
        a1_fallback_active=False,
    ):
        if not gate_active:
            return type_logits

        ap = attack_prob.detach()
        out = type_logits.clone()

        NEG = torch.full_like(out[..., 0], -1e9)

        # E2: only relax a1 under risk fallback. a2 stays strict.
        effective_gate_low = self.gate_low_fallback if a1_fallback_active else self.gate_low

        mask1 = ap < effective_gate_low
        mask2 = ap < self.gate_high

        out[..., 1] = torch.where(mask1, NEG, out[..., 1])
        out[..., 2] = torch.where(mask2, NEG, out[..., 2])

        return out

    def _entropy(self, type_dist, mu_dist):
        return self.type_ent_weight * type_dist.entropy() + self.mu_ent_weight * mu_dist.entropy()

    def act(
        self,
        b,
        phase_logits,
        attack_prob,
        gate_active=True,
        a1_fallback_active=False,
    ):
        type_logits, mu_mean = self.forward(b, phase_logits)
        type_logits = self._apply_gate(
            type_logits,
            attack_prob,
            gate_active=gate_active,
            a1_fallback_active=a1_fallback_active,
        )

        type_dist = Categorical(logits=type_logits)
        a_type = type_dist.sample()

        idx = a_type.unsqueeze(-1)
        mu_m = mu_mean.gather(-1, idx).squeeze(-1)
        mu_std = self._safe_logstd(a_type).exp()

        mu_dist = Normal(mu_m, mu_std)
        a_mu = mu_dist.sample().clamp(1e-4, 1.0 - 1e-4)

        logp = type_dist.log_prob(a_type) + mu_dist.log_prob(a_mu)
        ent = self._entropy(type_dist, mu_dist)

        return a_type, a_mu, logp, ent

    def evaluate(
        self,
        b,
        phase_logits,
        attack_prob,
        a_type,
        a_mu,
        gate_active=True,
        gate_attack_prob=None,
        a1_fallback_active=False,
    ):
        """
        gate_attack_prob:
            rollout-time attack probability used to reconstruct the same
            gate support as sampling time.
        a1_fallback_active:
            rollout-time fallback flag; must match act().
        """
        type_logits, mu_mean = self.forward(b, phase_logits)

        if gate_attack_prob is None:
            gate_attack_prob = attack_prob

        type_logits = self._apply_gate(
            type_logits,
            gate_attack_prob,
            gate_active=gate_active,
            a1_fallback_active=a1_fallback_active,
        )

        type_dist = Categorical(logits=type_logits)

        idx = a_type.unsqueeze(-1)
        mu_m = mu_mean.gather(-1, idx).squeeze(-1)
        mu_std = self._safe_logstd(a_type).exp()

        mu_dist = Normal(mu_m, mu_std)

        logp = type_dist.log_prob(a_type) + mu_dist.log_prob(a_mu.clamp(1e-4, 1.0 - 1e-4))
        ent = self._entropy(type_dist, mu_dist)

        return logp, ent


class QuantileCritic(nn.Module):
    def __init__(self, belief_dim, n_quant=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_quant),
        )

        self.n_quant = n_quant
        self.register_buffer("taus", (torch.arange(n_quant).float() + 0.5) / n_quant)

    def forward(self, b):
        return self.net(b)


class ScalarCritic(nn.Module):
    def __init__(self, belief_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, b):
        return self.net(b).squeeze(-1)


# =====================================================
# 2. Survivability + Empirical CVaR
# =====================================================
class DummyActor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.gate_low = 0.3
        self.gate_high = 0.7
        self.gate_low_fallback = 0.12
    
    def act(self, obs, b_prev, ap_old, gate_active, gate_ap_old, a1_fallback_active):
        import torch
        with torch.no_grad():
            b_new = self.encoder(obs, b_prev)
            atk_logits = self.encoder.attack_logits(b_new)
            atk_prob = torch.sigmoid(atk_logits)[0, 1].item()
            
        # SOC Rules (Hardcoded Heuristics)
        if atk_prob < 0.4:
            a_type = 0
            a_mu = 0.0
        elif atk_prob < 0.7:
            a_type = 1
            a_mu = 0.0
        else:
            a_type = 2
            a_mu = 0.8
            
        return a_type, a_mu, 0.0, b_new

