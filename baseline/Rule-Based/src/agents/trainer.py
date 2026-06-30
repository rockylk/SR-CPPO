import torch
import numpy as np
import time
import os
from collections import deque
from src.utils.logger import logger
from src.utils.math_utils import *
from src.models.networks import *

class RuleBasedDefender:
    def __init__(self, obs_dim, belief_dim, device):
        self.device = device
        self.encoder = Encoder(obs_dim, belief_dim).to(device)
        self.actor = DummyActor(self.encoder)
        self.pid = DummyPID()
        
        self.cost_buf_benign = deque(maxlen=80)
        self.cost_buf_mal = deque(maxlen=80)
        self.cost_recent_benign = deque(maxlen=10)
        self.cost_recent_mal = deque(maxlen=10)
        
        self.recent_R = deque(maxlen=30)
        self.recent_C = deque(maxlen=30)
        self.recent_C_b = deque(maxlen=30)
        self.recent_C_a = deque(maxlen=30)
        self.recent_Vmin_b = deque(maxlen=30)
        self.recent_Vmin_a = deque(maxlen=30)
        self.recent_a2_b = deque(maxlen=30)
        self.recent_a2_a = deque(maxlen=30)
        
        self.c_limit_b = 5.0
        self.c_limit_a = 20.0
        self.c_limit_joint = 14.0
        self.kl_target = 0.01
        self.ent_coef = 0.0
        self.episode_count = 0
        
        self.w_b = 0.45
        self.w_a = 0.55
        self.base_lr = 0.0
        self.clip_eps = 0.0
        self.alpha = 0.85
        self.gate_warmup_eps = 0
        self.pid_start_eps = 0
        self.pid_ramp_eps = 0
        self.ent_coef_floor = 0.0
        self.safe_ckpt_path = "rule_based_best_safe.pt"
        
        self.best_R_avg = 0.0
        self.best_safe_score = 0.0
        self.best_safe_ep = -1
        self.last_rollback_ep = 0
        self.underperf_streak = 0
        self.rollback_cooldown = 20
        self.underperf_threshold = 5.0
        self.sat_streak = 0
        
    def risk_fallback_active(self):
        return False
        
    def _save_snapshot(self):
        pass
        
    def _load_snapshot(self):
        pass
        
    def _save_safe_checkpoint(self, path):
        pass
        
    def clear_cost_buffers(self):
        pass
        
    def clear_memory(self, hard=False):
        pass
        
    def store(self, obs, a_type, a_mu, logp, r, c, d, attack_prob):
        pass
        
    def _cvar_with_decay(self, costs, decay=0.95, alpha=0.85):
        if len(costs) == 0: return 0.0
        n = len(costs)
        weights = np.array([decay**(n-1-i) for i in range(n)])
        return empirical_cvar_weighted(costs, weights, alpha)
        
    def _bucket_cvars(self):
        main_b = list(self.cost_buf_benign)
        main_m = list(self.cost_buf_mal)
        recent_b = list(self.cost_recent_benign)
        recent_m = list(self.cost_recent_mal)

        main_cvar_b = self._cvar_with_decay(main_b, decay=0.99)
        main_cvar_m = self._cvar_with_decay(main_m, decay=0.95)

        recent_cvar_b = self._cvar_with_decay(recent_b, decay=0.98)
        recent_cvar_m = self._cvar_with_decay(recent_m, decay=0.98)

        cvar_b = max(main_cvar_b, recent_cvar_b)
        cvar_a = max(main_cvar_m, recent_cvar_m)

        return cvar_b, cvar_a
        
    def joint_cvar(self, cvar_b, cvar_a):
        return 0.45 * cvar_b + 0.55 * cvar_a
        
    def update(self, *args, **kwargs):
        # Rule-based does not learn!
        return 0.0
        
    def log_scalars(self, *args, **kwargs):
        pass
