import numpy as np
import torch
import scipy.signal

class SurvivabilityMetric:
    def __init__(self, base_weights, beta=0.587, rebound=0.1):
        self.w = np.array(base_weights, dtype=np.float32)
        self.beta = beta
        self.rebound = rebound
        self.tau = np.zeros_like(self.w)

    def update_tau(self, migration_signal):
        for i, s in enumerate(migration_signal):
            if s > 0:
                self.tau[i] = min(1.0, self.tau[i] + s)
            else:
                self.tau[i] = max(0.0, self.tau[i] - self.rebound)

    def survivability(self, s_phy):
        return float(np.dot(self.w * np.exp(-self.beta * self.tau), s_phy))

