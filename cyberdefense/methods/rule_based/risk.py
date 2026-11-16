"""Static threshold response heuristic. Background: Stakhanova et al. (2007), A taxonomy of intrusion response systems.
Survivability, empirical upper-tail risk and multiplier control."""

import numpy as np


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


def compute_unweighted_cvar_exact(costs, alpha=0.85):
    arr = np.asarray(list(costs), dtype=np.float64)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if arr.size == 0:
        return float("nan")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite episode risk")
    arr = np.sort(arr)[::-1]
    mass = (1.0 - alpha) * arr.size
    weights = np.clip(mass - np.arange(arr.size), 0.0, 1.0)
    return float(np.dot(arr, weights) / mass)


def compute_pooled_cvar_exact(d_benign, d_mal, p_m=0.55, alpha=0.85):
    b = np.asarray(list(d_benign), dtype=np.float64)
    m = np.asarray(list(d_mal), dtype=np.float64)
    if not 0.0 <= p_m <= 1.0 or not 0.0 <= alpha < 1.0:
        raise ValueError("Invalid mixture weight or confidence level")
    if p_m < 1.0 and b.size == 0 or (p_m > 0.0 and m.size == 0):
        return float("nan")
    values, weights = ([], [])
    for arr, probability in ((b, 1.0 - p_m), (m, p_m)):
        if probability > 0.0:
            values.extend(arr)
            weights.extend(np.full(arr.size, probability / arr.size))
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite episode risk")
    order = np.argsort(values)[::-1]
    values, weights = (values[order], weights[order])
    tail = 1.0 - alpha
    taken = np.minimum(weights, np.maximum(0.0, tail - (np.cumsum(weights) - weights)))
    return float(np.dot(values, taken) / tail)


def undiscounted_cost_returns(costs):
    return np.cumsum(np.asarray(costs, dtype=np.float32)[::-1])[::-1].copy()
