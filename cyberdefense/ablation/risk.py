"""OpTC component ablations of SR-CPPO.
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


def _source_weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.atleast_1d(quantiles).astype(np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    weighted_cdf = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_cdf /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_cdf, values)


def empirical_cvar_weighted(costs, weights=None, alpha=0.85, safety_cap_mult=1.5):
    costs = np.asarray(costs, dtype=np.float64)
    n = len(costs)
    if n == 0:
        return 0.0
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if np.sum(weights) <= 1e-12:
        return float(np.mean(costs))
    if n < 8:
        return float(np.average(costs, weights=weights))
    var = float(weighted_quantile(costs, alpha, weights))
    mask = costs > var
    if not mask.any():
        sorter = np.argsort(costs)[::-1]
        k = max(1, int((1.0 - alpha) * n))
        top_idx = sorter[:k]
        cvar = float(np.average(costs[top_idx], weights=weights[top_idx]))
    else:
        tail_vals = costs[mask]
        tail_w = weights[mask]
        cvar = float(np.average(tail_vals, weights=tail_w))
    cap = safety_cap_mult * float(np.max(costs))
    return min(cvar, cap)


def compute_unweighted_cvar_exact(costs, alpha=0.85):
    arr = np.asarray(list(costs), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    candidates = np.unique(arr)
    f_vals = [nu + np.mean(np.maximum(arr - nu, 0.0)) / (1.0 - alpha) for nu in candidates]
    return float(np.min(f_vals))


def compute_pooled_cvar_exact(d_benign, d_mal, p_m=0.55, alpha=0.85):
    arr_b = np.asarray(list(d_benign), dtype=np.float64)
    arr_m = np.asarray(list(d_mal), dtype=np.float64)
    if arr_b.size == 0 and arr_m.size == 0:
        return 0.0
    if arr_b.size == 0:
        return compute_unweighted_cvar_exact(arr_m, alpha=alpha)
    if arr_m.size == 0:
        return compute_unweighted_cvar_exact(arr_b, alpha=alpha)
    candidates = np.unique(np.concatenate([arr_b, arr_m]))
    best_val = float("inf")
    for nu in candidates:
        f_b = nu + np.mean(np.maximum(arr_b - nu, 0.0)) / (1.0 - alpha)
        f_m = nu + np.mean(np.maximum(arr_m - nu, 0.0)) / (1.0 - alpha)
        val = (1.0 - p_m) * f_b + p_m * f_m
        if val < best_val:
            best_val = float(val)
    return best_val


class PIDLagrangian:

    def __init__(
        self,
        kp=0.5,
        ki=0.02,
        kd=0.3,
        lam_init=0.0,
        lam_max=30.0,
        leak=0.985,
        rate_limit=2.0,
        dead_band_frac=0.1,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lam = lam_init
        self.lam_max = lam_max
        self.leak = leak
        self.integral = 0.0
        self.prev_err = 0.0
        self.rate_limit = rate_limit
        self.dead_band_frac = dead_band_frac

    def update(self, cvar, limit):
        if not np.isfinite(cvar):
            return self.lam
        err = cvar - limit
        scale = 1.0 / (1.0 + abs(err) / max(1.0, abs(limit)))
        if abs(err) < self.dead_band_frac * abs(limit):
            self.integral *= 0.95
        not_sat_up = self.lam < self.lam_max - 0.001 or err < 0
        not_sat_down = self.lam > 0.001 or err > 0
        if err > 0 and not_sat_up or (err < 0 and not_sat_down):
            self.integral = max(-1000.0, min(self.integral + err, 10000.0))
        if self.lam >= 0.95 * self.lam_max:
            self.integral *= 0.9
        delta = (self.kp * err + self.ki * self.integral + self.kd * (err - self.prev_err)) * scale
        delta = float(np.clip(delta, -self.rate_limit, self.rate_limit))
        if err < 0:
            self.lam *= self.leak
        self.lam = float(np.clip(self.lam + delta, 0.0, self.lam_max))
        self.prev_err = err
        return self.lam

    def clear(self, hard=False):
        if hard:
            self.lam = 0.0
            self.integral = 0.0
            self.prev_err = 0.0
        else:
            self.lam *= 0.3
            self.integral *= 0.2


def weighted_quantile(values, quantiles, sample_weight):
    result = _source_weighted_quantile(values, quantiles, sample_weight)
    return np.asarray(result).reshape(-1)[0].item() if np.ndim(quantiles) == 0 else result


class AblationPID(PIDLagrangian):

    def __init__(self, enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.enabled = enabled
        self.diagnostics = {}

    def update(self, cvar, limit):
        err = cvar - limit
        not_up = self.lam < self.lam_max - 0.001 or err < 0
        not_down = self.lam > 0.001 or err > 0
        permitted = err > 0 and not_up or (err < 0 and not_down)
        damped = self.integral * (0.95 if abs(err) < self.dead_band_frac * abs(limit) else 1.0)
        self.diagnostics = {
            "integral_before": float(self.integral),
            "lambda_before": float(self.lam),
            "conditional_block_opportunity": int(err != 0 and (not permitted)),
            "integral_clip_opportunity": int(damped + err < -1000.0 or damped + err > 10000.0),
            "near_saturation_before": int(self.lam >= 0.95 * self.lam_max),
        }
        if self.enabled:
            result = super().update(cvar, limit)
        elif not np.isfinite(cvar):
            result = self.lam
        else:
            scale = 1.0 / (1.0 + abs(err) / max(1.0, abs(limit)))
            self.integral = damped + err
            delta = (
                self.kp * err + self.ki * self.integral + self.kd * (err - self.prev_err)
            ) * scale
            delta = float(np.clip(delta, -self.rate_limit, self.rate_limit))
            if err < 0:
                self.lam *= self.leak
            self.lam = float(np.clip(self.lam + delta, 0.0, self.lam_max))
            self.prev_err = err
            result = self.lam
        self.diagnostics["integral_after"] = float(self.integral)
        self.diagnostics["lambda_after_pid"] = float(self.lam)
        return result
