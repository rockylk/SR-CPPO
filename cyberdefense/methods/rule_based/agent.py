"""Static threshold response heuristic. Background: Stakhanova et al. (2007), A taxonomy of intrusion response systems.
Policy optimization and checkpoint selection."""

import numpy as np


class StaticRuleBasedAgent:

    def __init__(self, benign_data, theta_low=0.4, theta_high=0.7):
        self.mu_b = np.mean(benign_data, axis=0)
        self.std_b = np.std(benign_data, axis=0) + 1e-06
        self.theta_low = theta_low
        self.theta_high = theta_high

    def init_belief(self):
        return 0.0

    def set_eval_mode(self):
        pass

    def step_action(self, obs_np, prev_risk_score):
        z = np.abs((obs_np - self.mu_b) / self.std_b)
        step_score = float(1.0 / (1.0 + np.exp(-(np.mean(z) - 2.2))))
        risk_score = 0.7 * float(prev_risk_score) + 0.3 * step_score
        if risk_score < self.theta_low:
            a_type, a_mu = (0, 0.0)
        elif risk_score < self.theta_high:
            a_type, a_mu = (1, 0.3)
        else:
            a_type, a_mu = (2, 0.8)
        return (a_type, a_mu, 0.0, risk_score)
