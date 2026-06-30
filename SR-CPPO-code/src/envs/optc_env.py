"""
OpTC Data-Driven Environment
----------------------------
A simulated Partially Observable Markov Decision Process (POMDP) built from
real-world OpTC telemetry. Evaluates agent defense actions against Advanced
Persistent Threats (APTs) while tracking business continuity and survivability.
"""
import sys
import numpy as np
from src.utils.logger import logger
from src.utils.cvar_math import SurvivabilityMetric

class OpTC_DataDrivenEnv:
    def __init__(
        self,
        benign_path,
        mal_path,
        horizon=200,
        attack_rate=0.55,
        V_threshold=0.6,
        V_min_target=0.65,
        idle_bonus=0.03,
    ):
        self.horizon = horizon
        self.attack_rate = attack_rate
        self.obs_dim = 16

        try:
            self.benign_data = np.load(benign_path)
            self.mal_data = np.load(mal_path)
            logger.info(
                f"[Env] Data Loaded. Benign: {self.benign_data.shape}, "
                f"Malicious: {self.mal_data.shape}"
            )
        except Exception as e:
            logger.error(f"[Env] Data Load Failed! {e}")
            sys.exit(1)

        self.base_w = [0.2, 0.9]
        self.metric = SurvivabilityMetric(self.base_w, beta=0.587, rebound=0.05)

        self.V_threshold = V_threshold
        self.V_min_target = V_min_target
        self.idle_bonus = idle_bonus

        self.c_limit_benign = 5.0
        self.c_limit_attack = 20.0
        self.c_limit_eff = attack_rate * self.c_limit_attack + (1.0 - attack_rate) * self.c_limit_benign

        self.reset()

    def reset(self):
        self.t = 0
        self.is_malicious = np.random.rand() < self.attack_rate

        if self.is_malicious:
            self.dataset = self.mal_data
            self.attack_phase = int(np.random.choice([1, 1, 2]))
        else:
            self.dataset = self.benign_data
            self.attack_phase = 0

        max_start = max(0, len(self.dataset) - self.horizon - 1)
        self.start_idx = np.random.randint(0, max_start + 1)

        self.s_phy = np.array([1.0, 1.0])
        self.metric.tau = np.zeros(2)

        self.V_prev = self.metric.survivability(self.s_phy)
        self.V_min_ep = self.V_prev
        self.episode_cost_sum = 0.0

        return self._get_obs(), self.attack_phase

    def _get_obs(self):
        idx = min(self.start_idx + self.t, len(self.dataset) - 1)
        return self.dataset[idx].astype(np.float32)

    def step(self, a_type, a_mu):
        if a_type == 0:
            s_c, s_b, mig = 1.0, 1.0, [0, 0]
        elif a_type == 1:
            s_c, s_b, mig = 1.0, 1.0, [0, a_mu]
        else:
            s_c, s_b, mig = 1.0 - 0.5 * a_mu, 1.0 - a_mu, [0, 0]

        self.metric.update_tau(mig)
        self.s_phy = np.array([s_c, s_b])

        V_now = self.metric.survivability(self.s_phy)
        self.V_min_ep = min(self.V_min_ep, V_now)
        self.V_prev = V_now

        cost = 10.0 * max(0.0, self.V_min_target - V_now)

        reward = 0.02
        reward += 0.50 * V_now
        reward -= 0.15 * max(0.0, self.V_min_target - V_now)

        intercept_prob = {
            0: 0.05,
            1: 0.4 + 0.3 * a_mu,
            2: 0.8 + 0.2 * a_mu,
        }[a_type]

        if self.is_malicious:
            intercepted = (np.random.rand() < intercept_prob) and (self.attack_phase >= 1)

            if intercepted:
                reward += 1.5
                self.attack_phase = max(0, self.attack_phase - 1)
            else:
                reward -= 0.18

                if a_type == 0 and self.attack_phase >= 1:
                    reward -= 0.05

                if np.random.rand() < 0.2:
                    self.attack_phase = min(2, self.attack_phase + 1)

                if self.attack_phase == 2:
                    cost += 2.0
        else:
            self.attack_phase = 0

            if a_type == 0:
                reward += self.idle_bonus
            else:
                reward -= 0.12 * a_mu
                if a_type == 2:
                    reward -= 0.20 * a_mu

        self.episode_cost_sum += cost

        self.t += 1
        done = (self.t >= self.horizon) or (s_b <= 0.01)

        info = {}
        if done:
            v_short_ep = 100.0 * max(0.0, self.V_min_target - self.V_min_ep)
            risk_ep = v_short_ep + 0.05 * self.episode_cost_sum

            info["V_drop_ep"] = float(risk_ep)
            info["V_short_ep"] = float(v_short_ep)
            info["V_min_ep"] = float(self.V_min_ep)

        return self._get_obs(), reward, cost, done, self.attack_phase, info