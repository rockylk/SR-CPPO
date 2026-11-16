"""OpTC component ablations of SR-CPPO.
Telemetry-driven environment and service-state transitions."""

import sys
import numpy as np
from contextlib import contextmanager

from .config import REFERENCE_WEIGHTS
from .risk import SurvivabilityMetric
from .runtime import logger


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
                f"[Env] Data Loaded. Benign: {self.benign_data.shape}, Malicious: {self.mal_data.shape}"
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
        self.c_limit_eff = (
            attack_rate * self.c_limit_attack + (1.0 - attack_rate) * self.c_limit_benign
        )
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
        self.ep_attack_steps = 0
        self.ep_intercepts = 0
        return (self._get_obs(), self.attack_phase)

    def _get_obs(self):
        idx = min(self.start_idx + self.t, len(self.dataset) - 1)
        return self.dataset[idx].astype(np.float32)

    def step(self, a_type, a_mu):
        if a_type == 0:
            s_c, s_b, mig = (1.0, 1.0, [0, 0])
        elif a_type == 1:
            s_c, s_b, mig = (1.0, 1.0, [0, a_mu])
        else:
            s_c, s_b, mig = (1.0 - 0.5 * a_mu, 1.0 - a_mu, [0, 0])
        self.metric.update_tau(mig)
        self.s_phy = np.array([s_c, s_b])
        V_now = self.metric.survivability(self.s_phy)
        self.V_min_ep = min(self.V_min_ep, V_now)
        self.V_prev = V_now
        cost = 10.0 * max(0.0, self.V_min_target - V_now)
        reward = 0.02
        reward += 0.5 * V_now
        reward -= 0.15 * max(0.0, self.V_min_target - V_now)
        intercept_prob = {0: 0.05, 1: 0.4 + 0.3 * a_mu, 2: 0.8 + 0.2 * a_mu}[a_type]
        if self.is_malicious and self.attack_phase >= 1:
            self.ep_attack_steps += 1
        if self.is_malicious:
            intercepted = np.random.rand() < intercept_prob and self.attack_phase >= 1
            if intercepted:
                self.ep_intercepts += 1
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
                    reward -= 0.2 * a_mu
        self.episode_cost_sum += cost
        self.t += 1
        done = self.t >= self.horizon or s_b <= 0.01
        info = {}
        if done:
            v_short_ep = 100.0 * max(0.0, self.V_min_target - self.V_min_ep)
            risk_ep = v_short_ep + 0.05 * self.episode_cost_sum
            info["V_drop_ep"] = float(risk_ep)
            info["V_short_ep"] = float(v_short_ep)
            info["V_min_ep"] = float(self.V_min_ep)
            info["ep_attack_steps"] = self.ep_attack_steps
            info["ep_intercepts"] = self.ep_intercepts
        return (self._get_obs(), reward, cost, done, self.attack_phase, info)


@contextmanager
def numpy_state(state):
    outer = np.random.get_state()
    np.random.set_state(state)
    try:
        yield
    finally:
        np.random.set_state(outer)


def indexed_rng(seed, episode, step, event):
    value = np.random.SeedSequence([int(seed), int(episode), int(step), int(event)]).generate_state(
        1
    )[0]
    return np.random.RandomState(int(value))


class AblationEnv(OpTC_DataDrivenEnv):

    def __init__(self, benign_path, mal_path, arm, horizon=200):
        self.arm = arm
        self.schedule = None
        self.trace = []
        super().__init__(str(benign_path), str(mal_path), horizon=horizon)

    def reset(self):
        if hasattr(self, "metric"):
            self.metric.w = np.asarray(self.arm.weights, dtype=np.float32)
        if self.schedule is None:
            result = super().reset()
        else:
            seed, episode = self.schedule
            with numpy_state(indexed_rng(seed, episode, 0, 0).get_state()):
                result = super().reset()
        self.reference_min = float(
            np.dot(np.asarray(REFERENCE_WEIGHTS, dtype=np.float32), [1.0, 1.0])
        )
        self.reference_cost_sum = 0.0
        self.reference_reward_sum = 0.0
        self.native_reward_sum = 0.0
        self.trace = []
        return result

    def step(self, a_type, a_mu):
        phase_before = self.attack_phase
        count_before = self.ep_intercepts
        t = self.t
        if self.schedule is None:
            result = super().step(a_type, a_mu)
            u_intercept = u_advance = None
        else:
            seed, episode = self.schedule
            event_rng = indexed_rng(seed, episode, t, 1)
            u_intercept, u_advance = map(float, event_rng.rand(2))
            with numpy_state(indexed_rng(seed, episode, t, 1).get_state()):
                result = super().step(a_type, a_mu)
        obs, reward, cost, done, phase, info = result
        intercepted = self.ep_intercepts > count_before
        if a_type == 0:
            s_true = np.array([1.0, 1.0], dtype=np.float32)
        elif a_type == 1:
            s_true = np.array([1.0, 1.0], dtype=np.float32)
        else:
            s_true = np.array([1.0 - 0.5 * a_mu, 1.0 - a_mu], dtype=np.float32)
        if not self.arm.criticality_aware:
            if a_type == 2:
                self.s_phy = np.array([1.0 - 0.35 * a_mu, 1.0 - 0.35 * a_mu], dtype=np.float32)
            V_native = self.metric.survivability(self.s_phy)
            if t == 0:
                self.V_min_ep = V_native
            else:
                self.V_min_ep = min(self.V_min_ep, V_native)
            self.V_prev = V_native
            cost_native = 10.0 * max(0.0, self.V_min_target - V_native)
            reward_native = 0.02 + 0.5 * V_native - 0.15 * max(0.0, self.V_min_target - V_native)
            if self.is_malicious:
                if intercepted:
                    reward_native += 1.5
                else:
                    reward_native -= 0.18
                    if a_type == 0 and phase_before >= 1:
                        reward_native -= 0.05
                    if self.attack_phase == 2:
                        cost_native += 2.0
            elif a_type == 0:
                reward_native += self.idle_bonus
            else:
                reward_native -= 0.12 * a_mu
            self.episode_cost_sum += cost_native - cost
            reward = float(reward_native)
            cost = float(cost_native)
            if done:
                v_short_native = 100.0 * max(0.0, self.V_min_target - self.V_min_ep)
                risk_native = v_short_native + 0.05 * self.episode_cost_sum
                info["V_drop_ep"] = float(risk_native)
                info["V_short_ep"] = float(v_short_native)
                info["V_min_ep"] = float(self.V_min_ep)
        reference_v = float(
            np.dot(
                np.asarray(REFERENCE_WEIGHTS, dtype=np.float32)
                * np.exp(-self.metric.beta * self.metric.tau),
                s_true,
            )
        )
        self.reference_min = min(self.reference_min, reference_v)
        shortfall = max(0.0, self.V_min_target - reference_v)
        reference_cost = 10.0 * shortfall
        reference_reward = 0.02 + 0.5 * reference_v - 0.15 * shortfall
        if self.is_malicious:
            if intercepted:
                reference_reward += 1.5
            else:
                reference_reward -= 0.18
                if a_type == 0 and phase_before >= 1:
                    reference_reward -= 0.05
                if self.attack_phase == 2:
                    reference_cost += 2.0
        elif a_type == 0:
            reference_reward += self.idle_bonus
        else:
            reference_reward -= 0.12 * a_mu
            if a_type == 2:
                reference_reward -= 0.2 * a_mu
        self.reference_cost_sum += reference_cost
        self.reference_reward_sum += reference_reward
        self.native_reward_sum += reward
        info.update(
            reference_reward=reference_reward,
            reference_cost=reference_cost,
            reference_v=reference_v,
        )
        if done:
            info["reference_risk"] = (
                100.0 * max(0.0, self.V_min_target - self.reference_min)
                + 0.05 * self.reference_cost_sum
            )
            info["reference_vmin"] = self.reference_min
        self.trace.append(
            {
                "step": t,
                "phase_before": phase_before,
                "phase_after": phase,
                "action_type": a_type,
                "action_mu": a_mu,
                "intercepted": int(intercepted),
                "u_intercept": u_intercept,
                "u_advance": u_advance,
                "s_edge": float(s_true[0]),
                "s_critical": float(s_true[1]),
                "tau_edge": float(self.metric.tau[0]),
                "tau_critical": float(self.metric.tau[1]),
                "native_reward": reward,
                "reference_reward": reference_reward,
                "native_cost": cost,
                "reference_cost": reference_cost,
                "reference_v": reference_v,
            }
        )
        return (obs, reward, cost, done, phase, info)
