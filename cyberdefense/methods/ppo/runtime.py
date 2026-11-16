"""PPO. Reference: Schulman et al. (2017), Proximal Policy Optimization Algorithms.
Random seeds and method logger."""

import random
import logging
import numpy as np
import torch


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


logger = logging.getLogger("cyberdefense.ppo")


def setup_logger(mode_tag="run"):
    return logger
