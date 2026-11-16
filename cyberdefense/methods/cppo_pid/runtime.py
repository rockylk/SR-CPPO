"""CPPO-PID. Reference: Stooke et al. (2020), Responsive Safety in Reinforcement Learning by PID Lagrangian Methods, ICML.
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


logger = logging.getLogger("cyberdefense.cppo_pid")


def setup_logger(mode_tag="run"):
    return logger
