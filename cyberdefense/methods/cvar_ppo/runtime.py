"""Upper-tail episodic-cost CVaR-PPO, adapted from Ying et al. (2022). Static CVaR: Rockafellar and Uryasev (2000).
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


logger = logging.getLogger("cyberdefense.cvar_ppo")


def setup_logger(mode_tag="run"):
    return logger
