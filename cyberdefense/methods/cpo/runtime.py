"""CPO. Reference: Achiam et al. (2017), Constrained Policy Optimization, ICML.
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


logger = logging.getLogger("cyberdefense.cpo")


def setup_logger(mode_tag="run"):
    return logger
