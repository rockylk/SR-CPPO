import logging
import sys
import os
from datetime import datetime

def setup_logger():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(os.getcwd(), f"cpo_baseline_{ts}.log")
    logger = logging.getLogger("CPO_Baseline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


