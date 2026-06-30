"""
Logger Module
-------------
Handles the configuration and initialization of the training logger.
Outputs logs to both the console and a time-stamped log file.
"""
import os
import sys
import logging
from datetime import datetime

def setup_logger():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(os.getcwd(), f"optc_training_v721_{ts}.log")
    logger = logging.getLogger("SR_CPPO_V721")
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


logger = setup_logger()