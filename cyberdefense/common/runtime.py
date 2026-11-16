"""Run-scoped logging, paths and implementation fingerprints."""

import hashlib
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


@contextmanager
def output_directory(path):
    """Keep native relative checkpoint and CSV names inside the selected run folder."""
    previous = Path.cwd()
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(previous)


def configure_logging(path):
    logger = logging.getLogger("cyberdefense")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(message)s")
    filename = Path(path) / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    for handler in (
        logging.FileHandler(filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return filename


def package_digest():
    """Fingerprint all Python modules used by a packaged ablation run."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
