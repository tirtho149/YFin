"""Logging + reproducibility helpers.

``setup_logging`` sends every message to both stdout (captured by the SLURM
``--output`` file) and a timestamped file under ``logs/`` — so nothing is lost.
"""
from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

_LOGGER_NAME = "finbench"


def setup_logging(log_dir: Path, step: str) -> logging.Logger:
    """Configure the ``finbench`` logger for one pipeline step.

    Writes to ``logs/<step>_<timestamp>.log`` and to stdout.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{step}_{ts}.log"

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("=" * 70)
    logger.info("finbench step '%s' started", step)
    logger.info("log file: %s", log_file)
    return logger


def get_logger() -> logging.Logger:
    """Return the shared ``finbench`` logger (configure it once via setup_logging)."""
    return logging.getLogger(_LOGGER_NAME)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
