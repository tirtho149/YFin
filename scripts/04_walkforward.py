#!/usr/bin/env python
"""Step 4 — walk-forward validation, regime-split testing and calibration.

Re-trains XGBoost on Direction across expanding time folds and reports per-fold
and per-regime (bull / sideways / volatile) metrics plus a Brier score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finbench.config import load_config
from finbench.logging_utils import setup_logging
from finbench.walkforward import run_walkforward


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = setup_logging(cfg.log_dir, "04_walkforward")
    run_walkforward(cfg)
    log.info("STEP 4 COMPLETE — results in %s", cfg.results_dir)


if __name__ == "__main__":
    main()
