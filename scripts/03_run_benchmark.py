#!/usr/bin/env python
"""Step 3 — run the full model benchmark.

Trains every model family (classical / MLP / sequence / world model) on every
feature set and target. Results, predictions, checkpoints and plots are saved
incrementally; re-running resumes (already-completed models are skipped).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finbench.benchmark import run_benchmark
from finbench.config import load_config
from finbench.logging_utils import setup_logging


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = setup_logging(cfg.log_dir, "03_benchmark")
    run_benchmark(cfg)
    log.info("STEP 3 COMPLETE — results in %s", cfg.results_dir)


if __name__ == "__main__":
    main()
