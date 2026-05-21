#!/usr/bin/env python
"""Step 1 — download OHLCV data.

Needs internet. On an HPC cluster whose compute nodes are firewalled, run this
on a LOGIN NODE before submitting the GPU job (see slurm/prestage_data.sh).
Cached CSVs are reused, so re-running only fetches what is missing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finbench.config import load_config
from finbench.data import download_data
from finbench.logging_utils import setup_logging


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = setup_logging(cfg.log_dir, "01_download")
    log.info("project root: %s", cfg.root)
    download_data(cfg)
    log.info("STEP 1 COMPLETE — data in %s", cfg.data_dir)


if __name__ == "__main__":
    main()
