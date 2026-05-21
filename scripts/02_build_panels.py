#!/usr/bin/env python
"""Step 2 — build and cache the engineered feature panels.

Reads the per-ticker OHLCV CSVs, builds both feature sets (ohlc / finance),
z-scores them cross-sectionally and caches each as parquet under data/cache/.
CPU-only — no GPU needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finbench.config import load_config
from finbench.data import load_returns_and_meta
from finbench.logging_utils import setup_logging
from finbench.panel import prepare_panel


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = setup_logging(cfg.log_dir, "02_panels")
    _ret, _regime, _sector, tickers, dates = load_returns_and_meta(cfg)
    log.info("metadata: %d tickers, %d dates", len(tickers), len(dates))
    for feature_set in cfg.feature_sets:
        panel = prepare_panel(cfg, feature_set, tickers, dates)
        log.info("panel '%s': %d rows cached", feature_set, len(panel))
    log.info("STEP 2 COMPLETE — panels cached in %s", cfg.cache_dir)


if __name__ == "__main__":
    main()
