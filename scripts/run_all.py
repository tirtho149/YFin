#!/usr/bin/env python
"""Run the whole pipeline end to end — this is what the SLURM job calls.

Steps: (1) download [skipped if data already present] -> (2) build panels ->
(3) benchmark -> (4) walk-forward. Steps 3 and 4 are independently wrapped so a
failure in one still lets the other run and still writes the run manifest.

Everything is logged to one file under logs/ and every artifact is written
under results/. The benchmark resumes, so re-submitting after a time-out
continues where it stopped.
"""
import json
import platform
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finbench.benchmark import get_device, run_benchmark
from finbench.config import load_config
from finbench.data import download_data, load_returns_and_meta
from finbench.logging_utils import set_seed, setup_logging
from finbench.panel import prepare_panel
from finbench.walkforward import run_walkforward


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    log = setup_logging(cfg.log_dir, "run_all")
    set_seed(cfg.seed)

    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "device": str(get_device()),
        "config": cfg.to_dict(),
        "steps": {},
    }
    t0 = time.time()
    log.info("FINBENCH PIPELINE — host=%s device=%s", manifest["hostname"],
             manifest["device"])

    def run_step(name: str, fn) -> None:
        log.info("#" * 60)
        log.info("STEP: %s", name)
        s = time.time()
        try:
            fn()
            manifest["steps"][name] = {"status": "ok",
                                       "minutes": round((time.time() - s) / 60, 2)}
        except Exception as exc:                     # noqa: BLE001
            log.error("STEP '%s' FAILED: %s", name, exc)
            log.error(traceback.format_exc())
            manifest["steps"][name] = {"status": "failed", "error": str(exc),
                                       "minutes": round((time.time() - s) / 60, 2)}
        _write_manifest(cfg, manifest)

    # ---- step 1: data download (skip if already present) -------------------
    if cfg.returns_path.is_file():
        log.info("STEP: download — SKIPPED (data already present at %s)",
                 cfg.returns_path)
        manifest["steps"]["download"] = {"status": "skipped", "minutes": 0.0}
        _write_manifest(cfg, manifest)
    else:
        run_step("download", lambda: download_data(cfg))

    # ---- step 2: build feature panels --------------------------------------
    def _panels() -> None:
        _r, _g, _s, tickers, dates = load_returns_and_meta(cfg)
        for fs in cfg.feature_sets:
            prepare_panel(cfg, fs, tickers, dates)

    run_step("build_panels", _panels)

    # ---- step 3: benchmark -------------------------------------------------
    run_step("benchmark", lambda: run_benchmark(cfg))

    # ---- step 4: walk-forward ---------------------------------------------
    run_step("walkforward", lambda: run_walkforward(cfg))

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["total_minutes"] = round((time.time() - t0) / 60, 2)
    _write_manifest(cfg, manifest)
    log.info("#" * 60)
    log.info("PIPELINE COMPLETE in %.1f min — artifacts in %s",
             manifest["total_minutes"], cfg.results_dir)
    for name, info in manifest["steps"].items():
        log.info("  %-14s %-8s %6.1f min", name, info["status"],
                 info.get("minutes", 0.0))


def _write_manifest(cfg, manifest: dict) -> None:
    (cfg.results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
