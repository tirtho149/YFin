#!/bin/bash
# Step 1 — download OHLCV data. Run on a LOGIN NODE (needs internet) BEFORE
# submitting the GPU job, because cluster compute nodes are usually firewalled.
#
# Downloading ~325 tickers takes roughly 5-10 minutes; cached CSVs are reused.
set -euo pipefail
cd /work/mech-ai-scratch/tirtho/YFin

# EDIT: activate the environment created by slurm/setup_env.sh
# source "$HOME/envs/finbench/bin/activate"

export FINBENCH_ROOT=/work/mech-ai-scratch/tirtho/YFin
python scripts/01_download_data.py

echo
echo "data is staged under ./data — now submit the GPU job:"
echo "    sbatch slurm/run_pipeline.sbatch"
