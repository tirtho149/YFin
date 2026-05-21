#!/bin/bash
#SBATCH --job-name=finbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/finbench-%j.out
#SBATCH --error=logs/finbench-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ===========================================================================
# finbench — full close-price direction-prediction benchmark.
# Runs: build panels -> benchmark (classical/MLP/sequence/world model) ->
#       walk-forward + regime evaluation.  Data download is prestaged
#       separately (slurm/prestage_data.sh) since compute nodes lack internet.
#
# Expected wall time on 50 CPU + 1 A100: ~1.5-3 h (the 10 h limit is headroom).
# The benchmark saves results incrementally and RESUMES — if the job is killed,
# just resubmit and it continues from the last completed model.
# ===========================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

# --- environment -----------------------------------------------------------
# finbench runs inside the environment created by  slurm/setup_env.sh.
# Default location is $HOME/envs/finbench; override at submit time with:
#   sbatch --export=ALL,FINBENCH_ENV=/path/to/env  slurm/run_pipeline.sbatch
FINBENCH_ENV="${FINBENCH_ENV:-$HOME/envs/finbench}"

# Uncomment ONLY if your cluster requires it — the pip `torch` wheel already
# bundles its own CUDA runtime, so a CUDA module is usually unnecessary:
# module load cuda

if [ -f "$FINBENCH_ENV/bin/activate" ]; then
    echo "activating venv: $FINBENCH_ENV"
    source "$FINBENCH_ENV/bin/activate"
elif command -v conda >/dev/null 2>&1; then
    echo "activating conda env: finbench"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate finbench
else
    echo "ERROR: no Python environment found." >&2
    echo "       expected a venv at: $FINBENCH_ENV" >&2
    echo "       create it on a login node first:  bash slurm/setup_env.sh" >&2
    exit 1
fi

# Preflight — fail fast (before using GPU time) if the env is incomplete.
if ! python -c "import finbench, torch, xgboost" 2>/dev/null; then
    echo "ERROR: the active environment is missing finbench / torch / xgboost." >&2
    echo "       (re)create it on a login node:  bash slurm/setup_env.sh" >&2
    echo "       active python: $(command -v python || echo '<none found>')" >&2
    exit 1
fi

# Let sklearn / XGBoost / NumPy use all allocated CPU cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export FINBENCH_ROOT="$(pwd)"

echo "====================================================================="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  cpus=${SLURM_CPUS_PER_TASK:-NA}"
echo "python=$(command -v python)  ($(python --version 2>&1))"
echo "start=$(date)"
echo "====================================================================="
nvidia-smi || echo "nvidia-smi unavailable"
python -c "import torch; print('torch', torch.__version__, '| cuda?', torch.cuda.is_available())"

# Run the whole pipeline (download step is auto-skipped when data is present).
python scripts/run_all.py

echo "end=$(date)"
