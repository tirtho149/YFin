#!/bin/bash
# One-time environment setup — run on a LOGIN NODE.
#
# Requires Python 3.10 or 3.11 (PyTorch wheels; avoid 3.13+).
# Usage:  bash slurm/setup_env.sh  [env_dir]
set -euo pipefail
cd /work/mech-ai-scratch/tirtho/YFin

ENV_DIR="${1:-$HOME/envs/finbench}"
echo "creating virtualenv at: $ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install -e .                       # installs finbench + dependencies

python -c "import torch, xgboost, sklearn, pandas; \
print('env OK — torch', torch.__version__, '| cuda?', torch.cuda.is_available())"

echo
echo "done. activate it before prestaging / submitting:"
echo "    source $ENV_DIR/bin/activate"
