# finbench — close-price direction prediction benchmark

A staged ML benchmark that **predicts the direction of the next close price**
for ~325 US equities, comparing four model families across two feature
representations. Refactored from the original `DS2010_Final.ipynb` notebook into
a proper, cluster-runnable Python package.

## The task

For every (date, ticker) the **`Direction`** target is the next day's move —
`up` / `flat` / `down` (a ±0.5 % neutral band). Direction is the **primary
task**; next-day log return and forward volatility are also benchmarked.

**Model families** (`model_type`):
| family       | models |
|--------------|--------|
| `classical`  | XGBoost, RandomForest, Ridge / LogisticRegression |
| `mlp`        | feed-forward MLP (small, medium) |
| `sequence`   | LSTM, Transformer (over a 30-day look-back window) |
| `worldmodel` | latent world model (VAE encoder → GRU latent transition → head) |

**Feature sets:** `ohlc` (9 pure price/volume features) vs `finance`
(91 technical indicators).

## Repository layout

```
finbench/                  the package
  config.py                all settings (paths, hyperparameters, run options)
  logging_utils.py         logging + seeding
  data.py                  ticker universe, OHLCV download, metadata loading
  features.py              ohlc_features / finance_features
  panel.py                 panels, splits, look-back windows, z-scoring
  metrics.py               regression / classification metrics, rank IC
  models/                  mlp.py, classical.py, sequence.py, world_model.py
  benchmark.py             trains every model; incremental save + resume
  walkforward.py           walk-forward + regime-split + calibration
scripts/
  01_download_data.py      step 1 — download OHLCV  (needs internet)
  02_build_panels.py       step 2 — build & cache feature panels
  03_run_benchmark.py      step 3 — run the model benchmark
  04_walkforward.py        step 4 — walk-forward / regime evaluation
  run_all.py               run steps 1-4 end to end (the SLURM entry point)
slurm/
  setup_env.sh             one-time environment setup
  prestage_data.sh         download data on a login node
  run_pipeline.sbatch      the GPU job (50 CPU, 1 GPU, 10 h)
```

Runtime outputs (created automatically): `data/`, `results/`, `logs/`.

## Requirements

* **Python 3.10 or 3.11** — PyTorch 2.2.x has no wheels for 3.12+/3.13+.
* A **CUDA GPU** for step 3 (the benchmark). CPU works but is much slower.
* ~2 GB free disk for `data/` + `results/`.

## Installation

```bash
cd /path/to/YFIn
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .                 # installs finbench + all dependencies
# (equivalently: pip install -r requirements.txt)
```

## The pipeline

Four steps. Run them with the numbered scripts, or all at once with `run_all.py`
(this is also what the SLURM job calls).

| # | script | needs | notes |
|---|--------|-------|-------|
| 1 | `01_download_data.py` | **internet** | ~5–10 min first time; CSVs are cached |
| 2 | `02_build_panels.py`  | CPU | builds + caches the feature panels |
| 3 | `03_run_benchmark.py` | GPU recommended | the heavy step (all model families) |
| 4 | `04_walkforward.py`   | CPU/GPU | XGBoost walk-forward + regime eval |

`run_all.py` runs 1→4 and **auto-skips step 1** if the data is already present.

## Running it — Option A: local / single machine

```bash
source .venv/bin/activate
python scripts/run_all.py                 # everything, end to end
```

or one step at a time (later steps reuse the cached panels from step 2):

```bash
python scripts/01_download_data.py
python scripts/02_build_panels.py
python scripts/03_run_benchmark.py
python scripts/04_walkforward.py
```

## Running it — Option B: SLURM cluster (recommended)

Cluster compute nodes are usually firewalled, so the data download runs on a
**login node** and the GPU job runs steps 2–4.

**1 — one-time environment setup** (login node):

```bash
cd /path/to/YFIn
bash slurm/setup_env.sh                   # creates ~/envs/finbench
source ~/envs/finbench/bin/activate
```

**2 — prestage the data** (login node, needs internet):

```bash
bash slurm/prestage_data.sh               # downloads OHLCV into ./data
```

**3 — edit `slurm/run_pipeline.sbatch`** for your cluster:

* `#SBATCH --partition=` — your GPU partition (currently `nova`)
* `#SBATCH --gres=gpu:a100:1` — 1 A100; change the type tag if your cluster
  names its A100s differently, or use a plain `gpu:1` for any GPU

The environment is activated **automatically**: the script sources the venv at
`$HOME/envs/finbench` (created by `setup_env.sh`). Only touch the environment
block if your env lives elsewhere — then submit with
`sbatch --export=ALL,FINBENCH_ENV=/path/to/env slurm/run_pipeline.sbatch` — or
if your cluster needs a `module load`. The job emails `tirtho@iastate.edu` on
start / finish / failure.

**4 — submit the job:**

```bash
sbatch slurm/run_pipeline.sbatch
```

**5 — monitor it:**

```bash
squeue -u $USER                           # job state
tail -f logs/finbench-<jobid>.out         # live progress
```

The job requests **50 CPUs, 1 GPU, 10 h** and typically finishes in
**~1.5–3 h** on an A100 — the 10 h limit is headroom. If it is killed
(time-out / pre-emption), **just resubmit the same command** — it resumes from
the last completed model (see *Resume* below).

## Running a subset or a quick test

Edit `finbench/config.py` (`Config` dataclass) before launching:

* one model family — `model_families = ["classical"]`
* one feature set — `feature_sets = ["ohlc"]`
* fast shakeout — also set `n_epochs`, `seq_epochs`, `wm_epochs` to `1`

Any single step can be re-run on its own; steps 3–4 reuse the step-2 panel cache.

## Verifying a run

```bash
cat results/report.txt                    # benchmark summary (leads with Direction)
cat results/walkforward_report.txt        # walk-forward + per-regime summary
cat results/run_manifest.json             # per-step status (ok/skipped/failed) + timings
column -s, -t results/benchmark_results.csv | less   # full metric table
```

## Resume / crash-safety

Every artifact is written **incrementally**:

* `results/benchmark_results.csv` is rewritten after **each** model.
* completed models are skipped on re-run — if the job is killed (time-out,
  pre-emption), **just resubmit `run_pipeline.sbatch`** and it continues from
  the last finished model.
* steps 3 and 4 are independent — a failure in one does not block the other.

## Outputs (everything is saved)

```
logs/
  finbench-<jobid>.out / .err     SLURM stdout / stderr
  run_all_<timestamp>.log         full pipeline log (also per-step logs)
results/
  benchmark_results.csv           one row per (feature_set, model, target)
  walkforward_results.csv         one row per walk-forward fold
  report.txt                      benchmark summary (leads with Direction)
  walkforward_report.txt          per-fold + per-regime summary
  run_manifest.json               config snapshot, host, device, timings
  config_snapshot.json            exact config used
  predictions/*.npz               test predictions + truth for every model
  checkpoints/*.pt | *.joblib     trained model weights
  plots/*.png                     ranked metric bar charts, walk-forward curve
```

The benchmark CSV metrics include `accuracy`, `balanced_accuracy`, `f1`, `mcc`
(direction); `mse`, `mae`, `return_ic` (return / volatility).

## Configuration

All knobs live in `finbench/config.py` (`Config` dataclass). Common tweaks:

* `model_families` — subset of `["classical","mlp","sequence","worldmodel"]`
* `feature_sets` — `["ohlc","finance"]`
* `n_epochs`, `seq_epochs`, `wm_epochs` — training length
* `wf_n_folds` — walk-forward folds
* `seq_len`, `*_batch_size`, `num_workers`

Set `FINBENCH_ROOT` to relocate all data/results/logs.

## Notes

* The original `DS2010_Final.ipynb` is kept for reference but is superseded by
  this package.
* PyTorch 2.2.x needs `numpy<2` (pinned in `requirements.txt` / `pyproject.toml`).
# YFin
