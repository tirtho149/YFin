"""Central configuration — every tunable lives here.

A single :class:`Config` dataclass holds all paths, hyperparameters and run
options. ``load_config()`` returns it; the project root can be overridden with
the ``FINBENCH_ROOT`` environment variable (handy on a cluster).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    """All settings for the finbench pipeline."""

    # ---- project root (other directories are derived from it) ----------------
    root: Path = field(
        default_factory=lambda: Path(os.environ.get("FINBENCH_ROOT", Path.cwd()))
    )

    # ---- data download ------------------------------------------------------
    data_start_date: str = "2015-01-01"   # first date of OHLCV history
    min_history_rows: int = 300           # skip tickers with fewer rows
    download_pause: float = 0.3           # seconds between Yahoo requests

    # ---- feature engineering ------------------------------------------------
    direction_threshold: float = 0.005    # |next-day return| neutral band

    # ---- train / val / test date splits ------------------------------------
    train_end: str = "2024-06-30"
    val_end: str = "2025-03-31"
    test_end: str = "2026-03-31"

    # ---- targets & feature sets --------------------------------------------
    # Direction (next close up/flat/down) is the PRIMARY task.
    targets: list = field(
        default_factory=lambda: ["Direction", "LogReturn_Next", "Volatility_Next"]
    )
    primary_target: str = "Direction"
    feature_sets: list = field(default_factory=lambda: ["ohlc", "finance"])
    model_families: list = field(
        default_factory=lambda: ["classical", "mlp", "sequence", "worldmodel"]
    )

    # ---- MLP (deep feed-forward) -------------------------------------------
    batch_size: int = 4096
    n_epochs: int = 20
    learning_rate: float = 1e-3
    model_specs: dict = field(
        default_factory=lambda: {"small": [128, 64], "medium": [256, 128, 64]}
    )

    # ---- classical baselines ----------------------------------------------
    rf_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=300, max_depth=16, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        )
    )
    xgb_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=-1, random_state=42,
        )
    )
    ridge_params: dict = field(default_factory=lambda: dict(alpha=1.0))
    logreg_params: dict = field(default_factory=lambda: dict(C=1.0, max_iter=2000))

    # ---- sequence models (LSTM / Transformer) -----------------------------
    seq_len: int = 30
    seq_epochs: int = 15
    seq_batch_size: int = 2048
    num_workers: int = 8                  # DataLoader workers for windowing
    seq_specs: dict = field(
        default_factory=lambda: {
            "LSTM": dict(kind="lstm", hidden_size=128, num_layers=2, dropout=0.2),
            "Transformer": dict(
                kind="transformer", d_model=128, nhead=4, num_layers=2, dropout=0.2
            ),
        }
    )

    # ---- latent world model -----------------------------------------------
    wm_latent_dim: int = 24
    wm_hidden: int = 128
    wm_epochs: int = 15
    wm_beta: float = 0.5
    wm_gamma: float = 1.0
    wm_sup_weight: float = 5.0

    # ---- walk-forward & regime evaluation ---------------------------------
    wf_n_folds: int = 6
    wf_min_train_frac: float = 0.40

    # ---- run options ------------------------------------------------------
    seed: int = 42
    save_checkpoints: bool = True
    save_predictions: bool = True

    # ---- derived paths (properties so they always track `root`) -----------
    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        """Per-ticker OHLCV CSVs."""
        return self.data_dir / "ohlcv"

    @property
    def meta_dir(self) -> Path:
        """Returns matrix + regime labels."""
        return self.data_dir / "meta"

    @property
    def cache_dir(self) -> Path:
        """Cached engineered feature panels (parquet)."""
        return self.data_dir / "cache"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def predictions_dir(self) -> Path:
        return self.results_dir / "predictions"

    @property
    def checkpoints_dir(self) -> Path:
        return self.results_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    @property
    def returns_path(self) -> Path:
        return self.meta_dir / "returns_matrix.csv"

    @property
    def regime_path(self) -> Path:
        return self.meta_dir / "regime_labels.csv"

    @property
    def benchmark_csv(self) -> Path:
        return self.results_dir / "benchmark_results.csv"

    @property
    def walkforward_csv(self) -> Path:
        return self.results_dir / "walkforward_results.csv"

    def panel_path(self, feature_set: str) -> Path:
        """Cached z-scored panel for one feature set."""
        return self.cache_dir / f"panel_{feature_set}.parquet"

    def ensure_dirs(self) -> None:
        """Create every output directory the pipeline writes to."""
        for d in (
            self.data_dir, self.raw_dir, self.meta_dir, self.cache_dir,
            self.results_dir, self.log_dir, self.predictions_dir,
            self.checkpoints_dir, self.plots_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of the config (for the run manifest)."""
        d = asdict(self)
        d["root"] = str(self.root)
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def load_config() -> Config:
    """Return the default configuration."""
    return Config()
