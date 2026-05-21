"""Benchmark orchestration — trains every model family on every feature set and
target, saving results / predictions / checkpoints incrementally so a run can be
killed and resumed without losing work.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_returns_and_meta
from .features import get_feature_cols
from .logging_utils import get_logger, set_seed
from .metrics import classification_metrics, daily_rank_ic, regression_metrics
from .models.classical import classical_classifiers, classical_regressors
from .models.mlp import (MLPClassifier, MLPRegressor, make_loaders,
                         train_classifier, train_regressor)
from .models.sequence import build_sequence_model, make_sequence_loaders
from .models.world_model import WorldModel, train_world_model
from .panel import (build_sequence_panel, encode_sector, make_splits,
                    merge_train_val, prepare_panel)

REG_TARGETS = ("LogReturn_Next", "Volatility_Next")


def get_device() -> torch.device:
    """Return the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Incremental, resumable results writer
# ---------------------------------------------------------------------------
class ResultsWriter:
    """Accumulates metric rows and rewrites the results CSV after every model,
    so a crash never loses completed work. Successful (non-error) rows are
    treated as 'done' for resume; failed rows are retried on re-run.
    """

    def __init__(self, csv_path: Path, logger):
        self.csv_path = Path(csv_path)
        self.logger = logger
        self.rows: list[dict] = []
        self.done: set = set()
        if self.csv_path.is_file():
            existing = pd.read_csv(self.csv_path)
            # Drop stale error rows on load: they carry no 'done' info and the
            # model is retried anyway, so keeping them would leave a duplicate
            # error row next to the eventual success row.
            self.rows = [r for r in existing.to_dict("records")
                         if not _is_error_row(r)]
            for r in self.rows:
                self.done.add((r["feature_set"], r["model_name"], r["target"]))
            logger.info("resuming — %d completed rows", len(self.done))

    def is_done(self, feature_set, model_name, target) -> bool:
        return (feature_set, model_name, target) in self.done

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if not _is_error_row(row):
            self.done.add((row["feature_set"], row["model_name"], row["target"]))
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _is_error_row(row: dict) -> bool:
    err = row.get("error")
    return err is not None and not (isinstance(err, float) and pd.isna(err))


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------
def _save_predictions(cfg, feature_set, model_name, target, preds, y_true,
                      dates, tickers) -> None:
    if not cfg.save_predictions:
        return
    path = cfg.predictions_dir / f"{feature_set}__{model_name}__{target}.npz"
    np.savez_compressed(
        path,
        preds=np.asarray(preds),
        y_true=np.asarray(y_true),
        dates=np.asarray(dates).astype("datetime64[ns]"),
        tickers=np.asarray(tickers).astype(str),
    )


def _save_torch_ckpt(cfg, feature_set, model_name, target, model) -> None:
    if not cfg.save_checkpoints:
        return
    torch.save(model.state_dict(),
               cfg.checkpoints_dir / f"{feature_set}__{model_name}__{target}.pt")


def _save_sklearn_ckpt(cfg, feature_set, model_name, target, model) -> None:
    if not cfg.save_checkpoints:
        return
    import joblib

    joblib.dump(model,
                cfg.checkpoints_dir / f"{feature_set}__{model_name}__{target}.joblib",
                compress=3)


def _finish(writer, cfg, feature_set, model_type, model_name, target,
            metrics, elapsed, logger, preds=None, y_true=None,
            dates=None, tickers=None, error=None) -> None:
    """Record one model's result row, save its predictions, and log it."""
    row = {
        "feature_set": feature_set, "model_type": model_type,
        "model_name": model_name, "target": target,
        "train_seconds": round(elapsed, 1),
    }
    if error is not None:
        row["error"] = error
    row.update(metrics or {})
    writer.add(row)
    if error is None and preds is not None:
        _save_predictions(cfg, feature_set, model_name, target, preds,
                          y_true, dates, tickers)
        shown = {k: round(v, 4) for k, v in (metrics or {}).items()
                 if isinstance(v, (int, float))}
        logger.info("  OK  %-9s %-18s %-15s %5.1fs  %s",
                    model_type, model_name, target, elapsed, shown)
    else:
        logger.error("  FAIL %-9s %-18s %-15s  %s",
                     model_type, model_name, target, error)


# ---------------------------------------------------------------------------
# Model family runners
# ---------------------------------------------------------------------------
def run_classical(cfg, feature_set, splits, writer, device, logger) -> None:
    """XGBoost / RandomForest / Ridge / LogisticRegression on the panel splits."""
    train_split, val_split, test_split = splits
    X_full, yr, yd, yv = merge_train_val(train_split, val_split)
    X_te, yr_te, yd_te, yv_te, idx_te = test_split
    test_dates = idx_te.get_level_values(0)
    test_tickers = idx_te.get_level_values(1)

    for target in cfg.targets:
        is_clf = target == "Direction"
        if is_clf:
            models = classical_classifiers(cfg, device)
            y_full, y_te = yd + 1, yd_te + 1          # {-1,0,1} -> {0,1,2}
        else:
            models = classical_regressors(cfg, device)
            y_full = yr if target == "LogReturn_Next" else yv
            y_te = yr_te if target == "LogReturn_Next" else yv_te
        if y_full is None:
            continue
        for name, factory in models.items():
            if writer.is_done(feature_set, name, target):
                continue
            t0 = time.time()
            try:
                mdl = factory()
                mdl.fit(X_full, y_full)
                preds = mdl.predict(X_te)
                if is_clf:
                    metrics = classification_metrics(y_te, preds)
                else:
                    metrics = regression_metrics(y_te, preds)
                    if target == "LogReturn_Next":
                        ic, icir, _ = daily_rank_ic(test_dates, y_te, preds)
                        metrics.update({"return_ic": ic, "return_icir": icir})
                _save_sklearn_ckpt(cfg, feature_set, name, target, mdl)
                _finish(writer, cfg, feature_set, "classical", name, target,
                        metrics, time.time() - t0, logger, preds, y_te,
                        test_dates, test_tickers)
            except Exception as e:                   # noqa: BLE001
                _finish(writer, cfg, feature_set, "classical", name, target,
                        {}, time.time() - t0, logger, error=str(e))


def run_mlp(cfg, feature_set, splits, writer, device, logger) -> None:
    """Feed-forward MLP regressor / classifier on the panel splits."""
    train_split, val_split, test_split = splits
    X_va, yr_va, yd_va, yv_va, _ = val_split
    X_te, yr_te, yd_te, yv_te, idx_te = test_split
    X_full, yr, yd, yv = merge_train_val(train_split, val_split)
    input_dim = X_full.shape[1]
    test_dates = idx_te.get_level_values(0)
    test_tickers = idx_te.get_level_values(1)

    for target in cfg.targets:
        is_clf = target == "Direction"
        if is_clf:
            y_full, y_va, y_te = yd + 1, yd_va + 1, yd_te + 1
        elif target == "LogReturn_Next":
            y_full, y_va, y_te = yr, yr_va, yr_te
        else:
            y_full, y_va, y_te = yv, yv_va, yv_te
        if y_full is None:
            continue
        for size, hidden in cfg.model_specs.items():
            name = f"MLP_{size}"
            if writer.is_done(feature_set, name, target):
                continue
            t0 = time.time()
            try:
                tr_l, va_l = make_loaders(X_full, y_full, X_va, y_va, cfg.batch_size)
                if is_clf:
                    model = MLPClassifier(input_dim, hidden, n_classes=3)
                    model = train_classifier(model, tr_l, va_l, device,
                                             cfg.n_epochs, cfg.learning_rate)
                else:
                    model = MLPRegressor(input_dim, hidden)
                    model = train_regressor(model, tr_l, va_l, device,
                                            cfg.n_epochs, cfg.learning_rate)
                model.eval()
                with torch.no_grad():
                    out = model(torch.from_numpy(X_te.astype(np.float32)).to(device))
                    out = out.cpu().numpy()
                if is_clf:
                    preds = np.argmax(out, axis=1)
                    metrics = classification_metrics(y_te, preds)
                else:
                    preds = out
                    metrics = regression_metrics(y_te, preds)
                    if target == "LogReturn_Next":
                        ic, icir, _ = daily_rank_ic(test_dates, y_te, preds)
                        metrics.update({"return_ic": ic, "return_icir": icir})
                _save_torch_ckpt(cfg, feature_set, name, target, model)
                _finish(writer, cfg, feature_set, "mlp", name, target,
                        metrics, time.time() - t0, logger, preds, y_te,
                        test_dates, test_tickers)
            except Exception as e:                   # noqa: BLE001
                _finish(writer, cfg, feature_set, "mlp", name, target,
                        {}, time.time() - t0, logger, error=str(e))


def _predict_sequence(model, test_loader, device, is_clf, world_model=False):
    """Run a sequence / world model over the test loader -> numpy predictions."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for xb, _ in test_loader:
            out = model(xb.to(device))
            if world_model:
                out = out[0]
            chunks.append(out.cpu().numpy())
    preds = np.concatenate(chunks) if chunks else np.array([])
    return np.argmax(preds, axis=1) if is_clf else preds


def run_sequence(cfg, feature_set, seq_bundle, writer, device, logger) -> None:
    """LSTM / Transformer over the look-back windows."""
    if not seq_bundle["index"]["test"]:
        logger.warning("  no sequence windows — skipping sequence models")
        return
    input_dim = next(iter(seq_bundle["feat"].values())).shape[1]
    for target in cfg.targets:
        is_clf = target == "Direction"
        loaders = make_sequence_loaders(
            seq_bundle, target, cfg.seq_len, cfg.seq_batch_size,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
        fit_l, val_l, test_l, test_dates, y_te, test_tickers = loaders
        for name, spec in cfg.seq_specs.items():
            if writer.is_done(feature_set, name, target):
                continue
            t0 = time.time()
            try:
                n_out = 3 if is_clf else 1
                model = build_sequence_model(spec, input_dim, n_out)
                trainer = train_classifier if is_clf else train_regressor
                model = trainer(model, fit_l, val_l, device,
                                cfg.seq_epochs, cfg.learning_rate)
                preds = _predict_sequence(model, test_l, device, is_clf)
                if is_clf:
                    metrics = classification_metrics(y_te, preds)
                else:
                    metrics = regression_metrics(y_te, preds)
                    if target == "LogReturn_Next":
                        ic, icir, _ = daily_rank_ic(test_dates, y_te, preds)
                        metrics.update({"return_ic": ic, "return_icir": icir})
                _save_torch_ckpt(cfg, feature_set, name, target, model)
                _finish(writer, cfg, feature_set, "sequence", name, target,
                        metrics, time.time() - t0, logger, preds, y_te,
                        test_dates, test_tickers)
            except Exception as e:                   # noqa: BLE001
                _finish(writer, cfg, feature_set, "sequence", name, target,
                        {}, time.time() - t0, logger, error=str(e))


def run_worldmodel(cfg, feature_set, seq_bundle, writer, device, logger) -> None:
    """Latent world model over the look-back windows."""
    if not seq_bundle["index"]["test"]:
        logger.warning("  no sequence windows — skipping world model")
        return
    input_dim = next(iter(seq_bundle["feat"].values())).shape[1]
    for target in cfg.targets:
        is_clf = target == "Direction"
        name = "WorldModel"
        if writer.is_done(feature_set, name, target):
            continue
        t0 = time.time()
        try:
            loaders = make_sequence_loaders(
                seq_bundle, target, cfg.seq_len, cfg.seq_batch_size,
                num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
            fit_l, val_l, test_l, test_dates, y_te, test_tickers = loaders
            n_out = 3 if is_clf else 1
            model = WorldModel(input_dim, cfg.wm_latent_dim, cfg.wm_hidden, n_out)
            model = train_world_model(model, fit_l, val_l, is_clf, device,
                                      cfg.wm_epochs, cfg.learning_rate,
                                      cfg.wm_beta, cfg.wm_gamma, cfg.wm_sup_weight)
            preds = _predict_sequence(model, test_l, device, is_clf,
                                      world_model=True)
            if is_clf:
                metrics = classification_metrics(y_te, preds)
            else:
                metrics = regression_metrics(y_te, preds)
                if target == "LogReturn_Next":
                    ic, icir, _ = daily_rank_ic(test_dates, y_te, preds)
                    metrics.update({"return_ic": ic, "return_icir": icir})
            _save_torch_ckpt(cfg, feature_set, name, target, model)
            _finish(writer, cfg, feature_set, "worldmodel", name, target,
                    metrics, time.time() - t0, logger, preds, y_te,
                    test_dates, test_tickers)
        except Exception as e:                       # noqa: BLE001
            _finish(writer, cfg, feature_set, "worldmodel", name, target,
                    {}, time.time() - t0, logger, error=str(e))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_TARGET_METRIC = {
    "Direction": ("accuracy", True),
    "LogReturn_Next": ("return_ic", True),
    "Volatility_Next": ("mse", False),
}


def write_report(cfg, df: pd.DataFrame, logger) -> None:
    """Write a human-readable summary, leading with the primary (Direction) task."""
    path = cfg.results_dir / "report.txt"
    lines: list[str] = []

    def emit(s=""):
        lines.append(s)

    emit("=" * 70)
    emit("finbench — close-price DIRECTION prediction benchmark")
    emit("=" * 70)
    if df.empty:
        emit("no results."); path.write_text("\n".join(lines)); return

    ok = df[~df.apply(lambda r: _is_error_row(r.to_dict()), axis=1)].copy()
    ordered = [cfg.primary_target] + [t for t in cfg.targets if t != cfg.primary_target]

    for target in ordered:
        metric, higher = _TARGET_METRIC.get(target, ("accuracy", True))
        sub = ok[(ok["target"] == target)]
        if metric not in sub.columns or sub.dropna(subset=[metric]).empty:
            continue
        sub = sub.dropna(subset=[metric])
        emit()
        emit(f"### {target}   [metric: {metric}, {'higher' if higher else 'lower'} better]")
        best = sub.sort_values(metric, ascending=not higher).iloc[0]
        emit(f"  BEST OVERALL : {best[metric]:+.4f}  "
             f"({best['feature_set']}/{best['model_type']}/{best['model_name']})")
        emit("  by model type:")
        for mt in ("classical", "mlp", "sequence", "worldmodel"):
            s = sub[sub["model_type"] == mt]
            if s.empty:
                continue
            r = s.sort_values(metric, ascending=not higher).iloc[0]
            emit(f"    {mt:<11s} {r[metric]:+.4f}  ({r['feature_set']}/{r['model_name']})")
        emit("  by feature set:")
        for fs in cfg.feature_sets:
            s = sub[sub["feature_set"] == fs]
            if s.empty:
                continue
            r = s.sort_values(metric, ascending=not higher).iloc[0]
            emit(f"    {fs:<11s} {r[metric]:+.4f}  ({r['model_type']}/{r['model_name']})")

    n_err = int(df.apply(lambda r: _is_error_row(r.to_dict()), axis=1).sum())
    emit()
    emit(f"total result rows: {len(df)}   (errors: {n_err})")
    path.write_text("\n".join(lines))
    logger.info("report written -> %s", path)
    for ln in lines:
        logger.info("%s", ln)


def make_plots(cfg, df: pd.DataFrame, logger) -> None:
    """Save one ranked bar chart per target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return
    ok = df[~df.apply(lambda r: _is_error_row(r.to_dict()), axis=1)].copy()
    ok["label"] = ok["feature_set"] + "/" + ok["model_name"]
    for target in cfg.targets:
        metric, higher = _TARGET_METRIC.get(target, ("accuracy", True))
        sub = ok[ok["target"] == target]
        if metric not in sub.columns:
            continue
        sub = sub.dropna(subset=[metric]).sort_values(metric, ascending=higher)
        if sub.empty:
            continue
        plt.figure(figsize=(8, max(4, 0.4 * len(sub))))
        plt.barh(sub["label"], sub[metric])
        plt.xlabel(metric)
        plt.title(f"{target} — {metric}")
        plt.tight_layout()
        out = cfg.plots_dir / f"{target}_{metric}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("plot saved -> %s", out)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def run_benchmark(cfg) -> pd.DataFrame:
    """Train every configured model family on every feature set and target."""
    log = get_logger()
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    cfg.save(cfg.results_dir / "config_snapshot.json")
    device = get_device()
    log.info("device=%s  families=%s  feature_sets=%s  targets=%s",
             device, cfg.model_families, cfg.feature_sets, cfg.targets)

    ret, regime_int, sector_map, tickers, dates = load_returns_and_meta(cfg)
    log.info("metadata: %d tickers, %d dates", len(tickers), len(dates))

    writer = ResultsWriter(cfg.benchmark_csv, log)
    t_start = time.time()

    for feature_set in cfg.feature_sets:
        log.info("=" * 60)
        log.info("FEATURE SET: %s", feature_set)
        panel = prepare_panel(cfg, feature_set, tickers, dates)
        feature_cols = get_feature_cols(panel)
        sector_enc = encode_sector(sector_map, tickers)
        panel_dates = sorted(panel["date"].unique())
        regime_ser = (regime_int.reindex(panel_dates).ffill().bfill()
                      .fillna(1).astype(int))
        splits = make_splits(panel, feature_cols, regime_ser, sector_enc,
                             cfg.train_end, cfg.val_end, cfg.test_end)
        log.info("panel: %d features, %d/%d/%d train/val/test rows",
                 len(feature_cols), len(splits[0][0]), len(splits[1][0]),
                 len(splits[2][0]))

        if "classical" in cfg.model_families:
            log.info("-- classical baselines --")
            run_classical(cfg, feature_set, splits, writer, device, log)
        if "mlp" in cfg.model_families:
            log.info("-- MLP --")
            run_mlp(cfg, feature_set, splits, writer, device, log)

        if {"sequence", "worldmodel"} & set(cfg.model_families):
            seq_bundle = build_sequence_panel(
                panel, feature_cols, regime_ser, sector_enc, cfg.seq_len,
                cfg.train_end, cfg.val_end, cfg.test_end)
            idx = seq_bundle["index"]
            log.info("sequence windows: %d train+val, %d test",
                     len(idx["train"]) + len(idx["val"]), len(idx["test"]))
            if "sequence" in cfg.model_families:
                log.info("-- sequence models --")
                run_sequence(cfg, feature_set, seq_bundle, writer, device, log)
            if "worldmodel" in cfg.model_families:
                log.info("-- latent world model --")
                run_worldmodel(cfg, feature_set, seq_bundle, writer, device, log)

    results = writer.frame()
    log.info("benchmark done in %.1f min — %d rows -> %s",
             (time.time() - t_start) / 60, len(results), cfg.benchmark_csv)
    write_report(cfg, results, log)
    make_plots(cfg, results, log)
    return results
