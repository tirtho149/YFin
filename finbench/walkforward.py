"""Walk-forward validation, regime-split testing and calibration (Stage 5).

Re-trains a representative model (XGBoost on the Direction target) on expanding
time folds, labels each test day bull / sideways / volatile, and reports
per-fold and per-regime metrics plus a Brier calibration score.
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from .data import load_returns_and_meta
from .features import get_feature_cols
from .logging_utils import get_logger
from .metrics import brier_score, classification_metrics
from .models.classical import HAS_XGB
from .panel import encode_sector, prepare_panel


def _xgb_device() -> str:
    """'cuda' if a GPU is visible to torch, else 'cpu' (XGBoost tree device)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:               # noqa: BLE001
        return "cpu"


def label_market_regimes(ret: pd.DataFrame) -> pd.Series:
    """Label each trading day bull / sideways / volatile from the equal-weight
    market return: high rolling volatility -> 'volatile' (captures crises such as
    the COVID crash), else a positive quarterly trend -> 'bull', else
    'sideways'."""
    mkt = ret.mean(axis=1)
    trend = mkt.rolling(63).sum()
    vol = mkt.rolling(21).std()
    labels = pd.Series("sideways", index=ret.index)
    labels[trend > 0.03] = "bull"
    labels[vol > vol.quantile(0.80)] = "volatile"
    return labels


def make_walkforward_folds(all_dates, n_folds, min_train_frac):
    """Expanding-window folds: reserve the first ``min_train_frac`` of the
    timeline as the initial training floor, then tile the remainder into
    ``n_folds`` contiguous test blocks."""
    all_dates = sorted(pd.to_datetime(all_dates))
    floor = int(len(all_dates) * min_train_frac)
    blocks = [b for b in np.array_split(all_dates[floor:], n_folds) if len(b)]
    folds = []
    for blk in blocks:
        test_dates = list(pd.to_datetime(blk))
        train_dates = [d for d in all_dates if d < test_dates[0]]
        if train_dates and test_dates:
            folds.append((train_dates, test_dates))
    return folds


def wf_features(panel, feature_cols, regime_ser, sector_enc, date_set):
    """Assemble (X, y_dir, dates, tickers) for the panel rows on the given dates.
    Direction labels are remapped {-1,0,1} -> {0,1,2}."""
    sub = panel[panel["date"].isin(date_set)]
    if sub.empty:
        empty = np.empty((0, len(feature_cols) + 2), np.float32)
        return empty, np.empty(0, int), np.empty(0), np.empty(0)
    X = sub[feature_cols].fillna(0).values.astype(np.float32)
    reg = (regime_ser.reindex(sub["date"]).ffill().bfill().fillna(1)
           .values.reshape(-1, 1).astype(np.float32))
    sec = (sub["ticker"].map(lambda t: sector_enc.get(t, 0))
           .values.reshape(-1, 1).astype(np.float32))
    X = np.hstack([X, reg, sec])
    y = sub["Direction"].values.astype(int) + 1
    return X, y, sub["date"].values, sub["ticker"].values


def run_walkforward(cfg) -> pd.DataFrame:
    """Walk-forward + regime-split + calibration for XGBoost on Direction."""
    log = get_logger()
    cfg.ensure_dirs()
    log.info("=" * 60)
    log.info("WALK-FORWARD & REGIME EVALUATION  (XGBoost / Direction)")
    if not HAS_XGB:
        log.error("xgboost unavailable — walk-forward skipped.")
        return pd.DataFrame()
    from xgboost import XGBClassifier

    ret, regime_int, sector_map, tickers, dates = load_returns_and_meta(cfg)
    regimes = label_market_regimes(ret)
    sector_enc = encode_sector(sector_map, tickers)
    xgb_params = dict(cfg.xgb_params)
    xgb_params["device"] = _xgb_device()

    fold_rows: list[dict] = []
    pred_chunks: list[dict] = []
    t_start = time.time()

    for feature_set in cfg.feature_sets:
        log.info("--- feature set: %s ---", feature_set)
        panel = prepare_panel(cfg, feature_set, tickers, dates)
        feature_cols = get_feature_cols(panel)
        panel_dates = sorted(panel["date"].unique())
        regime_ser = (regime_int.reindex(panel_dates).ffill().bfill()
                      .fillna(1).astype(int))
        folds = make_walkforward_folds(panel_dates, cfg.wf_n_folds,
                                       cfg.wf_min_train_frac)
        for fi, (train_dates, test_dates) in enumerate(folds, 1):
            X_tr, y_tr, _, _ = wf_features(panel, feature_cols, regime_ser,
                                           sector_enc, set(train_dates))
            X_te, y_te, d_te, tk_te = wf_features(panel, feature_cols, regime_ser,
                                                  sector_enc, set(test_dates))
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            t0 = time.time()
            try:
                mdl = XGBClassifier(**xgb_params)
                mdl.fit(X_tr, y_tr)
                proba = mdl.predict_proba(X_te)
                preds = np.argmax(proba, axis=1)
            except Exception as e:                   # noqa: BLE001
                log.error("  fold %d (%s): ERROR %s", fi, feature_set, e)
                continue
            met = classification_metrics(y_te, preds)
            met["brier"] = brier_score(proba, y_te)
            row = {
                "feature_set": feature_set, "fold": fi,
                "test_start": str(pd.Timestamp(min(test_dates)).date()),
                "test_end": str(pd.Timestamp(max(test_dates)).date()),
                "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                "train_seconds": round(time.time() - t0, 1),
            }
            row.update(met)
            fold_rows.append(row)
            reg_lbl = (regimes.reindex(pd.DatetimeIndex(d_te))
                       .fillna("sideways").values)
            pred_chunks.append(dict(feature_set=feature_set, fold=fi,
                                    y_true=y_te, y_pred=preds, proba=proba,
                                    dates=d_te, tickers=tk_te, regime=reg_lbl))
            log.info("  fold %d  %s..%s  acc=%.3f  bal_acc=%.3f  mcc=%+.3f  "
                     "brier=%.3f", fi, row["test_start"], row["test_end"],
                     met["accuracy"], met["balanced_accuracy"], met["mcc"],
                     met["brier"])

    if not fold_rows:
        log.warning("no walk-forward folds produced.")
        return pd.DataFrame()

    wf_df = pd.DataFrame(fold_rows)
    wf_df.to_csv(cfg.walkforward_csv, index=False)
    log.info("walk-forward results -> %s", cfg.walkforward_csv)

    # save all walk-forward predictions in one archive
    np.savez_compressed(
        cfg.predictions_dir / "walkforward_predictions.npz",
        y_true=np.concatenate([c["y_true"] for c in pred_chunks]),
        y_pred=np.concatenate([c["y_pred"] for c in pred_chunks]),
        dates=np.concatenate([c["dates"] for c in pred_chunks]).astype("datetime64[ns]"),
        tickers=np.concatenate([c["tickers"] for c in pred_chunks]).astype(str),
        regime=np.concatenate([c["regime"] for c in pred_chunks]).astype(str),
        feature_set=np.concatenate(
            [np.full(len(c["y_true"]), c["feature_set"]) for c in pred_chunks]).astype(str),
        fold=np.concatenate(
            [np.full(len(c["y_true"]), c["fold"]) for c in pred_chunks]),
    )

    _write_wf_report(cfg, wf_df, pred_chunks, log)
    _plot_walkforward(cfg, wf_df, log)
    log.info("walk-forward done in %.1f min", (time.time() - t_start) / 60)
    return wf_df


def _write_wf_report(cfg, wf_df, pred_chunks, log) -> None:
    """Per-fold summary + regime-split metrics -> walkforward_report.txt + log."""
    lines = ["=" * 70,
             "finbench — WALK-FORWARD & REGIME EVALUATION (XGBoost / Direction)",
             "=" * 70, "", "WALK-FORWARD SUMMARY  (mean +/- std across folds)"]
    for fs in cfg.feature_sets:
        sub = wf_df[wf_df["feature_set"] == fs]
        if sub.empty:
            continue
        lines.append(
            f"  {fs:<10s} acc={sub['accuracy'].mean():.3f}+/-{sub['accuracy'].std():.3f}"
            f"  bal_acc={sub['balanced_accuracy'].mean():.3f}"
            f"  mcc={sub['mcc'].mean():+.3f}"
            f"  f1={sub['f1'].mean():.3f}"
            f"  brier={sub['brier'].mean():.3f}")

    lines += ["", "REGIME-SPLIT TESTING  (pooled across folds & feature sets)"]
    Y = np.concatenate([c["y_true"] for c in pred_chunks])
    P = np.concatenate([c["y_pred"] for c in pred_chunks])
    R = np.concatenate([c["regime"] for c in pred_chunks])
    for reg in ("bull", "sideways", "volatile"):
        m = R == reg
        if m.sum() == 0:
            lines.append(f"  {reg:<10s} (no test days)")
            continue
        met = classification_metrics(Y[m], P[m])
        lines.append(f"  {reg:<10s} n={int(m.sum()):>8d}  acc={met['accuracy']:.3f}"
                     f"  bal_acc={met['balanced_accuracy']:.3f}"
                     f"  mcc={met['mcc']:+.3f}  f1={met['f1']:.3f}")

    path = cfg.results_dir / "walkforward_report.txt"
    path.write_text("\n".join(lines))
    log.info("walk-forward report -> %s", path)
    for ln in lines:
        log.info("%s", ln)


def _plot_walkforward(cfg, wf_df, log) -> None:
    """Accuracy across walk-forward folds, one line per feature set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.figure(figsize=(8, 4))
        for fs in cfg.feature_sets:
            sub = wf_df[wf_df["feature_set"] == fs]
            if not sub.empty:
                plt.plot(sub["fold"], sub["accuracy"], marker="o", label=fs)
        plt.xlabel("walk-forward fold")
        plt.ylabel("accuracy")
        plt.title("Walk-forward accuracy by fold — XGBoost / Direction")
        plt.legend()
        plt.tight_layout()
        out = cfg.plots_dir / "walkforward_accuracy.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("plot saved -> %s", out)
    except Exception as e:                           # noqa: BLE001
        log.warning("walk-forward plot skipped: %s", e)
