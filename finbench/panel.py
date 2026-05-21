"""Panel construction — cross-sectional (date, ticker) panels, train/val/test
splits, and the lazy look-back windows used by the sequence / world models.

This module is intentionally torch-free so the panel-building step can run on a
CPU-only node.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .features import EPS, FEATURE_SETS, get_feature_cols
from .logging_utils import get_logger


def build_feature_panel(raw_dir, tickers, dates, feature_fn,
                        min_history_rows, direction_threshold) -> pd.DataFrame:
    """Build a long (date, ticker) panel of engineered features + targets."""
    log = get_logger()
    keep = set(pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="coerce")).dropna())
    frames, skipped = [], 0
    for i, sym in enumerate(tickers, 1):
        if i % 50 == 0:
            log.info("  building features %d/%d ...", i, len(tickers))
        path = raw_dir / f"{sym}.csv"
        if not path.is_file():
            skipped += 1
            continue
        try:
            raw = pd.read_csv(path, index_col=0)
            raw.index = pd.to_datetime(raw.index, errors="coerce")
            raw = raw[~raw.index.isna()].sort_index()
            if "Close" not in raw.columns or len(raw) < min_history_rows:
                skipped += 1
                continue
            feat = feature_fn(raw, direction_threshold)
            if feat.empty or "LogReturn_Next" not in feat.columns:
                skipped += 1
                continue
            feat = feat[feat.index.isin(keep)]
            if feat.empty:
                skipped += 1
                continue
            feat = feat.reset_index().rename(columns={feat.reset_index().columns[0]: "date"})
            feat.insert(1, "ticker", sym)
            frames.append(feat)
        except Exception:           # noqa: BLE001
            skipped += 1
    if skipped:
        log.info("  skipped %d tickers", skipped)
    if not frames:
        raise RuntimeError("No feature rows built — check the OHLCV CSVs.")
    return pd.concat(frames, ignore_index=True)


def zscore_cross_sectional(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Z-score each feature cross-sectionally (across tickers) within each day.

    Vectorised with groupby-transform — far faster than a per-(date, column)
    Python loop.
    """
    out = panel.copy()
    grp = out.groupby("date")
    mu = grp[feature_cols].transform("mean")
    sd = grp[feature_cols].transform("std")
    z = (out[feature_cols] - mu) / sd
    out[feature_cols] = z.where(sd > EPS, 0.0).fillna(0.0)
    return out


def encode_sector(sector_map: dict, tickers: list) -> dict:
    """Deterministically integer-encode sector labels."""
    le = LabelEncoder()
    labels = [sector_map.get(t, "Other") for t in tickers]
    le.fit(labels)
    return {t: int(le.transform([sector_map.get(t, "Other")])[0]) for t in tickers}


def make_splits(panel, feature_cols, regime_ser, sector_enc,
                train_end, val_end, test_end):
    """Split the panel into train/val/test by date, appending regime and sector
    columns to the feature matrix. Returns three (X, y_ret, y_dir, y_vol, index)
    tuples."""
    p = panel.set_index(["date", "ticker"])
    all_dates = sorted(p.index.get_level_values(0).unique())
    train_dates = [d for d in all_dates if d <= pd.Timestamp(train_end)]
    val_dates = [d for d in all_dates
                 if pd.Timestamp(train_end) < d <= pd.Timestamp(val_end)]
    test_dates = [d for d in all_dates
                  if pd.Timestamp(val_end) < d <= pd.Timestamp(test_end)]

    def get_split(date_list):
        mask = p.index.get_level_values(0).isin(date_list)
        sub = p.loc[mask].copy().dropna(subset=feature_cols, how="all")
        X = sub[feature_cols].fillna(0).values.astype(np.float32)
        sec = np.array([sector_enc.get(t, 0) for t in sub.index.get_level_values(1)],
                       dtype=np.float32).reshape(-1, 1)
        reg = (regime_ser.reindex(sub.index.get_level_values(0))
               .values.reshape(-1, 1).astype(np.float32))
        X = np.hstack([X, reg, sec])
        y_ret = sub["LogReturn_Next"].values if "LogReturn_Next" in sub.columns else None
        y_dir = sub["Direction"].values if "Direction" in sub.columns else None
        y_vol = sub["Volatility_Next"].values if "Volatility_Next" in sub.columns else None
        return X, y_ret, y_dir, y_vol, sub.index

    return get_split(train_dates), get_split(val_dates), get_split(test_dates)


def merge_train_val(train_split, val_split):
    """Concatenate train + val into one fit set. Returns (X, y_ret, y_dir, y_vol)."""
    X_tr, yr_tr, yd_tr, yv_tr, _ = train_split
    X_va, yr_va, yd_va, yv_va, _ = val_split
    X_full = np.vstack([X_tr, X_va])
    yr = np.concatenate([yr_tr, yr_va]) if yr_tr is not None else None
    yd = np.concatenate([yd_tr, yd_va]) if yd_tr is not None else None
    yv = np.concatenate([yv_tr, yv_va]) if yv_tr is not None else None
    return X_full, yr, yd, yv


def build_sequence_panel(panel, feature_cols, regime_ser, sector_enc, seq_len,
                         train_end, val_end, test_end) -> dict:
    """Turn the z-scored panel into per-ticker feature / target / date arrays
    plus a window index split by each window's last (prediction) date.

    The per-timestep feature vector matches the cross-sectional input
    (feature_cols + regime + sector). Returns a dict with keys
    ``feat``, ``targets``, ``dates``, ``index``.
    """
    feat_by_ticker, targets_by_ticker, dates_by_ticker = {}, {}, {}
    index = {"train": [], "val": [], "test": []}
    t_end, v_end, x_end = (pd.Timestamp(train_end), pd.Timestamp(val_end),
                           pd.Timestamp(test_end))

    for ticker, grp in panel.groupby("ticker", sort=False):
        grp = grp.sort_values("date")
        if len(grp) < seq_len:
            continue
        feats = grp[feature_cols].fillna(0).values.astype(np.float32)
        dts = pd.DatetimeIndex(grp["date"].values)
        reg = (regime_ser.reindex(dts).ffill().bfill().fillna(1)
               .values.reshape(-1, 1).astype(np.float32))
        sec = np.full((len(grp), 1), float(sector_enc.get(ticker, 0)), dtype=np.float32)
        feat_by_ticker[ticker] = np.hstack([feats, reg, sec])
        dates_by_ticker[ticker] = dts.values
        targets_by_ticker[ticker] = {
            "LogReturn_Next": grp["LogReturn_Next"].values.astype(np.float32),
            "Volatility_Next": grp["Volatility_Next"].values.astype(np.float32),
            "Direction": grp["Direction"].values.astype(np.int64),
        }
        for i in range(seq_len - 1, len(grp)):
            d = dts[i]
            if d <= t_end:
                index["train"].append((ticker, i))
            elif d <= v_end:
                index["val"].append((ticker, i))
            elif d <= x_end:
                index["test"].append((ticker, i))

    return {"feat": feat_by_ticker, "targets": targets_by_ticker,
            "dates": dates_by_ticker, "index": index}


def prepare_panel(cfg, feature_set: str, tickers, dates) -> pd.DataFrame:
    """Return the z-scored feature panel for one feature set, using the parquet
    cache when present and building + caching it otherwise."""
    log = get_logger()
    cache = cfg.panel_path(feature_set)
    if cache.is_file():
        log.info("loading cached panel: %s", cache)
        return pd.read_parquet(cache)
    log.info("building panel for feature set '%s' ...", feature_set)
    panel = build_feature_panel(
        cfg.raw_dir, tickers, dates, FEATURE_SETS[feature_set],
        cfg.min_history_rows, cfg.direction_threshold,
    )
    feature_cols = get_feature_cols(panel)
    panel = zscore_cross_sectional(panel, feature_cols)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache)
    log.info("cached panel -> %s  (%d rows, %d features)",
             cache, len(panel), len(feature_cols))
    return panel
