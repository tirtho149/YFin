"""Evaluation metrics — regression, classification (with Stage-5 richer metrics)
and the daily cross-sectional rank IC used for the return target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)

EPS = 1e-12


def regression_metrics(y_true, y_pred) -> dict:
    """MSE and MAE for a regression target."""
    if y_true is None or y_pred is None:
        return {}
    yt = np.asarray(y_true).ravel().astype(float)
    yp = np.asarray(y_pred).ravel().astype(float)
    return {
        "mse": float(mean_squared_error(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
    }


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, balanced accuracy, macro precision / recall / F1, MCC and AUC.

    Balanced accuracy, F1 and MCC are robust to the Direction class imbalance.
    """
    if y_true is None or y_pred is None:
        return {}
    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)
    try:
        auc = roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else 0.5
    except Exception:               # noqa: BLE001
        auc = 0.5
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "auc": float(auc),
    }


def daily_rank_ic(dates_arr, y_true, y_pred):
    """Mean daily cross-sectional Spearman rank IC and its ICIR (mean / std)."""
    if y_true is None or y_pred is None:
        return np.nan, np.nan, pd.Series(dtype=float)
    dates = np.asarray(dates_arr)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    daily = {}
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 5:
            continue
        rho, _ = spearmanr(yt[mask], yp[mask])
        if np.isfinite(rho):
            daily[d] = rho
    if not daily:
        return np.nan, np.nan, pd.Series(dtype=float)
    s = pd.Series(daily)
    mean_ic, std_ic = s.mean(), s.std()
    icir = mean_ic / std_ic if (std_ic and std_ic > EPS) else np.nan
    return float(mean_ic), float(icir), s


def brier_score(proba, y_true) -> float:
    """Multiclass Brier score — mean squared error of predicted probabilities."""
    proba = np.asarray(proba, dtype=float)
    yt = np.asarray(y_true).ravel().astype(int)
    onehot = np.eye(proba.shape[1])[yt]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
