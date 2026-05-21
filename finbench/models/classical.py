"""Classical ML baselines — XGBoost, RandomForest, Ridge / LogisticRegression.

These establish the non-deep-learning floor. ``HAS_XGB`` is False if xgboost is
not importable, in which case the XGBoost baselines are silently skipped.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:                 # pragma: no cover
    HAS_XGB = False


def _xgb_params(xgb_params: dict, device) -> dict:
    """Copy of the XGBoost params with the runtime device injected.

    ``device`` may be a ``torch.device`` or a string; XGBoost wants a plain
    string ("cuda" rides the GPU, "cpu" otherwise), so coerce it.
    """
    params = dict(xgb_params)
    params["device"] = str(device)
    return params


def classical_regressors(cfg, device: str) -> dict:
    """Name -> factory for the classical regression baselines (fresh each call)."""
    models = {
        "RandomForest": lambda: RandomForestRegressor(**cfg.rf_params),
        "Ridge": lambda: Ridge(**cfg.ridge_params),
    }
    if HAS_XGB:
        models["XGBoost"] = lambda: XGBRegressor(**_xgb_params(cfg.xgb_params, device))
    return models


def classical_classifiers(cfg, device: str) -> dict:
    """Name -> factory for the classical Direction classifiers (fresh each call)."""
    models = {
        "RandomForest": lambda: RandomForestClassifier(**cfg.rf_params),
        "LogisticRegression": lambda: LogisticRegression(**cfg.logreg_params),
    }
    if HAS_XGB:
        models["XGBoost"] = lambda: XGBClassifier(**_xgb_params(cfg.xgb_params, device))
    return models
