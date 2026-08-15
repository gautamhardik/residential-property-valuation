"""Unified training & evaluation helpers.

Every experiment shares the same split and the same metric protocol so the
baseline and multimodal models are directly comparable.
"""
import json

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.config import REPORTS_DIR, RANDOM_STATE


def load_champion_config() -> tuple:
    """Read the tuned-champion configuration (model family + params) from disk.

    The tuning artifact (`tuned_best.json`) is the single source of truth. All
    downstream phases (final model, error analysis, SHAP, split & temporal
    validation, multimodal) consume this so they can never drift from the tuned
    configuration.
    """
    best = json.loads((REPORTS_DIR / "tuned_best.json").read_text(encoding="utf-8"))
    model_type = best.get("model_type", "xgboost")
    params = {k: v for k, v in best["params"].items()}
    return model_type, params


def load_tuned_params() -> dict:
    """Deprecated-ish alias returning just the tuned parameter dict (XGBoost-compatible)."""
    _, params = load_champion_config()
    params = dict(params)
    params.update(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    return params


def make_champion_estimator():
    """Instantiate the tuned champion model from `tuned_best.json`.

    Supports both gradient-boosting families currently in the model space so that
    whichever tuned best is used consistently across every experiment.
    """
    model_type, params = load_champion_config()
    params = dict(params)
    if model_type == "catboost":
        from catboost import CatBoostRegressor
        params.pop("n_jobs", None)
        params.pop("verbosity", None)
        params.update(random_seed=RANDOM_STATE, verbose=0)
        return CatBoostRegressor(**params)
    params.update(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    from xgboost import XGBRegressor
    return XGBRegressor(**params)


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def improvement(base_metrics: dict, new_metrics: dict) -> dict:
    """Percentage RMSE reduction and R2 change relative to the baseline."""
    rmse_improvement = (base_metrics["rmse"] - new_metrics["rmse"]) / base_metrics["rmse"] * 100.0
    r2_change = new_metrics["r2"] - base_metrics["r2"]
    return {
        "rmse_improvement_pct": rmse_improvement,
        "r2_change": r2_change,
    }


def fit_predict(model, X_train, y_train, X_val):
    model.fit(X_train, y_train)
    return model.predict(X_val), model