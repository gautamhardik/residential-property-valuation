"""Final-model artifacts (persisted once, reused by training, submission and inference).

The final model is the tuned XGBoost on engineered tabular features (selected in
Phase 11 because the multimodal experiments never beat it). This module stores:

  model.joblib          fitted XGBoost
  pipeline.joblib       dict with zip target-encoder, global mean, feature names

Artifacts are loaded once at application startup via `preload_tabular_artifacts()`
and cached in module-level globals so no disk I/O occurs on individual requests.
"""
import joblib

from src.config import APP_MODELS_DIR

PIPELINE_FILE = APP_MODELS_DIR / "tabular_pipeline.joblib"
MODEL_FILE = APP_MODELS_DIR / "tabular_model.joblib"

# Module-level cache — populated once at startup, never retrained.
_PIPELINE = None
_MODEL = None


def save_tabular_artifacts(model, zip_enc, global_mean, feature_cols, model_type="xgboost"):
    joblib.dump({
        "zip_enc": zip_enc,
        "global_mean": global_mean,
        "feature_cols": feature_cols,
        "model_type": model_type,
    }, PIPELINE_FILE)
    joblib.dump(model, MODEL_FILE)


def preload_tabular_artifacts() -> None:
    """Load XGBoost artifacts from disk into the module cache.

    Call this once at application startup.  Subsequent calls to
    `load_tabular_artifacts()` will return the cached objects without
    touching the filesystem.
    """
    global _PIPELINE, _MODEL
    if _PIPELINE is None or _MODEL is None:
        _PIPELINE = joblib.load(PIPELINE_FILE)
        _MODEL = joblib.load(MODEL_FILE)


def load_tabular_artifacts():
    """Return the cached (pipeline, model) tuple.

    Falls back to disk loading if the cache has not been populated yet
    (e.g. when called directly from the CLI or tests without a prior
    `preload_tabular_artifacts()` call).
    """
    global _PIPELINE, _MODEL
    if _PIPELINE is None or _MODEL is None:
        preload_tabular_artifacts()
    return _PIPELINE, _MODEL