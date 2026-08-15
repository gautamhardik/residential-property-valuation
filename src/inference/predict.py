"""Inference pipeline — reuses the persisted training artifacts, no retraining.

predict_single(features: dict) accepts the raw King-County fields and returns a
predicted price plus the top global factors of the selected model.
"""
import numpy as np
import pandas as pd

from src.inference.artifacts import load_tabular_artifacts
from src.features.build_features import _engineer

REQUIRED_FIELDS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    "lat", "long", "sqft_living15", "sqft_lot15",
]
OPTIONAL_FIELDS = {"sale_year": 2015, "sale_quarter": 3}
# Defaults for amenity/quality fields the user rarely fills in.
DEFAULT_VALUES = {
    "waterfront": 0, "view": 0, "condition": 3, "grade": 7,
    "sqft_basement": 0, "yr_renovated": 0,
}


def to_row(features: dict) -> pd.DataFrame:
    row = {k: features.get(k, d) for k, d in OPTIONAL_FIELDS.items()}
    missing = []
    for f in REQUIRED_FIELDS:
        if f not in features:
            if f in DEFAULT_VALUES:
                row[f] = DEFAULT_VALUES[f]
            else:
                missing.append(f)
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for f in REQUIRED_FIELDS:
        row[f] = features.get(f, row.get(f, DEFAULT_VALUES.get(f)))
    return pd.DataFrame([row])


def predict_single(features: dict):
    pipeline, model = load_tabular_artifacts()
    cols = pipeline["feature_cols"]
    zip_enc, global_mean = pipeline["zip_enc"], pipeline["global_mean"]

    df = to_row(features)
    X = _engineer(df)
    X["zip_target"] = X["zipcode"].map(zip_enc).fillna(global_mean)

    price = float(model.predict(X[cols])[0])
    names = np.array(cols)
    importances = model.feature_importances_
    top = [(str(names[i]), float(importances[i])) for i in np.argsort(importances)[::-1][:8]]
    model_type = pipeline.get("model_type", "xgboost")
    model_label = {"xgboost": "XGBoost (tuned)", "catboost": "CatBoost (tuned)"}.get(model_type, model_type)
    importance_type = {"xgboost": "xgboost_gain", "catboost": "catboost_prediction_values_change"}.get(model_type, "model_importance")
    return {
        "predicted_price": round(price, 2),
        "model": f"{model_label} on engineered tabular features",
        "importance_type": importance_type,
        "top_factors_gain": [{"feature": f, "importance": w} for f, w in top],
    }