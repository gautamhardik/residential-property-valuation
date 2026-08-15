"""Local (TreeSHAP) explanation and empirical error band for a single prediction.

These are ADDITIVE to the deployed prediction path:
- The prediction value itself is NOT changed (parity with predict_single is preserved).
- local_summary computes TreeSHAP contributions for the *actual* request features.
- error_band_for reads the real 20% holdout residuals and reports a value-segment
  typical error. It is an aggregate/empirical guide, never a per-property
  confidence interval or prediction interval.
"""
import numpy as np
import pandas as pd

from src.features.build_features import _engineer
from src.inference.artifacts import load_tabular_artifacts
from src.inference.predict import to_row
from src.config import PROJECT_ROOT

HOLDOUT_RESIDUALS = PROJECT_ROOT / "reports" / "val_predictions_best_tabular.csv"

# Engineered + raw feature names -> human-readable labels (single source mirrored in the UI).
FEATURE_LABELS = {
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "sqft_living": "Living area",
    "sqft_lot": "Lot size",
    "floors": "Floors",
    "waterfront": "Waterfront",
    "view": "View quality",
    "condition": "Condition",
    "grade": "Grade",
    "sqft_above": "Above-ground area",
    "sqft_basement": "Basement area",
    "yr_built": "Year built",
    "yr_renovated": "Year renovated",
    "zipcode": "ZIP code",
    "lat": "Latitude",
    "long": "Longitude",
    "sqft_living15": "Avg. nearby living area",
    "sqft_lot15": "Avg. nearby lot size",
    "sale_year": "Sale year",
    "sale_quarter": "Sale quarter",
    "age": "Property age",
    "renovated": "Recently renovated",
    "renovation_age": "Years since renovation",
    "total_sqft": "Total built area",
    "basement_frac": "Basement share",
    "above_frac": "Above-ground share",
    "living_per_bedroom": "Living area per bedroom",
    "lot_living_ratio": "Lot-to-living ratio",
    "has_basement": "Has basement",
    "lat_long_interaction": "Location interaction",
    "dist_to_center_km": "Distance to city center",
    "zip_freq": "Neighborhood frequency",
    "zip_target": "Neighborhood location signal",
}

_explainer = None
_bands = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        from shap import TreeExplainer

        _, model = load_tabular_artifacts()
        _explainer = TreeExplainer(model)
    return _explainer


def _prepared(features: dict):
    pipeline, model = load_tabular_artifacts()
    cols = list(pipeline["feature_cols"])
    zip_enc, global_mean = pipeline["zip_enc"], pipeline["global_mean"]
    df = to_row(features)
    X = _engineer(df)
    X["zip_target"] = X["zipcode"].map(zip_enc).fillna(global_mean)
    return X, cols, model


def _expected_value(explainer) -> float:
    ev = explainer.expected_value
    arr = np.asarray(ev).reshape(-1)
    return float(arr[0])


def local_summary(features: dict, top_n: int = 5):
    """Return the top positive/negative TreeSHAP contributions for this request."""
    try:
        X, cols, _model = _prepared(features)
        explainer = _get_explainer()
        sv = np.atleast_2d(np.asarray(explainer.shap_values(X[cols]), dtype=np.float64))
        if sv.shape[1] != len(cols):
            sv = sv.reshape(1, len(cols))
        row = sv[0]
        expected = _expected_value(explainer)

        def item(i, direction):
            return {
                "feature": cols[i],
                "label": FEATURE_LABELS.get(cols[i], cols[i].replace("_", " ").title()),
                "contribution": round(float(row[i]), 2),
                "direction": direction,
            }

        order_pos = sorted(range(len(cols)), key=lambda i: row[i], reverse=True)
        order_neg = sorted(range(len(cols)), key=lambda i: row[i])
        positives = [item(i, "up") for i in order_pos if row[i] > 0][:top_n]
        negatives = [item(i, "down") for i in order_neg if row[i] < 0][:top_n]

        return {
            "expected_value": round(float(expected), 2),
            "total_contribution": round(float(np.sum(row)), 2),
            "predicted_price": round(float(expected) + float(np.sum(row)), 2),
            "top_positive": positives,
            "top_negative": negatives,
            "method": "TreeSHAP on the deployed model, evaluated at request time",
        }
    except Exception:  # noqa: BLE001 - explanation is best-effort, never blocks prediction
        return None


def _load_bands():
    global _bands
    if _bands is None:
        vp = pd.read_csv(HOLDOUT_RESIDUALS)
        vp["abs_err"] = (vp["predicted"] - vp["price"]).abs()
        edges = pd.Series(vp["predicted"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).unique()).sort_values().tolist()
        bins = pd.cut(vp["predicted"], edges, include_lowest=True)
        stats = vp.groupby(bins, observed=True)["abs_err"].agg(["count", "median"])
        _bands = {"edges": edges, "stats": stats}
    return _bands


def error_band_for(price: float):
    """Typical (median) absolute error for the value segment containing `price`."""
    try:
        bands = _load_bands()
        edges, stats = bands["edges"], bands["stats"]
        interval = pd.cut([price], edges, include_lowest=True)[0]
        if pd.isna(interval):
            return None
        row = stats.loc[interval]
        lo, hi = int(interval.left), int(interval.right)
        return {
            "segment_label": f"${lo / 1000:.0f}K–${hi / 1000:.0f}K",
            "typical_error": round(float(row["median"]), 2),
            "n": int(row["count"]),
            "method": "median absolute prediction error over the 20% holdout, bucketed by predicted value",
            "note": "Empirical segment-level error; it is not a per-property confidence interval.",
        }
    except Exception:  # noqa: BLE001
        return None