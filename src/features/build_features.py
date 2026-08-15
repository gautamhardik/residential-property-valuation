"""Tabular feature engineering.

The raw dataset provides 20 columns; the original project used only 5. Here we
retain the complete competition-provided set and add justified derived features.

Retained raw features (rationale):
- bedrooms / bathrooms / sqft_living: structural size signals.
- sqft_lot / sqft_living15 / sqft_lot15: lot size + 15-neighbour averages (neighbourhood wealth proxy).
- floors / sqft_above / sqft_basement: vertical layout and finished-basement signal.
- waterfront / view / condition / grade: quality / amenity markers.
- yr_built / yr_renovated: vintage and renovation.
- zipcode / lat / long: location.
- date -> sale_year, sale_quarter: temporal market drift.

Engineered features (rationale):
- age / renovated / renovation_age: age and recency of renovation shift value.
- total_sqft / basement_frac / above_frac / living_per_bedroom / lot_living_ratio: density and layout ratios.
- has_basement / has_view: binarised amenity flags.
- lat_long_interaction: combined coordinate proxy for spatial gradients.
- dist_to_center_km: straight-line distance to Seattle CBD (location quality).
- zip_freq / zip_target: frequency and target-encoded neighbourhood id (target encoding fitted ONLY on the training split to avoid leakage).
"""
import numpy as np
import pandas as pd

from src.config import SEATTLE_CENTER
from src.utils import haversine_km

BASE_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    "lat", "long", "sqft_living15", "sqft_lot15", "sale_year", "sale_quarter",
]

ENGINEERED_FEATURES = [
    "age", "renovated", "renovation_age",
    "total_sqft", "basement_frac", "above_frac",
    "living_per_bedroom", "lot_living_ratio", "has_basement",
    "lat_long_interaction", "dist_to_center_km",
    "zip_freq", "zip_target",
]

FEATURE_COLS = BASE_FEATURES + ENGINEERED_FEATURES
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in ("zipcode", "zip_freq", "zip_target", "sale_year", "sale_quarter")]


def _engineer(df: pd.DataFrame, reference_year: int = None) -> pd.DataFrame:
    df = df.copy()
    ref = reference_year or int(df["sale_year"].max())
    df["age"] = (ref - df["yr_built"]).clip(lower=0)
    df["renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["renovation_age"] = np.where(
        df["renovated"] == 1, (ref - df["yr_renovated"]).clip(lower=0), 0
    )
    df["total_sqft"] = df["sqft_living"] + df["sqft_basement"]
    df["basement_frac"] = df["sqft_basement"] / (df["sqft_living"] + 1e-6)
    df["above_frac"] = df["sqft_above"] / (df["sqft_living"] + 1e-6)
    df["living_per_bedroom"] = df["sqft_living"] / (df["bedrooms"] + 1.0)
    df["lot_living_ratio"] = df["sqft_lot"] / (df["sqft_living"] + 1e-6)
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    df["lat_long_interaction"] = df["lat"] * df["long"]
    df["dist_to_center_km"] = df.apply(
        lambda r: haversine_km(r["lat"], r["long"], SEATTLE_CENTER[0], SEATTLE_CENTER[1]), axis=1
    )
    df["zip_freq"] = df["zipcode"].map(df["zipcode"].value_counts(normalize=True))
    return df


def fit_target_encoder(df: pd.DataFrame, col: str = "zipcode", target: str = "price", smoothing: float = 20.0):
    """Target-encode `col` using global + group means (James–Stein style shrinkage)."""
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(["mean", "count"])
    enc = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    return global_mean, enc


class ZipTargetEncoder:
    """Fold-safe target encoder for `zipcode`.

    Usable standalone (fit/transform) or inside a scikit-learn Pipeline so that
    hyperparameter search encodes each CV fold with training-fold targets only.
    """

    def __init__(self, col: str = "zipcode", target: str = "price", smoothing: float = 20.0):
        self.col = col
        self.target = target
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y=None):
        labels = X[self.col] if self.col in X.columns else X.index.to_series()
        y_vals = pd.Series(y if y is not None else X[self.target], index=labels.index)
        self.global_mean_ = float(y_vals.mean())
        agg = pd.DataFrame({"price": y_vals.values}, index=labels.values).groupby(level=0)["price"].agg(["mean", "count"])
        self.enc_ = (agg["count"] * agg["mean"] + self.smoothing * self.global_mean_) / (
            agg["count"] + self.smoothing
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["zip_target"] = out[self.col].map(self.enc_).fillna(self.global_mean_)
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def build_features(train: pd.DataFrame, val: pd.DataFrame = None, test: pd.DataFrame = None):
    """Return engineered X/y frames. Target encoding is fitted on `train` only."""
    X_train = _engineer(train)
    if val is not None:
        X_val = _engineer(val)
    if test is not None:
        X_test = _engineer(test)

    global_mean, zip_enc = fit_target_encoder(X_train)
    for frame in (X_train, X_val, X_test):
        if frame is None:
            continue
        frame["zip_target"] = frame["zipcode"].map(zip_enc).fillna(global_mean)

    y_train = X_train["price"].values
    keep = [c for c in FEATURE_COLS if c != "price"]

    def _final(frame):
        return frame[keep], frame["price"].values

    X_tr, y_tr = _final(X_train)
    X_va, y_va = (_final(X_val) if val is not None else (None, None))
    X_te, y_te = (_final(X_test) if test is not None else (None, None))
    return X_tr, y_tr, X_va, y_va, X_te, y_te