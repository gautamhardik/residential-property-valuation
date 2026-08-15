"""Temporal (out-of-time) validation.

The primary canonical split is random 80/20. Because the market drifts over time
(May 2014 - May 2015 here), a *random* holdout lets the model peek at nearby sales
in time. This phase evaluates the same tuned XGBoost + same engineered features on
a strict chronological split: train on sales on/before 2015-02-28, validate on
sales in 2015-03 .. 2015-05. No training example is interleaved in time with the
validation set, so it measures how well the model predicts into the future.

Controlled comparison: identical model config, identical feature set, identical
fold-safe zip_target treatment - only the way train/validation are separated changes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR
from src.data.load import load_clean_train
from src.features.build_features import FEATURE_COLS, _engineer, ZipTargetEncoder
from src.models.train import evaluate, make_champion_estimator
from src.utils import haversine_km

COLS = [c for c in FEATURE_COLS if c != "price"]
CUTOFF = "2015-03-01"


def load_tuned_params() -> dict:
    path = REPORTS_DIR / "tuned_best.json"
    best = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in best["params"].items()}


def fit_eval(df_tr, df_va, params):
    X_tr, X_va = _engineer(df_tr), _engineer(df_va)
    enc = ZipTargetEncoder().fit(X_tr, df_tr["price"].values)
    X_tr["zip_target"] = enc.transform(X_tr)["zip_target"]
    X_va["zip_target"] = enc.transform(X_va)["zip_target"]
    y_tr, y_va = df_tr["price"].values, df_va["price"].values
    model = make_champion_estimator().fit(X_tr[COLS], y_tr)
    return evaluate(y_va, model.predict(X_va[COLS])), len(df_tr), len(df_va)


def nearest_neighbor_km(train_df, val_df):
    tr = train_df[["lat", "long"]].to_numpy()
    va = val_df[["lat", "long"]].to_numpy()
    out = np.empty(len(va))
    for i in range(len(va)):
        out[i] = haversine_km(va[i, 0], va[i, 1], tr[:, 0], tr[:, 1]).min()
    return out


def run():
    df = load_clean_train()
    df["date_dt"] = pd.to_datetime(df["date"].str.slice(0, 8), format="%Y%m%d", errors="coerce")
    train = df[df["date_dt"] < CUTOFF]
    val = df[df["date_dt"] >= CUTOFF]
    params = load_tuned_params()

    met, n_tr, n_va = fit_eval(train, val, params)

    # geographic proximity of the temporal val set to the temporal train set
    nn_km = nearest_neighbor_km(train, val)
    prox = {
        "median_nn_km": float(np.median(nn_km)),
        "pct_val_within_1km_of_train": float((nn_km <= 1.0).mean() * 100),
    }

    # date range of each fold
    date_fmt = lambda s: (s.min().strftime("%Y-%m-%d"), s.max().strftime("%Y-%m-%d"))
    tr_lo, tr_hi = date_fmt(train["date_dt"])
    va_lo, va_hi = date_fmt(val["date_dt"])

    result = {
        "method": "Chronological (out-of-time) split; same tuned XGBoost + same engineered features; fold-safe zip_target on train only",
        "cutoff": CUTOFF,
        "train_period": {"start": tr_lo, "end": tr_hi, "rows": int(n_tr)},
        "val_period": {"start": va_lo, "end": va_hi, "rows": int(n_va)},
        "spatial_proximity": prox,
        "metrics": {k: round(float(v), 4) for k, v in met.items()},
        "vs_random_holdout_r2": 0.9205,
        "r2_delta_vs_random": round(met["r2"] - 0.9205, 4),
    }
    out = REPORTS_DIR / "temporal_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
