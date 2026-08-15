"""Phase 9 — Error analysis of the best tuned tabular model.

Trains the optimised XGBoost on the training fold and analyses residuals on the
never-touched validation fold across price bands, size, waterfront, geography,
and image-availability segments.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR, FIGURES_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.satellite.align import is_valid_image
from src.models.train import evaluate, make_champion_estimator


def segment_summary(df, y_true, y_pred, group_col, metric="mae"):
    out = df.assign(y_true=y_true, y_pred=y_pred, abs_err=np.abs(y_pred - y_true),
                    rel_err=np.abs(y_pred - y_true) / np.maximum(y_true, 1.0),
                    signed=y_pred - y_true)
    g = out.groupby(group_col).agg(
        n=("y_true", "size"),
        rmse=(group_col, lambda s: np.sqrt(np.mean((out.loc[s.index, "signed"]) ** 2))),
        mae=("abs_err", "mean"),
        median_rel_err=("rel_err", "median"),
        mean_bias=("signed", "mean"),
        mean_price=("y_true", "mean"),
    )
    g["median_rel_err"] = g["median_rel_err"].round(3)
    g[["rmse", "mae", "mean_bias", "mean_price"]] = g[[
        "rmse", "mae", "mean_bias", "mean_price"]].round(0)
    return g


def run():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train, val = df[df["id"].isin(tr_ids)], df[df["id"].isin(va_ids)]

    X_tr = _engineer(train)
    X_va = _engineer(val)
    gmean, enc = fit_target_encoder(X_tr)
    for fr in (X_tr, X_va):
        fr["zip_target"] = fr["zipcode"].map(enc).fillna(gmean)
    cols = [c for c in FEATURE_COLS if c != "price"]
    X_tr, X_va, y_tr, y_va = X_tr[cols], X_va[cols], train["price"].values, val["price"].values

    model = make_champion_estimator().fit(X_tr, y_tr)
    y_pred = model.predict(X_va)

    summary = {"overall": evaluate(y_va, y_pred)}
    segments = {}
    for col, bins in (("price_band", 4), ("size_band", 4), ("condition", None),
                      ("waterfront", None), ("location_band", 4)):
        if col == "price_band":
            val["seg"] = pd.qcut(val["price"], q=4, labels=["Low", "Mid", "High", "Luxury"])
        elif col == "size_band":
            val["seg"] = pd.qcut(val["sqft_living"], q=4, labels=["Small", "Medium", "Large", "XL"])
        elif col == "waterfront":
            val["seg"] = val["waterfront"].map({1: "Waterfront", 0: "Non-waterfront"})
        elif col == "location_band":
            lat_bins = pd.qcut(val["lat"], q=4, labels=["S", "C-S", "C-N", "N"])
            lon_bins = pd.qcut(val["long"], q=4, labels=["W", "C-W", "C-E", "E"])
            val["seg"] = lat_bins.astype(str) + "-" + lon_bins.astype(str)
        else:
            val["seg"] = val[col]
        res = segment_summary(val, y_va, y_pred, "seg")
        segments[col] = res.to_dict(orient="index")

    # image availability
    val["has_img"] = val["id"].map(is_valid_image)
    res_img = segment_summary(val, y_va, y_pred, "has_img")
    segments["has_image"] = res_img.to_dict(orient="index")

    # worst cases
    err_df = val.assign(abs_err=np.abs(y_pred - y_va), rel_err=np.abs(y_pred - y_va) / val["price"],
                        signed_err=y_pred - y_va, predicted=y_pred)
    worst_abs = err_df.nlargest(10, "abs_err")[["id", "price", "predicted", "abs_err", "rel_err"]]
    worst_rel = err_df.nlargest(10, "rel_err")[["id", "price", "predicted", "abs_err", "rel_err"]]

    summary["worst_absolute"] = worst_abs.round(0).to_dict(orient="records")
    summary["worst_relative"] = worst_rel.round(0).to_dict(orient="records")
    summary["segments"] = segments
    summary["n_val"] = len(val)

    (REPORTS_DIR / "error_analysis.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    err_df[["id", "price", "predicted"]].to_csv(REPORTS_DIR / "val_predictions_best_tabular.csv", index=False)

    print("Overall:", {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["overall"].items()})
    print("\nBy price band (MAE, median rel err, bias):")
    print(segments["price_band"])
    print("\nBy waterfront:")
    print(segments["waterfront"])
    print("\nBy image availability:")
    print(segments["has_image"])


if __name__ == "__main__":
    run()