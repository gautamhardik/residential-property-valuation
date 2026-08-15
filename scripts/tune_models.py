"""Phase 4 — Hyperparameter tuning (RandomizedSearchCV, 30 iters, 3-fold CV).

Methodology notes:
- Search space is explored with `RandomizedSearchCV` (30 random iterations) inside
  the training fold using 3-fold cross-validation, scoring neg RMSE.
- The canonical validation split (val, 20% holdout) is NEVER touched during search.
- Target encoding (`zip_target`) is applied via a `Pipeline` so that, inside each CV
  fold, the encoder is fitted on the *training part of that fold only* (fold-safe).
  This removes target-encoding leakage from model selection.
- Best model is re-fitted on the full training fold and scored once on val.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, ZipTargetEncoder
from src.models.train import evaluate

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

N_ITER = 30
N_FOLDS = 3


def tuned_pipeline(estimator):
    """Engineered features -> fold-safe zip target encoding -> estimator."""
    return Pipeline([("zip_enc", ZipTargetEncoder()), ("model", estimator)])


def tune(estimator, params, X, y):
    pipe = tuned_pipeline(estimator)
    grid = {f"model__{k}": v for k, v in params.items()}
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe, grid, n_iter=N_ITER, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=RANDOM_STATE,
    )
    search.fit(X, y)
    return search


def build_cols(df_tr, df_va):
    X_tr = _engineer(df_tr)
    X_va = _engineer(df_va)
    # `zip_target` is added inside the pipeline (fold-safe); the model input here
    # contains the base + engineered columns only.
    cols = [c for c in FEATURE_COLS if c not in ("price", "zip_target")]
    return X_tr[cols], X_va[cols], cols


def temporal_r2(df, model_name, params):
    """Out-of-time R2 used to break near-ties between model families."""
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"].str.slice(0, 8), format="%Y%m%d", errors="coerce")
    tr = df[df["date_dt"] < "2015-03-01"]
    va = df[df["date_dt"] >= "2015-03-01"]
    X_tr, X_va = _engineer(tr), _engineer(va)
    enc = ZipTargetEncoder().fit(X_tr, tr["price"].values)
    X_tr["zip_target"] = enc.transform(X_tr)["zip_target"]
    X_va["zip_target"] = enc.transform(X_va)["zip_target"]
    cols = [c for c in FEATURE_COLS if c != "price"]
    p = dict(params)
    if model_name == "CatBoost_tuned":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(**p, verbose=0, random_seed=RANDOM_STATE)
    elif model_name == "RandomForest_tuned":
        m = RandomForestRegressor(**p, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        m = XGBRegressor(**p, n_jobs=-1, verbosity=0, random_state=RANDOM_STATE)
    m.fit(X_tr[cols], tr["price"].values)
    return evaluate(va["price"].values, m.predict(X_va[cols]))["r2"]


def run():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train, val = df[df["id"].isin(tr_ids)], df[df["id"].isin(va_ids)]
    y_tr, y_va = train["price"].values, val["price"].values

    X_tr, X_va, cols = build_cols(train, val)

    results = []

    # ---- Random Forest ----
    rf_search = tune(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": randint(150, 500),
            "max_depth": randint(15, 45),
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 8),
            "max_features": ["sqrt", "log2", None, 0.5, 0.7, 0.9],
        },
        X_tr, y_tr,
    )
    rf_best = rf_search.best_estimator_.fit(X_tr, y_tr)
    rf_params = {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()}
    rf_met = evaluate(y_va, rf_best.predict(X_va))
    results.append({"model": "RandomForest_tuned", "best_params": rf_params, **rf_met})

    # ---- XGBoost ----
    xgb_search = tune(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        {
            "n_estimators": randint(150, 600),
            "learning_rate": uniform(0.01, 0.2),
            "max_depth": randint(3, 9),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": randint(1, 8),
            "reg_lambda": uniform(0, 5),
        },
        X_tr, y_tr,
    )
    xgb_best = xgb_search.best_estimator_.fit(X_tr, y_tr)
    xgb_params = {k.replace("model__", ""): v for k, v in xgb_search.best_params_.items()}
    xgb_met = evaluate(y_va, xgb_best.predict(X_va))
    results.append({"model": "XGBoost_tuned", "best_params": xgb_params, **xgb_met})

    # ---- CatBoost ----
    cat_search = tune(
        CatBoostRegressor(verbose=0, random_seed=RANDOM_STATE),
        {
            "iterations": randint(150, 600),
            "learning_rate": uniform(0.01, 0.2),
            "depth": randint(3, 9),
            "l2_leaf_reg": uniform(0, 5),
            "subsample": uniform(0.6, 0.4),
            "random_strength": uniform(0, 2),
            "bagging_temperature": uniform(0, 2),
        },
        X_tr, y_tr,
    )
    cat_best = cat_search.best_estimator_.fit(X_tr, y_tr)
    cat_params = {k.replace("model__", ""): v for k, v in cat_search.best_params_.items()}
    cat_met = evaluate(y_va, cat_best.predict(X_va))
    results.append({"model": "CatBoost_tuned", "best_params": cat_params, **cat_met})

    # ---- persist ----
    df_res = pd.DataFrame(results)
    df_res.to_csv(REPORTS_DIR / "results_tuned.csv", index=False)

    # ---- champion selection (generalization-aware) ----
    # A model that wins on the in-distribution random holdout by a tiny margin but
    # generalises poorly out-of-time is not the best champion. If the best models
    # are within 1% random-holdout RMSE, break the tie on out-of-time R2.
    ranked = sorted(results, key=lambda r: r["rmse"])
    best_row = ranked[0]
    tie = [r for r in results if (r["rmse"] - best_row["rmse"]) / best_row["rmse"] <= 0.01]
    selection = "best random-holdout RMSE (clear margin)"
    if len(tie) > 1:
        for r in tie:
            r["_temporal_r2"] = temporal_r2(df, r["model"], r["best_params"])
        best_row = max(tie, key=lambda r: r["_temporal_r2"])
        selection = "temporal-generalization tie-break within 1% random-holdout RMSE"

    best_params = {k: (float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v)
                   for k, v in best_row["best_params"].items()}
    model_type = {"CatBoost_tuned": "catboost", "XGBoost_tuned": "xgboost"}.get(best_row["model"], "xgboost")
    (REPORTS_DIR / "tuned_best.json").write_text(
        json.dumps({"model_type": model_type,
                    "method": f"RandomizedSearchCV, {N_ITER} iters, {N_FOLDS}-fold CV",
                    "selection": selection,
                    "params": best_params, "metrics": {k: round(v, 4) for k, v in best_row.items()
                                                       if k in ("rmse", "mse", "mae", "r2")}},
                   indent=2, default=str), encoding="utf-8")

    for r in results:
        print(f"\n{r['model']}: RMSE={r['rmse']:,.0f} R2={r['r2']:.4f}")
        print("  best_params:", r["best_params"])
    print("\nChampion:", best_row["model"], f"({selection})")
    return results


if __name__ == "__main__":
    run()