"""Broader model space + stacking experiment.

The champion is a tuned XGBoost. This phase checks whether expanding the model
space helps: gradient boosting (XGBoost, LightGBM), categorical boosting
(CatBoost), and a simple stacked ensemble of all three.

Protocol (all on the SAME canonical random split used by the champion, for direct
comparison):
- Same engineered features; fold-safe zip_target fitted on the training fold only.
- Base models are compared individually on the validation fold.
- A stacked blend is built with out-of-fold (3-fold, within the training fold)
  predictions feeding a Ridge stacker - so the stacker weights are learned
  without touching the validation fold. This is a leak-free stack.
- Metrics reported in original price units on the never-touched validation fold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, ZipTargetEncoder
from src.models.train import evaluate, load_tuned_params

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

COLS = [c for c in FEATURE_COLS if c != "price"]


def make_models():
    """Base learners in the ensemble, each with its own tuned/fixed config."""
    xgb_params = load_tuned_params()  # the champion (tuned XGBoost) config
    cat_params = dict(
        iterations=366, learning_rate=0.1185, depth=4, l2_leaf_reg=1.4047,
        subsample=0.9209, random_strength=0.2818, bagging_temperature=1.0794,
    )
    return {
        "XGBoost_tuned": lambda: XGBRegressor(**xgb_params),
        "CatBoost_tuned": lambda: CatBoostRegressor(**cat_params, verbose=0, random_seed=RANDOM_STATE),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=500, learning_rate=0.08, num_leaves=40, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1),
    }


def fit_estimators(factories, X, y):
    return {name: fac().fit(X, y) for name, fac in factories.items()}


def fit_transform_zip(df_tr, df_va):
    X_tr, X_va = _engineer(df_tr), _engineer(df_va)
    enc = ZipTargetEncoder().fit(X_tr, df_tr["price"].values)
    X_tr["zip_target"] = enc.transform(X_tr)["zip_target"]
    X_va["zip_target"] = enc.transform(X_va)["zip_target"]
    return X_tr[COLS], X_va[COLS], df_tr["price"].values, df_va["price"].values


def run():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train, val = df[df["id"].isin(tr_ids)], df[df["id"].isin(va_ids)]
    X_tr, X_va, y_tr, y_va = fit_transform_zip(train, val)

    models = make_models()
    names = list(models.keys())

    # ---- individual models on the full training fold -> val ----
    rows = []
    full_preds = {}
    for name in names:
        m = models[name]().fit(X_tr, y_tr)
        p = m.predict(X_va)
        full_preds[name] = p
        rows.append({"model": name, **evaluate(y_va, p)})

    # ---- leak-free stacked blend via out-of-fold predictions ----
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    oof = {name: np.zeros(len(X_tr)) for name in names}
    for tr_idx, va_idx in cv.split(X_tr):
        for name in names:
            m = models[name]().fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
            oof[name][va_idx] = m.predict(X_tr.iloc[va_idx])

    stack_X_oof = np.column_stack([oof[n] for n in names])
    stack_X_val = np.column_stack([full_preds[n] for n in names])
    stacker = Ridge(alpha=10.0).fit(stack_X_oof, y_tr)
    stack_val = stacker.predict(stack_X_val)
    rows.append({"model": "StackedBlend_Ridge", "blend_weights": dict(zip(names, stacker.coef_)),
                 **evaluate(y_va, stack_val)})

    df_res = pd.DataFrame(rows)
    out = REPORTS_DIR / "results_ensemble.csv"
    df_res.to_csv(out, index=False)

    for _, r in df_res.iterrows():
        print(f"{r['model']:<18} RMSE={r['rmse']:>10,.0f}  MAE={r['mae']:>9,.0f}  R2={r['r2']:.4f}")
    print("\nStack blend weights:", dict(zip(names, stacker.coef_)))


if __name__ == "__main__":
    run()
