"""Phase 3 — Tabular baselines.

E1  original 5-feature baseline (as in the existing notebooks)
E2  full competition-provided tabular features
E3  engineered tabular features (age/ratios/spatial + zip target encoding)

Metrics computed on the same canonical validation split for every model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import BASE_FEATURES, ENGINEERED_FEATURES, FEATURE_COLS, _engineer, fit_target_encoder
from src.models.train import evaluate

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------
E1_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "lat", "long"]
E2_FEATURES = [c for c in BASE_FEATURES if c != "price"]
E3_FEATURES = [c for c in FEATURE_COLS if c != "price"]


def make_features(df_tr, df_va, cols, wants_zip_target=False):
    if wants_zip_target:
        X_tr = _engineer(df_tr)
        X_va = _engineer(df_va)
        global_mean, enc = fit_target_encoder(X_tr)
        for fr in (X_tr, X_va):
            fr["zip_target"] = fr["zipcode"].map(enc).fillna(global_mean)
    else:
        X_tr, X_va = df_tr.copy(), df_va.copy()
    return X_tr[cols], X_va[cols]


def run():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train, val = df[df["id"].isin(tr_ids)], df[df["id"].isin(va_ids)]
    y_tr, y_va = train["price"].values, val["price"].values

    model_factory = {
        "LinearRegression": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0,
        ),
    }

    experiment_defs = [
        ("E1_original_5feat", E1_FEATURES, False),
        ("E2_full_tabular", E2_FEATURES, False),
        ("E3_engineered_tabular", E3_FEATURES, True),
    ]

    rows = []
    for exp_name, cols, zip_enc in experiment_defs:
        X_tr, X_va = make_features(train, val, cols, zip_enc)
        # scale for linear models only (fit on train)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)
        for m_name, factory in model_factory.items():
            if "Linear" in m_name:
                m = factory().fit(X_tr_s, y_tr)
                yp = m.predict(X_va_s)
            else:
                m = factory().fit(X_tr, y_tr)
                yp = m.predict(X_va)
            met = evaluate(y_va, yp)
            rows.append({**{"experiment": exp_name, "model": m_name}, **met})

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "results_tabular.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    run()