"""Phase 8 — Fair multimodal experiment.

Subset = properties with valid imagery (2,189 rows) and uses the SAME canonical
train/val ids. Experiments:

  E4  tabular-only on the image subset
  E4B image-only (embeddings only)
  E5  tabular + resnet18 embeddings
  E5B tabular + resnet50 embeddings

Every model uses the identical split and identical metric protocol. Improvements
are measured against E4 (the fair tabular control on the same properties).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.satellite.embeddings import load_cache
from src.models.train import evaluate, improvement, make_champion_estimator

from sklearn.ensemble import RandomForestRegressor


def build_tabular(df_tr, df_va):
    X_tr = _engineer(df_tr)
    X_va = _engineer(df_va)
    global_mean, enc = fit_target_encoder(X_tr)
    for fr in (X_tr, X_va):
        fr["zip_target"] = fr["zipcode"].map(enc).fillna(global_mean)
    cols = [c for c in FEATURE_COLS if c != "price"]
    return X_tr[cols].values, X_va[cols].values


# The tabular control (E4) uses the exact final tuned champion configuration, so the
# multimodal comparison stacks imagery against the strongest available tabular model.
RF_PARAMS = dict(
    n_estimators=250, max_depth=30, max_features=0.5, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1,
)


def run():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train, val = df[df["id"].isin(tr_ids)], df[df["id"].isin(va_ids)]

    img_train = train[train["id"].isin(set(load_cache("resnet18")[0]))]
    img_val = val[val["id"].isin(set(load_cache("resnet18")[0]))]
    y_tr, y_va = img_train["price"].values, img_val["price"].values
    print(f"Image subset: train={len(img_train)} val={len(img_val)}")

    X_tr_tab, X_va_tab = build_tabular(img_train, img_val)

    ids18, emb18 = load_cache("resnet18")
    ids50, emb50 = load_cache("resnet50")
    emb_by_id_18 = dict(zip(ids18, emb18))
    emb_by_id_50 = dict(zip(ids50, emb50))

    E_tr_18 = np.array([emb_by_id_18[pid] for pid in img_train["id"]])
    E_va_18 = np.array([emb_by_id_18[pid] for pid in img_val["id"]])
    E_tr_50 = np.array([emb_by_id_50[pid] for pid in img_train["id"]])
    E_va_50 = np.array([emb_by_id_50[pid] for pid in img_val["id"]])

    experiment_inputs = {
        "E4_tabular_only": (X_tr_tab, X_va_tab),
        "E4B_image_only_rn18": (E_tr_18, E_va_18),
        "E5_mm_rn18": (np.hstack([X_tr_tab, E_tr_18]), np.hstack([X_va_tab, E_va_18])),
        "E5B_mm_rn50": (np.hstack([X_tr_tab, E_tr_50]), np.hstack([X_va_tab, E_va_50])),
    }

    rows = []
    base = {"rmse": None}
    for name, (Xte, Xve) in experiment_inputs.items():
        for fam, factory in (("Champion", lambda: make_champion_estimator()),
                             ("RF", lambda: RandomForestRegressor(**RF_PARAMS))):
            m = factory()
            m.fit(Xte, y_tr)
            met = evaluate(y_va, m.predict(Xve))
            rows.append({"experiment": name, "family": fam, "rmse": met["rmse"],
                         "mse": met["mse"], "mae": met["mae"], "r2": met["r2"]})
            if name == "E4_tabular_only" and fam == "Champion":
                base = met

    res = pd.DataFrame(rows)
    # improvements vs E4 Champion (tabular control)
    res["rmse_improvement_pct"] = (base["rmse"] - res["rmse"]) / base["rmse"] * 100.0
    res["r2_change"] = res["r2"] - base["r2"]

    res.to_csv(REPORTS_DIR / "results_multimodal.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    run()