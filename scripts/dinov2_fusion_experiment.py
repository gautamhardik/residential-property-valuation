"""DINOv2 + tabular fusion experiment on the canonical split.

Runs on image-covered properties only, using:
  - E4_dino_tabular_only: champion tabular model on covered subset
  - E6_dino_fusion: champion model on [tabular || dinov2_vits14 embedding]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.models.train import evaluate, make_champion_estimator
from src.satellite.embeddings import load_cache


def build_tabular(df_tr, df_va):
    X_tr = _engineer(df_tr)
    X_va = _engineer(df_va)
    global_mean, enc = fit_target_encoder(X_tr)
    for fr in (X_tr, X_va):
        fr["zip_target"] = fr["zipcode"].map(enc).fillna(global_mean)
    cols = [c for c in FEATURE_COLS if c != "price"]
    return X_tr[cols].values, X_va[cols].values


def run():
    cached = load_cache("dinov2_vits14")
    if cached is None:
        raise FileNotFoundError(
            "Missing DINOv2 cache. Run scripts/extract_embeddings.py --encoder dinov2_vits14 first."
        )
    emb_ids, emb = cached
    emb_by_id = dict(zip(emb_ids, emb))
    covered_ids = set(emb_ids)

    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train = df[df["id"].isin(tr_ids)]
    val = df[df["id"].isin(va_ids)]

    img_train = train[train["id"].isin(covered_ids)]
    img_val = val[val["id"].isin(covered_ids)]
    y_tr = img_train["price"].values
    y_va = img_val["price"].values

    X_tr_tab, X_va_tab = build_tabular(img_train, img_val)
    E_tr = np.array([emb_by_id[pid] for pid in img_train["id"]])
    E_va = np.array([emb_by_id[pid] for pid in img_val["id"]])

    rows = []
    tab_model = make_champion_estimator().fit(X_tr_tab, y_tr)
    tab_met = evaluate(y_va, tab_model.predict(X_va_tab))
    rows.append({"experiment": "E4_dino_tabular_only", **tab_met})

    mm_model = make_champion_estimator().fit(np.hstack([X_tr_tab, E_tr]), y_tr)
    mm_met = evaluate(y_va, mm_model.predict(np.hstack([X_va_tab, E_va])))
    rows.append({"experiment": "E6_dino_fusion", **mm_met})

    base_rmse = tab_met["rmse"]
    for row in rows:
        row["rmse_improvement_pct"] = (base_rmse - row["rmse"]) / base_rmse * 100.0
        row["r2_change"] = row["r2"] - tab_met["r2"]

    out = {
        "encoder": "dinov2_vits14",
        "split": {"covered_train": int(len(img_train)), "covered_val": int(len(img_val))},
        "results": rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "results_dinov2_fusion.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "results_dinov2_fusion.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved", REPORTS_DIR / "results_dinov2_fusion.json")


if __name__ == "__main__":
    run()
