"""Phase 8b — controlled dimensionality test.

If frozen ImageNet embeddings only add noise, PCA-compressing them (keeping the
top few signal directions) should either stay flat or recover marginally.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.satellite.embeddings import load_cache
from src.models.train import evaluate, make_champion_estimator


def main():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train, val = df[df["id"].isin(tr_ids)], df[df["id"].isin(va_ids)]
    ids18, emb18 = load_cache("resnet18")
    img_train = train[train["id"].isin(set(ids18))]
    img_val = val[val["id"].isin(set(ids18))]
    y_tr, y_va = img_train["price"].values, img_val["price"].values

    X_tr = _engineer(img_train)
    X_va = _engineer(img_val)
    gmean, enc = fit_target_encoder(X_tr)
    for fr in (X_tr, X_va):
        fr["zip_target"] = fr["zipcode"].map(enc).fillna(gmean)
    cols = [c for c in FEATURE_COLS if c != "price"]
    T_tr, T_va = X_tr[cols].values, X_va[cols].values

    emb_by = dict(zip(ids18, emb18))
    E_tr = np.array([emb_by[p] for p in img_train["id"]])
    E_va = np.array([emb_by[p] for p in img_val["id"]])

    rows = []
    # tabular control
    m = make_champion_estimator().fit(T_tr, y_tr)
    met = evaluate(y_va, m.predict(T_va))
    rows.append({"variant": "tabular_only", "rmse": met["rmse"], "r2": met["r2"]})

    for dims in (8, 16, 32, 64):
        pca = PCA(n_components=min(dims, E_tr.shape[1]), random_state=RANDOM_STATE).fit(
            StandardScaler().fit_transform(E_tr)
        )
        P_tr = pca.transform(StandardScaler().fit_transform(E_tr))
        P_va = pca.transform(StandardScaler().fit_transform(E_va))
        for label, (A_tr, A_va) in (
            ("img_only", (P_tr, P_va)),
            ("tabular_pca", (np.hstack([T_tr, P_tr]), np.hstack([T_va, P_va]))),
        ):
            mm = make_champion_estimator().fit(A_tr, y_tr)
            met2 = evaluate(y_va, mm.predict(A_va))
            rows.append({"variant": f"{label}_pca{dims}", "rmse": met2["rmse"], "r2": met2["r2"]})

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS_DIR / "results_multimodal_pca.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()