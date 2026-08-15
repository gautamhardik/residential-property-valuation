"""Phase 2 — Split-strategy validation (controlled).

Primary protocol: deterministic random 80/20 split (canonical split shared by all
experiments). Because this is geospatial data, we also quantify how spatially
proximate the validation set is to training (potential optimism) and compare a
spatially-aware holdout as a secondary robustness experiment.

Controlled comparison:
- The SAME tuned XGBoost configuration is used for both legs.
- The SAME engineered feature set (incl. `zip_target`) is used for both legs.
- `zip_target` is always fitted on the corresponding training fold only.
So the only thing that changes between the two legs is the way train/validation
were separated, making the R2/RMSE gap attributable to geographic generalisation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, TEST_SIZE, REPORTS_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, ZipTargetEncoder
from src.models.train import evaluate, make_champion_estimator
from src.utils import haversine_km

from sklearn.cluster import KMeans

COLS = [c for c in FEATURE_COLS if c != "price"]


def load_tuned_params() -> dict:
    path = REPORTS_DIR / "tuned_best.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["params"]
    return {}


def spatial_split(ids, coords, test_ratio=TEST_SIZE, n_centers=8, random_state=RANDOM_STATE):
    """Tile the coordinate plane into n_centers^2 spatial cells and split by cell.

    Ensures no two properties sharing a cell end up on opposite sides of the split,
    reducing spatial leakage between train and validation. Cells are KMeans clusters;
    the split is therefore coarse-grained spatially (unlike a random split).
    """
    cells = KMeans(n_clusters=n_centers, n_init=10, random_state=random_state).fit(coords)
    cell_of = pd.Series(cells.labels_, index=ids)
    unique_cells = np.unique(cells.labels_)
    rng = np.random.default_rng(random_state)
    val_cells = set(rng.choice(unique_cells, size=int(len(unique_cells) * test_ratio), replace=False))
    val_ids = {i for i, c in cell_of.items() if c in val_cells}
    train_ids = set(ids) - val_ids
    return train_ids, val_ids


def fit_eval_xgb(df_tr, df_va, params):
    """Engineer + fold-safe zip_target + tuned XGBoost on `df_tr`, eval on `df_va`."""
    X_tr = _engineer(df_tr)
    X_va = _engineer(df_va)
    enc = ZipTargetEncoder().fit(X_tr, df_tr["price"].values)
    X_tr["zip_target"] = enc.transform(X_tr)["zip_target"]
    X_va["zip_target"] = enc.transform(X_va)["zip_target"]
    y_tr, y_va = df_tr["price"].values, df_va["price"].values
    model = make_champion_estimator().fit(X_tr[COLS], y_tr)
    return evaluate(y_va, model.predict(X_va[COLS])), len(df_tr), len(df_va)


def nearest_neighbor_km(train_df, val_df):
    """For each val property, distance to its nearest train property."""
    tr = train_df[["lat", "long"]].to_numpy()
    va = val_df[["lat", "long"]].to_numpy()
    out = np.empty(len(va))
    for i in range(len(va)):
        out[i] = haversine_km(va[i, 0], va[i, 1], tr[:, 0], tr[:, 1]).min()
    return out


def run():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train = df[df["id"].isin(tr_ids)]
    val = df[df["id"].isin(va_ids)]
    params = load_tuned_params()

    # spatial proximity audit (random split)
    nn_km = nearest_neighbor_km(train, val)
    prox = {
        "median_nn_km": float(np.median(nn_km)),
        "pct_val_within_1km_of_train": float((nn_km <= 1.0).mean() * 100),
        "pct_val_within_10km_of_train": float((nn_km <= 10.0).mean() * 100),
    }

    # controlled comparison: same model, same features, two splits
    random_met, random_tr, random_va = fit_eval_xgb(train, val, params)

    s_tr_ids, s_va_ids = spatial_split(df["id"].tolist(), df[["lat", "long"]].to_numpy())
    s_train, s_val = df[df["id"].isin(s_tr_ids)], df[df["id"].isin(s_va_ids)]
    spatial_met, spatial_tr, spatial_va = fit_eval_xgb(s_train, s_val, params)

    result = {
        "method": "Controlled: same tuned XGBoost + same engineered features; zip_target fitted on each split's training fold only",
        "tuned_params": {k: float(v) for k, v in params.items()},
        "primary": {"split": "random 80/20", "train_size": random_tr, "val_size": random_va},
        "spatial_proximity": prox,
        "robustness": {
            "random_split": {k: round(float(v), 4) for k, v in random_met.items()},
            "spatial_split": {k: round(float(v), 4) for k, v in spatial_met.items()},
            "spatial_train_size": spatial_tr,
            "spatial_val_size": spatial_va,
            "r2_delta": round(spatial_met["r2"] - random_met["r2"], 4),
            "rmse_increase_pct": round((spatial_met["rmse"] - random_met["rmse"]) / random_met["rmse"] * 100, 2),
            "note": (
                "Random and spatial legs differ ONLY in how train/validation were split. "
                "The gap therefore measures geographic generalisation of the final model, "
                "not a model/feature difference."
            ),
        },
    }

    out = REPORTS_DIR / "split_strategy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(result).to_json(out, indent=2, default_handler=str)
    print(pd.json_normalize(result).T.to_string())


if __name__ == "__main__":
    run()