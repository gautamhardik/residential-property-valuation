"""Phase 11-12 — select final model, retrain on all clean training data, submit.

Final model = tuned XGBoost (engineered tabular). The multimodal and image-only
models were all inferior on the fair subset experiment (see results_multimodal.csv),
so the tabular model is the evidence-based selection.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PREDICTIONS_DIR, REPORTS_DIR
from src.data.load import load_clean_train, load_clean_test
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.inference.artifacts import save_tabular_artifacts
from src.models.train import make_champion_estimator, load_champion_config

COLS = [c for c in FEATURE_COLS if c != "price"]

CHAMPION_TYPE, _ = load_champion_config()
CHAMPION_LABEL = {"xgboost": "XGBoost", "catboost": "CatBoost"}.get(CHAMPION_TYPE, CHAMPION_TYPE.title())


def prepare(df):
    X = _engineer(df)
    return X[COLS]


def run():
    train = load_clean_train()
    test = load_clean_test()

    X_all = _engineer(train)
    global_mean, zip_enc = fit_target_encoder(X_all)
    X_all["zip_target"] = X_all["zipcode"].map(zip_enc).fillna(global_mean)
    X_test = _engineer(test)
    X_test["zip_target"] = X_test["zipcode"].map(zip_enc).fillna(global_mean)

    model = make_champion_estimator().fit(X_all[COLS], train["price"].values)
    save_tabular_artifacts(model, zip_enc, global_mean, COLS, model_type=CHAMPION_TYPE)

    preds = model.predict(X_test[COLS])
    submission = pd.DataFrame({"id": test["id"].values, "predicted_price": preds})

    n_dups = submission["id"].duplicated().sum()
    n_missing = int(submission["predicted_price"].isna().sum())
    assert len(submission) == len(test), "row count mismatch (1:1 with test)"
    assert set(submission["id"]) == set(test["id"]), "id correspondence mismatch"
    assert n_missing == 0, "missing predictions"
    assert submission["predicted_price"].dtype.kind in "fi", "non-numeric predictions"

    out = PREDICTIONS_DIR / "submission.csv"
    submission.to_csv(out, index=False)

    summary = {
        "final_model": f"{CHAMPION_LABEL} tuned (champion), engineered tabular features",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "submission_rows": int(len(submission)),
        "duplicate_ids_in_test": int(n_dups),
        "missing_predictions": int(n_missing),
        "submission": str(out.relative_to(out.parents[1])) if out.exists() else str(out),
        "prediction_stats": {
            "min": float(preds.min()), "mean": float(preds.mean()),
            "max": float(preds.max()),
        },
    }
    (REPORTS_DIR / "final_model.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nsubmission head:")
    print(submission.head())


if __name__ == "__main__":
    run()