"""Phase 10a — SHAP for the best tuned tabular model.

Global importance (mean |SHAP|) + individual explanations (waterfall) on a sample
of the untouched validation fold. SHAP describes feature contributions, not
causation.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR, FIGURES_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.models.train import evaluate, make_champion_estimator

N_SAMPLE = 300
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
    print("Overall on holdout:", evaluate(y_va, model.predict(X_va)))

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_va), size=min(N_SAMPLE, len(X_va)), replace=False)
    X_sample = X_va.iloc[sample_idx]
    y_sample = val.iloc[sample_idx]["price"].values

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = [{"feature": cols[i], "mean_abs_shap": float(mean_abs[i])} for i in order[:20]]

    # figures
    shap.summary_plot(shap_values, X_sample, feature_names=cols, show=False, max_display=20)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # waterfall for two individual predictions
    for label, orderk in (("lowest", np.argsort(y_sample)[:1]), ("highest", np.argsort(y_sample)[-1:])):
        k = int(orderk[0])
        shap.Explanation(
            shap_values[k], base_values=explainer.expected_value,
            data=X_sample.iloc[k].values, feature_names=cols,
        )
        waterfall = shap.Explanation(values=shap_values[k],
                                     base_values=explainer.expected_value[()] if hasattr(explainer.expected_value, "__iter__") else explainer.expected_value,
                                     data=X_sample.iloc[k].values,
                                     feature_names=cols)
        shap.waterfall_plot(waterfall, max_display=12, show=False)
        plt.savefig(FIGURES_DIR / f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    out = {
        "overall": evaluate(y_va, model.predict(X_va)),
        "top_features": top,
        "n_sampled": len(X_sample),
    }
    (REPORTS_DIR / "shap_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved figures to", FIGURES_DIR)
    print("Top features:", [t["feature"] for t in top[:10]])


if __name__ == "__main__":
    run()