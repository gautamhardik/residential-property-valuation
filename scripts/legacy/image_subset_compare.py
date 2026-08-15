"""Image-subset-only comparison + final DL report.

Every valuation model exposed in the DL-vs-frozen comparison is scored on the
SAME validation set: the 434 image-covered properties of the canonical split.

Sources (no retraining):
  - E4 / E4B / E5  -> reports/results_multimodal.csv  (already image-subset-only)
  - Final XGBoost  -> reports/val_predictions_best_tabular.csv out-of-sample
                       predictions, subset to the 434 image-covered ids.
  - Trainable head / partial fine-tune -> reports/results_dl.json (DL runs)

Writes reports/results_image_subset.json and figures/fig_dl.png (image-subset view).
NOTE: The full-set Final XGBoost number ($103.8k / 0.9205) is deliberately NOT
shown in this figure because it is measured on all 16,110 properties, not on the
image-covered subset — comparing it against the CNN rows would be apples-to-oranges.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR, FIGURES_DIR
from src.data.load import load_clean_train, canonical_split
from src.satellite.align import is_valid_image

MM = pd.read_csv(REPORTS_DIR / "results_multimodal.csv")


def champion(name):
    row = MM[(MM["experiment"] == name) & (MM["family"] == "Champion")].iloc[0]
    return float(row["rmse"]), float(row["r2"])


def main():
    # --- image-covered val ids ---
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    val = df[df["id"].isin(va_ids)]
    img_val_ids = val["id"][val["id"].map(is_valid_image)].tolist()
    print(f"image-covered val ids: {len(img_val_ids)}")

    # --- Final XGBoost on the image subset (out-of-sample champion predictions) ---
    vp = pd.read_csv(REPORTS_DIR / "val_predictions_best_tabular.csv")
    sub = vp[vp["id"].isin(img_val_ids)]
    y = sub["price"].values
    p = sub["predicted"].values
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    print(f"Final XGBoost on image subset -> RMSE ${rmse:,.0f}  R2 {r2:.4f}")

    e4_rmse, e4_r2 = champion("E4_tabular_only")
    e4b_rmse, e4b_r2 = champion("E4B_image_only_rn18")
    e5_rmse, e5_r2 = champion("E5_mm_rn18")

    dl = json.loads((REPORTS_DIR / "results_dl.json").read_text(encoding="utf-8"))
    dl_rows = {r["label"]: r for r in dl["models"]
               if r.get("input") not in ("tabular", "image (frozen embeddings)",
                                         "tabular + frozen embeddings",
                                         "full tabular (all 16,110)")}

    rows = [
        {"label": "E4_tabular_control", "family": "tabular", "model": "champion XGBoost",
         "rmse": e4_rmse, "mae": None, "r2": e4_r2},
        {"label": "E4B_image_only_frozen", "family": "image (frozen embeddings)", "model": "XGBoost on resnet18 pool",
         "rmse": e4b_rmse, "mae": None, "r2": e4b_r2},
        {"label": "E5_mm_rn18", "family": "tabular + frozen embeddings", "model": "XGBoost",
         "rmse": e5_rmse, "mae": None, "r2": e5_r2},
        {"label": "Final_XGBoost_image_subset", "family": "tabular", "model": "champion XGBoost (out-of-sample)",
         "rmse": rmse, "mae": mae, "r2": r2},
    ]
    for r in sorted(dl_rows.values(), key=lambda x: x["r2"], reverse=True):
        if r["label"].startswith("Trained head") or r["label"].startswith("Partial FT"):
            rows.append({"label": r["label"], "family": "image (trained CNN)",
                         "model": "resnet18 + trained head" if "Trained head" in r["label"]
                         else "resnet18 layer4 partial fine-tune",
                         "rmse": r["rmse"], "mae": r.get("mae"), "r2": r["r2"]})

    out = {
        "scope": "494 excluded; ONLY the 434 image-covered validation properties",
        "n_image_val": len(img_val_ids),
        "note_full_set_excluded": (
            "cite full-set Final XGBoost (RMSE 103,803 / R2 0.9205 on all 16,110) "
            "only as a context figure; it is NOT on the image subset and is not "
            "part of this comparison."),
        "models": rows,
        "conclusion": (
            "A trained CNN (partial fine-tune R2 0.296) more than doubles the frozen "
            "embedding signal (0.138) but still falls far short of the tabular control "
            "(0.872). Satellite imagery alone is insufficient; imagery can at most act "
            "as a weak auxiliary signal."
        ),
    }
    (REPORTS_DIR / "results_image_subset.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nSaved", REPORTS_DIR / "results_image_subset.json")

    # --- figure (image-subset comparable rows only) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = [
        "Final_XGBoost_image_subset", "E4_tabular_control",
        "E5_mm_rn18", "E4B_image_only_frozen",
    ] + sorted([r["label"] for r in rows if r["label"] not in {
        "Final_XGBoost_image_subset", "E4_tabular_control", "E5_mm_rn18", "E4B_image_only_frozen"}],
        key=lambda x: x, reverse=True)
    labels = []
    rmses, r2s = [], []
    for o in order:
        r = next(x for x in rows if x["label"] == o)
        short = {"E4_tabular_control": "Tabular control (E4)",
                 "E4B_image_only_frozen": "Frozen embeddings (E4B)",
                 "E5_mm_rn18": "Tabular + frozen emb (E5)",
                 "Final_XGBoost_image_subset": "Final XGBoost (image subset)"}.get(o, o.split("(")[0])
        labels.append(short)
        rmses.append(r["rmse"] / 1000)
        r2s.append(r["r2"])
    colors = ["#0f766e"] + ["#94a3b8"] + ["#94a3b8"] + ["#94a3b8"] + ["#2563eb"] * 4
    bars = ax.bar(range(len(labels)), rmses, color=colors, edgecolor="white")
    ax.set_ylim(0, max(rmses) * 1.15)
    for i, (v, r2) in enumerate(zip(rmses, r2s)):
        ax.text(i, v + 4, f"${v:,.0f}K\nR2 {r2:.2f}", ha="center", fontsize=9)
    margin = len(labels) - 4.5
    ax.axvline(margin, color="#cbd5e1", lw=1, ymax=0.95)
    ax.text(margin - 0.05, ax.get_ylim()[1] * 0.97, "trained CNN",
            rotation=90, va="top", ha="right", fontsize=10, color="#2563eb")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE on the 434 image val properties (thousands USD)")
    ax.set_title("Image-only comparison: same 434 validation properties, all models\n"
                 "satellite signal stays far below the tabular control regardless of CNN training",
                 fontsize=12, pad=12)
    fig.savefig(FIGURES_DIR / "fig_dl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIGURES_DIR / "fig_dl.png")


if __name__ == "__main__":
    main()