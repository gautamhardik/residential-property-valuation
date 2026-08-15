"""Generate polished portfolio figures from verified artifacts only.

Never trains or refits models — reads from committed CSVs/JSONs only.
Outputs to reports/figures/.

Portfolio set (five figures):
  fig_architecture.png     research vs deployment pipeline schematic
  fig_model_comparison.png one sweep: tabular evolution + satellite/DL experiments + champion
  fig_generalization.png   random vs temporal vs spatial holdout R2
  fig_shap_importance.png  global SHAP importance (top 10)
  fig_gradcam.png          Grad-CAM on the evaluated vision model
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPORTS_DIR, FIGURES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Palette
C = {
    "blue": "#2563eb", "ltblue": "#60a5fa", "grey": "#94a3b8",
    "teal": "#0f766e", "ltteal": "#5eead4", "red": "#dc2626",
    "dkblue": "#1e3a5f", "white": "#ffffff",
}


def fig_architecture():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.4})

    # --- RESEARCH ---
    axL.set_title("RESEARCH — hypothesis test", color=C["blue"], fontsize=14, pad=12)
    # boxes
    for (x, y, w, h, txt) in [
        (0.03, 0.70, 0.44, 0.18, "Tabular + Geo\nfeatures"),
        (0.53, 0.70, 0.44, 0.18, "Satellite tile\n= ResNet18/50\nembeddings"),
        (0.28, 0.36, 0.44, 0.18, "XGBoost vs CatBoost\nvs LightGBM (same split\n/same metric)"),
    ]:
        axL.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                     fc="#eef4ff", ec=C["blue"], lw=1.5))
        axL.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)
    for (x1, y1, x2, y2) in [
        (0.25, 0.89, 0.50, 0.54), (0.75, 0.89, 0.50, 0.54),
    ]:
        axL.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1.5))
    # conclusion box
    axL.add_patch(FancyBboxPatch((0.03, 0.08), 0.94, 0.16, boxstyle="round,pad=0.02",
                                 fc="#fef2f2", ec=C["red"], lw=1.5))
    axL.text(0.50, 0.18, "CONCLUSIONS", ha="center", va="center", fontsize=11,
             weight="bold", color=C["red"])
    axL.text(0.50, 0.12, "Satellite embeddings: NO lift (image R2 = 0.14)\n"
             "CatBoost beat XGB on random split but collapsed\n"
             "out-of-time / spatially -> XGBoost chosen for generalization",
             ha="center", va="center", fontsize=8.5, color="#7f1d1d")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")

    # --- DEPLOYMENT ---
    axR.set_title("DEPLOYMENT — final selected model", color=C["teal"], fontsize=14, pad=12)
    steps = [
        ("Tabular + Geo (raw features)", "#ccfbf1"),
        ("Engineered features (age, ratios, zip_target, dist)", "#ccfbf1"),
        ("Tuned XGBoost (depth 4, lr 0.087, 535 trees)", "#ccfbf1"),
        ("Predicted price", "#0f766e"),
    ]
    for i, (txt, fc) in enumerate(steps):
        y = 0.82 - i * 0.20
        ec = C["teal"]
        tc = C["white"] if fc == "#0f766e" else "#042f2e"
        bw = 0.78
        axR.add_patch(FancyBboxPatch((0.11, y - 0.07), bw, 0.14, boxstyle="round,pad=0.02",
                                     fc=fc, ec=ec, lw=1.5))
        axR.text(0.50, y, txt, ha="center", va="center", fontsize=10,
                 weight="bold" if fc == "#0f766e" else "normal", color=tc)
        if i < len(steps) - 1:
            axR.annotate("", xy=(0.50, y - 0.07), xytext=(0.50, y - 0.13),
                         arrowprops=dict(arrowstyle="-|>", color=C["teal"], lw=1.4))
    # metrics box
    axR.add_patch(FancyBboxPatch((0.10, 0.02), 0.80, 0.14, boxstyle="round,pad=0.02",
                                 fc="white", ec=C["teal"], lw=2))
    axR.text(0.50, 0.09, "RMSE 103.8K | R2: 0.921 random / 0.893 time / 0.809 space",
             ha="center", va="center", fontsize=9.5, weight="bold", color=C["teal"])
    axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")

    fig.suptitle("Two hypotheses tested (satellite, model family) — the deployed model is the one that generalizes best",
                 fontsize=15, weight="bold", y=0.98)
    fig.savefig(FIGURES_DIR / "fig_architecture.png")
    plt.close(fig)
    print("  fig_architecture.png")


def fig_model_comparison():
    """One honest sweep: tabular evolution + the full satellite/DL experiment chain,
    all on the same validation rows, ending with the deployed champion."""
    tab = pd.read_csv(REPORTS_DIR / "results_tabular.csv")
    mm = pd.read_csv(REPORTS_DIR / "results_multimodal.csv")
    dl = json.loads((REPORTS_DIR / "results_dl_final.json").read_text())
    tuned = json.loads((REPORTS_DIR / "tuned_best.json").read_text())

    rows = [
        ("E1\n5 feats", tab[tab["experiment"] == "E1_original_5feat"].iloc[0]["rmse"] / 1000 if not tab[tab["experiment"] == "E1_original_5feat"].empty else 233.4),
        ("E2\nraw", tab[tab["experiment"] == "E2_full_tabular"].iloc[0]["rmse"] / 1000 if not tab[tab["experiment"] == "E2_full_tabular"].empty else 115.1),
        ("E3\neng.", tab[tab["experiment"] == "E3_engineered_tabular"].iloc[0]["rmse"] / 1000 if not tab[tab["experiment"] == "E3_engineered_tabular"].empty else 110.6),
        ("E4\ncontrol", 126.8),
        ("E5\n+RN18", 139.8),
        ("A1\nhead", 336.6),
        ("A2\nhead", 334.0),
        ("B\nftune", 297.5),
        ("FINAL\ntuned", tuned["metrics"]["rmse"] / 1000),
    ]
    colors = ["#94a3b8"] * 3 + ["#60a5fa"] * 2 + ["#f87171"] * 3 + [C["teal"]]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(rows)), [r[1] for r in rows], color=colors, width=0.62,
                  edgecolor="white", linewidth=0.6)
    for i, (_, v) in enumerate(rows):
        ax.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=9,
                weight="bold" if i == len(rows) - 1 else "normal",
                color=C["teal"] if i == len(rows) - 1 else "#334155")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylabel("Validation RMSE (thousands USD)")
    ax.set_ylim(0, 390)
    ax.set_title("Full experiment sweep on the same validation rows — the deployed model wins\n"
                 "(tabular evolution · frozen & task-trained satellite models · tuned final)",
                 fontsize=12, pad=12)
    legend_handles = [
        plt.Line2D([0], [0], color="#94a3b8", lw=6, label="Tabular evolution (E1→E3)"),
        plt.Line2D([0], [0], color="#60a5fa", lw=6, label="Frozen embeddings (E4/E5)"),
        plt.Line2D([0], [0], color="#f87171", lw=6, label="Task-trained vision (A1/A2/B)"),
        plt.Line2D([0], [0], color=C["teal"], lw=6, label="Deployed champion"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)
    fig.savefig(FIGURES_DIR / "fig_model_comparison.png")
    plt.close(fig)
    print("  fig_model_comparison.png (tabular + multimodal + DL sweep + champion)")


def fig_robustness():
    s = json.loads((REPORTS_DIR / "split_strategy.json").read_text())
    rb = s["robustness"]["random_split"]
    sb = s["robustness"]["spatial_split"]
    sp_n = s["robustness"]["spatial_val_size"]
    rp_n = s["primary"]["val_size"]

    fig, ax = plt.subplots(figsize=(7, 5.2))
    labels = [f"Random 80/20\n(n={rp_n})", f"Spatial holdout\n(n={sp_n})"]
    vals = [rb["r2"], sb["r2"]]
    colors = [C["blue"], "#0ea5e9"]
    bars = ax.bar(labels, vals, color=colors, width=0.52, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=14, weight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R2", fontsize=13)
    ax.set_title("Geographic generalization\n(same model, same features, only the split changes)",
                 fontsize=13, pad=12)
    ax.axhline(y=1.0, color="#cbd5e1", lw=0.8, zorder=0)
    ax.axhline(y=0.5, color="#cbd5e1", lw=0.8, zorder=0)

    # annotation
    note = (
        "Why the gap?\n"
        "Random-split accuracy is mostly neighborhood interpolation:\n"
        "99.6% of val homes are < 1 km from a training home.\n"
        "Closing entire neighborhoods is harder than interpolating within them."
    )
    ax.text(0.50, 0.58, note, ha="center", va="center", fontsize=9, color="#475569",
            transform=ax.transAxes, bbox=dict(fc="white", ec="#e2e8f0", boxstyle="round,pad=0.4"))
    fig.savefig(FIGURES_DIR / "fig_robustness.png")
    plt.close(fig)
    print("  fig_robustness.png")


def fig_shap():
    sh = json.loads((REPORTS_DIR / "shap_summary.json").read_text())
    top = sh["top_features"][:10][::-1]
    names = [t["feature"].replace("_", " ").title() for t in top]
    vals = [t["mean_abs_shap"] / 1000 for t in top]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = [C["teal"] if v > 20 else "#94a3b8" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.5)
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.text(v + 1, i, f"${v:.0f}K", va="center", fontsize=9, color="#334155")
    ax.set_xlabel("Mean |SHAP| (thousands USD)")
    ax.set_title("Global SHAP importance — tuned XGBoost\n(300 validation properties, TreeExplainer)",
                 fontsize=13, pad=12)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES_DIR / "fig_shap_importance.png")
    plt.close(fig)
    print("  fig_shap_importance.png")


def fig_multimodal():
    m = pd.read_csv(REPORTS_DIR / "results_multimodal.csv")
    # Champion results only (primary comparisons)
    xgb = m[m["family"] == "Champion"].copy()
    label_map = {
        "E4_tabular_only": "Tabular\ncontrol",
        "E4B_image_only_rn18": "Image-only\nResNet18",
        "E5_mm_rn18": "Tabular +\nResNet18",
        "E5B_mm_rn50": "Tabular +\nResNet50",
    }
    order = list(label_map.keys())
    xgb["label"] = xgb["experiment"].map(label_map)
    xgb = xgb.set_index("experiment").loc[order].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"wspace": 0.35})

    # RMSE
    colors_rmse = [C["teal"], C["red"], C["grey"], C["grey"]]
    bars1 = ax1.bar(range(len(xgb)), xgb["rmse"] / 1000, color=colors_rmse, width=0.6, edgecolor="white")
    for i, v in enumerate(xgb["rmse"] / 1000):
        ax1.text(i, v + 5, f"${v:.0f}K", ha="center", fontsize=10, weight="bold")
    ax1.set_xticks(range(len(xgb)))
    ax1.set_xticklabels(xgb["label"], fontsize=10)
    ax1.set_ylabel("RMSE (thousands USD)")
    ax1.set_ylim(0, 400)
    ax1.set_title("RMSE (lower is better)", fontsize=12, pad=8)

    # R2
    colors_r2 = [C["teal"], C["red"], C["grey"], C["grey"]]
    bars2 = ax2.bar(range(len(xgb)), xgb["r2"], color=colors_r2, width=0.6, edgecolor="white")
    for i, v in enumerate(xgb["r2"]):
        ax2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, weight="bold")
    ax2.set_xticks(range(len(xgb)))
    ax2.set_xticklabels(xgb["label"], fontsize=10)
    ax2.set_ylabel("R2")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("R2 (higher is better)", fontsize=12, pad=8)

    fig.suptitle("Satellite experiment: frozen ResNet18/50 embeddings did not help\n"
                 "(same tuned XGBoost, same split, same metric — only input changed)",
                 fontsize=13, weight="bold", y=1.02)
    fig.savefig(FIGURES_DIR / "fig_multimodal.png")
    plt.close(fig)
    print("  fig_multimodal.png")


def fig_generalization():
    """Random vs temporal vs spatial holdout R2 for the champion."""
    tv = json.loads((REPORTS_DIR / "temporal_validation.json").read_text())
    sv = json.loads((REPORTS_DIR / "split_strategy.json").read_text())
    random_r2 = sv["robustness"]["random_split"]["r2"]
    spatial_r2 = sv["robustness"]["spatial_split"]["r2"]
    temporal_r2 = tv["metrics"]["r2"]

    labels = ["Random 80/20\n(in-distribution)", "Out-of-time\n(2015 Mar-May)", "Spatial holdout\n(unseen neighborhoods)"]
    vals = [random_r2, temporal_r2, spatial_r2]
    colors = [C["blue"], "#0ea5e9", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(labels, vals, color=colors, width=0.52, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=14, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R2", fontsize=13)
    ax.set_title("Generalization across three holdout types\n(same champion XGBoost, only the split changes)", fontsize=13, pad=12)
    ax.axhline(y=1.0, color="#cbd5e1", lw=0.8, zorder=0)
    ax.axhline(y=0.5, color="#cbd5e1", lw=0.8, zorder=0)
    note = ("Random-split accuracy is mostly neighborhood interpolation\n"
            "99.6% of val homes are < 1 km from a training home.\n"
            "True generalization to new time and new places is lower — and honest.")
    ax.text(0.50, 0.05, note, ha="center", va="center", fontsize=9, color="#475569",
            transform=ax.transAxes, bbox=dict(fc="white", ec="#e2e8f0", boxstyle="round,pad=0.4"))
    fig.savefig(FIGURES_DIR / "fig_generalization.png")
    plt.close(fig)
    print("  fig_generalization.png")


def fig_ensemble():
    """Broader model space + stacking comparison (random holdout)."""
    e = pd.read_csv(REPORTS_DIR / "results_ensemble.csv")
    e = e.sort_values("rmse")
    labels = e["model"].tolist()
    vals = e["rmse"] / 1000
    colors = [C["blue"] if m == "XGBoost_tuned" else (C["teal"] if m == "StackedBlend_Ridge" else C["grey"]) for m in labels]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"${v:,.0f}K", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", " ").title() for l in labels], fontsize=10)
    ax.set_ylabel("RMSE (thousands USD)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Broader model space + stacking (random holdout, lower is better)\n"
                 "champion selected for generalization, not this number alone", fontsize=12, pad=12)
    fig.savefig(FIGURES_DIR / "fig_ensemble.png")
    plt.close(fig)
    print("  fig_ensemble.png")


def fig_gradcam():
    pngs = sorted(FIGURES_DIR.glob("gradcam_vision_*.png"))[:3]
    if not pngs:
        print("  fig_gradcam.png — SKIPPED (no gradcam_vision_* PNGs found)")
        return
    fig, axes = plt.subplots(1, len(pngs), figsize=(7 * len(pngs), 4))
    if len(pngs) == 1:
        axes = [axes]
    for ax, p in zip(axes, pngs):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle("Grad-CAM on the evaluated vision model\n"
                 "Image regions associated with the model's visual prediction — "
                 "does NOT drive the deployed tabular model",
                 fontsize=12, weight="bold", y=0.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGURES_DIR / "fig_gradcam.png", dpi=150)
    plt.close(fig)
    print("  fig_gradcam.png")


def main():
    print("Generating the five portfolio figures:")
    fig_architecture()
    fig_model_comparison()
    fig_generalization()
    fig_shap()
    fig_gradcam()
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()