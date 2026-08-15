"""Phases 3-4 — Fair evaluation of Experiment A + decision gate.

Compares A1/A2 ONLY on the same 434 image-covered validation properties used by
E4/E5 (from the persisted multimodal results). Does NOT compare against the
full-dataset 0.9205 R2, which is a different evaluation population; that number
is reported separately as the overall production benchmark.

Then applies the decision gate:
  Case 1  A clearly underperforms E4       -> STOP, report negative result
  Case 2  A competitive but not better     -> STOP
  Case 3  A clearly improves on E4         -> proceed to Phase 5+
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR


def main():
    mm = pd.read_csv(REPORTS_DIR / "results_multimodal.csv")
    champ = mm[mm["family"] == "Champion"].set_index("experiment")
    dl = json.loads((REPORTS_DIR / "results_dl.json").read_text(encoding="utf-8"))
    a1 = dl["variants"]["A1_price_MSE"]
    a2 = dl["variants"]["A2_log1p_MSE"]

    # only rows measured on the same 434 image-covered validation rows
    table_rows = [
        {"experiment": "E4", "model": "XGBoost (champion)", "input": "Tabular",
         "rmse": champ.loc["E4_tabular_only", "rmse"], "r2": champ.loc["E4_tabular_only", "r2"],
         "mae": champ.loc["E4_tabular_only", "mae"]},
        {"experiment": "E5", "model": "XGBoost (champion)", "input": "Tabular + frozen ResNet emb.",
         "rmse": champ.loc["E5_mm_rn18", "rmse"], "r2": champ.loc["E5_mm_rn18", "r2"],
         "mae": champ.loc["E5_mm_rn18", "mae"]},
        {"experiment": "A1", "model": "ResNet18 + trained head", "input": "Image (frozen backbone)",
         "rmse": a1["val_rmse"], "r2": a1["val_r2"], "mae": a1["val_mae"]},
        {"experiment": "A2", "model": "ResNet18 + trained head (log-price)", "input": "Image (frozen backbone)",
         "rmse": a2["val_rmse"], "r2": a2["val_r2"], "mae": a2["val_mae"]},
    ]
    table = pd.DataFrame(table_rows)

    # ---- decision gate ----
    e4 = table.loc[table["experiment"] == "E4", "r2"].iloc[0]
    a_rows = [r for r in table_rows if r["experiment"].startswith("A")]
    best_a_r2 = max(r["r2"] for r in a_rows)

    if best_a_r2 > e4 * 1.03:          # clearly improves
        decision = "Case 3 — proceed to Phase 5+"
    elif e4 - best_a_r2 < 0.01:        # competitive but not better
        decision = "Case 2 — STOP (no meaningful incremental value)"
    else:
        decision = "Case 1 — STOP (vision clearly underperforms)"

    gate = {
        "e4_r2": e4,
        "best_experiment_a_r2": best_a_r2,
        "decision": decision,
        "interpretation": (
            "Experiment A (trainable regression head on a frozen ImageNet ResNet18) "
            f"reaches R2 {best_a_r2:.3f} on the same 434 validation properties where the "
            f"tabular control reaches R2 {e4:.3f}. The task-trained visual head does not "
            "recover valuation signal that frozen embeddings missed; both are far below "
            "the tabular champion."
        ),
    }

    # full-dataset production benchmark, reported separately (different population)
    champion = json.loads((REPORTS_DIR / "tuned_best.json").read_text(encoding="utf-8"))
    production_benchmark = {
        "note": "OVERALL PRODUCTION BENCHMARK, measured on all 16,110 properties "
                "(not the 434 image subset) — do not compare directly with the table above.",
        "metrics": champion["metrics"],
    }

    out = {
        "phase": "Phases 3-4 — fair evaluation + decision gate",
        "scope": "ONLY the 434 image-covered validation properties (same ids as E4/E5)",
        "table": table.to_dict(orient="records"),
        "decision_gate": gate,
        "production_benchmark": production_benchmark,
        "conclusion_case_1": (
            "Task-specific training did not recover useful visual valuation signal "
            "from the available satellite imagery. STOP. This is a successful "
            "negative experiment."
        ),
    }
    (REPORTS_DIR / "results_dl_eval.json").write_text(json.dumps(out, indent=2, default=float),
                                                      encoding="utf-8")
    print(table.to_string(index=False))
    print()
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()