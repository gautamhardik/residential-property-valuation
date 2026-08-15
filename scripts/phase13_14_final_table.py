"""Phases 13-14 — Authoritative final experiment table + scientific conclusion.

Assembles the one and only experiment table that the report/README cite, and
records the evidence-based conclusion for the DL extension. Uses only persisted
results; nothing is re-trained here.
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR


def load_eval_dl():
    return json.loads((REPORTS_DIR / "results_dl.json").read_text(encoding="utf-8"))


def load_serialization():
    return json.loads((REPORTS_DIR / "results_dl_serialization.json").read_text(encoding="utf-8"))


def main():
    mm = pd.read_csv(REPORTS_DIR / "results_multimodal.csv")
    champ = mm[mm["family"] == "Champion"].set_index("experiment")
    dl = load_eval_dl()
    a1 = dl["variants"]["A1_price_MSE"]
    a2 = dl["variants"]["A2_log1p_MSE"]
    ser = load_serialization()
    b_price = ser["verification"]

    # B (partial fine-tune) is the executed prior model. The planned E6
    # (tabular + task-trained greenfield embedding) was stopped by the Phase 4
    # gate, but E6 (tabular + DINOv2-vits14 embeddings) and PCA-ablated
    # embeddings WERE executed as separate representation checks; both degraded
    # the tabular control, so the negative result is representation-robust. See
    # reports/results_dinov2_fusion.json and reports/results_multimodal_pca.csv.
    rows = [
        # executed experiments, all on the SAME 434 image-covered validation rows
        {"exp": "E4", "model": "XGBoost (champion)", "input": "Tabular",
         "rmse": champ.loc["E4_tabular_only", "rmse"], "r2": champ.loc["E4_tabular_only", "r2"],
         "source": "reports/results_multimodal.csv (E4_tabular_only, XGB)"},
        {"exp": "E5", "model": "XGBoost (champion)", "input": "Tabular + frozen ResNet emb.",
         "rmse": champ.loc["E5_mm_rn18", "rmse"], "r2": champ.loc["E5_mm_rn18", "r2"],
         "source": "reports/results_multimodal.csv (E5_mm_rn18, XGB)"},
        {"exp": "A1", "model": "ResNet18 + trained head", "input": "Image",
         "rmse": a1["val_rmse"], "r2": a1["val_r2"],
         "source": "reports/results_dl.json (A1_price_MSE, best epoch)"},
        {"exp": "A2", "model": "ResNet18 + trained head (log-price)", "input": "Image",
         "rmse": a2["val_rmse"], "r2": a2["val_r2"],
         "source": "reports/results_dl.json (A2_log1p_MSE, best epoch)"},
        {"exp": "B", "model": "ResNet18 partial fine-tune (layer4)", "input": "Image",
         "rmse": b_price["py_rmse"], "r2": b_price["py_r2"],
         "source": "reports/results_dl_serialization.json (PyTorch on 434 val)"},
    ]
    table = pd.DataFrame(rows)

    # Production benchmark (full dataset population) — separate, NOT comparable to the 434 rows.
    prod = {
        "exp": "Production",
        "model": "Tuned XGBoost (champion)",
        "input": "Full engineered tabular + geospatial (all 16,110)",
        "rmse": 103802.7632,
        "r2": 0.9205,
        "source": "reports/tuned_best.json (random 80/20 holdout, n=3,222)",
    }
    table = pd.concat([table, pd.DataFrame([prod])], ignore_index=True)

    # E6 (tabular + DINOv2-vits14 frozen embeddings) WAS executed as a separate
    # representation check and degraded the model (R2 0.9117 vs 0.9203 tabular on
    # the DINOv2-covered split, reports/results_dinov2_fusion.json). The planned
    # E6 with a TASK-TRAINED embedding was stopped by the gate.
    e6_decision = {
        "executed": True,
        "variant": "tabular + DINOv2-vits14 frozen embeddings (reports/results_dinov2_fusion.json)",
        "r2": 0.9116874600490137,
        "reason": "An E6 fusion with DINOv2-vits14 embeddings was executed and degraded "
                  "the tabular control (R2 0.9203 -> 0.9117), reproducing the frozen-"
                  "embedding negative result with a second representation. The planned "
                  "E6 with a task-trained embedding was additionally gated after "
                  "Experiment A (R2 0.113 vs E4 0.872).",
    }

    conclusion = {
        "outcome": "OUTCOME A — the DL negative result stands.",
        "summary": (
            "Frozen ImageNet embeddings failed to improve valuation (E4 0.872 -> "
            "E5 0.844). Training a task-specific ResNet18 regression head "
            "(A1 0.099 / A2 0.113) also failed to recover meaningful visual signal. "
            "The only model that measurably moved visual information was a partial "
            "fine-tune of layer4 (B, R2 0.296), yet it still trails the tabular "
            "control (0.872) by a wide margin. Parallel representation checks "
            "(E6 DINOv2-vits14 fusion R2 0.9117 vs 0.9203; PCA-ablated embeddings, "
            "best tabular variant 0.8634 vs 0.8721 control) also failed to beat the "
            "tabular baseline; the negative result is robust across representations."
        ),
        "scientific_statement": (
            "Under the available satellite coverage (2,189 of 16,110 properties, 13.6%) "
            "and single-tile 256px imagery, the visual modality provides limited "
            "incremental information beyond the structured/geospatial features. "
            "Representation learning narrows the frozen-embedding gap but does not "
            "approach the tabular signal."
        ),
        "decision": "XGBoost remains the production champion. The vision model is "
                    "archived as research evidence only and is not exposed in the "
                    "production app (no vision prediction endpoint exists).",
        "e6": e6_decision,
    }

    out = {
        "phase": "Phases 13-14 — final experiment table",
        "section": "E4/E5/A1/A2/B rows measured on the SAME 434 image-covered validation "
                   "properties (1,755-train image-covered subset). The Production row is a "
                   "DIFFERENT population: the full 16,110-property random holdout — never "
                   "compare it directly to the 434-row experiments.",
        "table": table.to_dict(orient="records"),
        "production_benchmark_note": (
            "Full-dataset champion (all 16,110 properties, random 80/20 holdout of 3,222): "
            "RMSE $103,802.8, R2 0.9205. Different population; not directly comparable to the "
            "E4/E5/A1/A2/B rows."),
        "conclusion": conclusion,
    }
    (REPORTS_DIR / "results_dl_final.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")

    print(table.to_string(index=False))
    print()
    print(json.dumps(conclusion, indent=2))
    print("\nSaved", REPORTS_DIR / "results_dl_final.json")


if __name__ == "__main__":
    main()