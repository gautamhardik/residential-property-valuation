"""Phase 0 — Freeze the existing champion: record a reproducibility snapshot.

Captures checksums and the champion's canonical metrics so the DL extension can
never silently alter the production baseline. Nothing here is modified; this is
purely a read-only snapshot for the QA cascade (Phase 16).
"""
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR, PREDICTIONS_DIR, APP_MODELS_DIR


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    champion = json.loads((REPORTS_DIR / "tuned_best.json").read_text(encoding="utf-8"))
    metrics = champion["metrics"]
    assert abs(metrics["rmse"] - 103802.7632) < 1e-3, "champion RMSE drifted"
    assert abs(metrics["r2"] - 0.9205) < 1e-4, "champion R2 drifted"

    snapshot = {
        "phase": "Phase 0 — champion freeze snapshot",
        "created_for": "DL extension (final research pass)",
        "champion": {
            "model_type": champion["model_type"],
            "selection": champion.get("selection"),
            "metrics": metrics,
            "asserted": "RMSE ~= $103,802.8, R2 ~= 0.9205",
        },
        "checksums_sha256_16": {
            "submission.csv": sha16(PREDICTIONS_DIR / "submission.csv"),
            "models/deployed/tabular_model.joblib": sha16(APP_MODELS_DIR / "tabular_model.joblib"),
            "models/deployed/tabular_pipeline.joblib": sha16(APP_MODELS_DIR / "tabular_pipeline.joblib"),
            "reports/results_multimodal.csv": sha16(REPORTS_DIR / "results_multimodal.csv"),
            "reports/results_ensemble.csv": sha16(REPORTS_DIR / "results_ensemble.csv"),
            "reports/tuned_best.json": sha16(REPORTS_DIR / "tuned_best.json"),
        },
        "do_not_modify": [
            "Final XGBoost model", "XGBoost features", "E1-E5 results",
            "submission.csv", "existing /predict endpoint",
            "existing validation methodology",
        ],
    }
    out = REPORTS_DIR / "baseline_snapshot.json"
    out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()