"""Phase 16 — final QA cascade for the production tabular release."""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import app  # noqa: E402
from src.inference.predict import predict_single  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"


def check(label, cond, detail=""):
    print(f"[{PASS if cond else FAIL}] {label}  {detail}")
    return cond


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    all_ok = True
    root = Path(__file__).resolve().parents[1]
    client = TestClient(app)

    print("=== PHASE 16 QA CASCADE ===")

    manifest = json.loads((root / "reports" / "baseline_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]

    # 1. Frozen baseline
    all_ok &= check("baseline manifest exists", manifest["metadata"]["model_version"] == "xgboost-tuned-2026-08-15")
    for item in artifacts:
        got = sha256(root / item["path"])
        all_ok &= check(f"sha256 {item['path']}", got == item["sha256"], item["sha256"][:12])

    # 2. Production path
    payload = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1910,
        "sqft_lot": 7600,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1560,
        "sqft_basement": 0,
        "yr_built": 1975,
        "yr_renovated": 0,
        "zipcode": 98065,
        "lat": 47.5724,
        "long": -122.2300,
        "sqft_living15": 1840,
        "sqft_lot15": 7620,
    }
    offline = predict_single({**payload, "sale_year": 2015, "sale_quarter": 3})
    api = client.post("/predict", json={**payload, "sale_year": 2015, "sale_quarter": 3})
    all_ok &= check("health endpoint", client.get("/health").status_code == 200)
    all_ok &= check("/predict endpoint", api.status_code == 200)
    all_ok &= check("offline/api parity",
                    abs(float(api.json()["predicted_price"]) - float(offline["predicted_price"])) < 1e-6,
                    f"offline={offline['predicted_price']} api={api.json()['predicted_price']}")
    api_body = api.json()
    ls = api_body.get("local_shap") or {}
    band = api_body.get("error_band") or {}
    shap_parity = ls and abs(float(ls.get("expected_value", 0)) + float(ls.get("total_contribution", 0)) - float(api_body["predicted_price"])) < 1.0
    all_ok &= check("local SHAP present + parity", bool(ls) and shap_parity)
    all_ok &= check("empirical error band present", bool(band) and band.get("typical_error") and "not a per-property" in (band.get("note") or ""))
    all_ok &= check("no fake uncertainty interval in frontend", "Uncertainty" not in (root / "app" / "frontend" / "index.html").read_text(encoding="utf-8"))
    all_ok &= check("deprecated vision endpoint absent", client.post("/predict-image", json={"pid": 1}).status_code == 404)
    all_ok &= check("deprecated tile endpoint absent", client.get("/tile/1").status_code == 404)

    # 3. Documentation truth pass
    readme = (root / "README.md").read_text(encoding="utf-8")
    report_path = root / "reports" / "project_report.md"
    artifact_guide_path = root / "reports" / "ARTIFACT_GUIDE.md"
    report = report_path.read_text(encoding="utf-8")
    artifact_guide = artifact_guide_path.read_text(encoding="utf-8")
    all_ok &= check("README positions satellite as research", "research only" in readme.lower())
    all_ok &= check("report separates production vs research", "archived research evidence" in report and "do **not** expose a vision prediction path" in report)
    all_ok &= check("artifact guide present", "Authoritative Artifact Guide" in artifact_guide and "Final report" in artifact_guide)
    all_ok &= check("no live vision claim in README", "/predict-image" not in readme and "vision_only" not in readme)
    all_ok &= check("no live vision claim in report", "served via the API's `POST /predict-image`" not in report)

    # 4. Security hygiene
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    env_example = (root / ".env.example").read_text(encoding="utf-8")
    all_ok &= check(".env ignored", "\n.env\n" in gitignore)
    all_ok &= check(".env.example is placeholder only", "pk.your_public_token_here" in env_example and "sk-" not in env_example)

    # 5. Repository cleanliness / archived research
    all_ok &= check("historical vision artifact preserved", (root / "models" / "deployed" / "vision_price.pt").exists())
    all_ok &= check("legacy scripts still labelled historical", (root / "scripts" / "legacy" / "README.md").exists())

    # 6. High-risk claim scan
    high_risk = []
    for path in [root / "README.md", report_path, root / "scripts" / "smoke_api.py", root / "app" / "backend" / "main.py"]:
        txt = Path(path).read_text(encoding="utf-8")
        if re.search(r"POST /predict-image|/tile/|vision_only", txt):
            high_risk.append(str(path))
    all_ok &= check("no stale live-vision claims in primary docs", not high_risk, str(high_risk))

    print()
    print("ALL QA CHECKS PASS" if all_ok else "QA FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
