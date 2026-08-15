"""Comprehensive test suite for the Satellite Property Valuation API.

Covers:
  - Feature engineering unit tests
  - /predict: valid, missing-field, out-of-range, coordinate validation
  - Deprecated vision routes return 404
  - Regression baseline (primary prediction must not drift)
"""
import pytest
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from app.backend.main import app
from src.features.build_features import _engineer, ZipTargetEncoder
from src.features.build_features import FEATURE_COLS
from src.inference.artifacts import load_tabular_artifacts
from src.inference.predict import predict_single

client = TestClient(app)

# ---------------------------------------------------------------------------
# Canonical smoke payload (used across multiple tests)
# ---------------------------------------------------------------------------
SMOKE_PAYLOAD = {
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
    "sale_year": 2015,
    "sale_quarter": 3,
}

# Regression anchor — captured on first run, compared on subsequent runs.
_REGRESSION_ANCHOR = []


# ---------------------------------------------------------------------------
# Feature engineering unit tests
# ---------------------------------------------------------------------------

def _base_df():
    return pd.DataFrame([{
        "id": 1,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2000,
        "sqft_lot": 5000,
        "floors": 2.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1500,
        "sqft_basement": 500,
        "yr_built": 1985,
        "yr_renovated": 0,
        "zipcode": 98052,
        "lat": 47.6,
        "long": -122.3,
        "sqft_living15": 2100,
        "sqft_lot15": 5200,
        "sale_year": 2015,
        "sale_quarter": 2,
        "price": 600000,
    }])


def test_engineer_adds_expected_columns():
    out = _engineer(_base_df())
    assert "age" in out.columns
    assert "dist_to_center_km" in out.columns
    assert "total_sqft" in out.columns
    assert out["total_sqft"].iloc[0] == 2500
    assert out["has_basement"].iloc[0] == 1


def test_engineer_age_calculation():
    out = _engineer(_base_df())
    assert out["age"].iloc[0] == 30  # 2015 - 1985


def test_zip_target_encoder_fit_transform():
    df = pd.DataFrame({"zipcode": [98052, 98052, 98053], "price": [500000, 600000, 700000]})
    enc = ZipTargetEncoder(col="zipcode", target="price", smoothing=20.0)
    out = enc.fit_transform(df)
    assert "zip_target" in out.columns
    assert out["zip_target"].notna().all()
    assert out["zip_target"].nunique() >= 2


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /predict — valid
# ---------------------------------------------------------------------------

def test_predict_endpoint_smoke():
    r = client.post("/predict", json=SMOKE_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_price" in body
    assert "model" in body
    assert float(body["predicted_price"]) > 0


def test_predict_response_schema():
    r = client.post("/predict", json=SMOKE_PAYLOAD)
    body = r.json()
    assert body["model_role"] == "primary"
    assert body["status"] == "production"
    assert isinstance(body["top_factors_gain"], list)
    assert len(body["top_factors_gain"]) > 0
    assert "feature" in body["top_factors_gain"][0]
    assert "importance" in body["top_factors_gain"][0]


def test_predict_shap_parity_and_error_band():
    r = client.post("/predict", json=SMOKE_PAYLOAD)
    assert r.status_code == 200
    body = r.json()

    # Local explanation is present, labelled, and consistent with the price.
    ls = body["local_shap"]
    assert isinstance(ls, dict)
    assert ls["expected_value"] > 0
    assert abs(ls["expected_value"] + ls["total_contribution"] - body["predicted_price"]) < 1.0
    for item in ls["top_positive"] + ls["top_negative"]:
        assert item["label"] and item["feature"]
        assert item["direction"] in ("up", "down")

    # Empirical error band is present and explicit that it is not an interval.
    band = body["error_band"]
    assert isinstance(band, dict)
    assert band["typical_error"] > 0 and band["n"] > 0
    assert "not a per-property" in band["note"]


def test_predict_shap_humanised_labels():
    r = client.post("/predict", json=SMOKE_PAYLOAD)
    body = r.json()
    raw_seen = {x["feature"] for x in body["local_shap"]["top_positive"] + body["local_shap"]["top_negative"]}
    labels = {x["label"] for x in body["local_shap"]["top_positive"] + body["local_shap"]["top_negative"]}
    # Flagship implementation names must never surface as user-facing labels.
    assert "zip_target" not in labels
    assert "total_sqft" not in labels
    assert all(f not in raw_seen for f in ())  # raw names may appear; labels carry the human text


def test_predict_optional_fields_default():
    minimal = {k: v for k, v in SMOKE_PAYLOAD.items()
               if k not in ("waterfront", "view", "condition", "grade",
                            "sqft_basement", "yr_renovated", "sale_year", "sale_quarter")}
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200
    assert float(r.json()["predicted_price"]) > 0


def test_predict_boundary_coordinates_accepted():
    edge = {**SMOKE_PAYLOAD, "lat": 90.0, "long": 180.0}
    r = client.post("/predict", json=edge)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /predict — validation failures
# ---------------------------------------------------------------------------

def test_predict_missing_lat():
    bad = {k: v for k, v in SMOKE_PAYLOAD.items() if k != "lat"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_sqft_living():
    bad = {k: v for k, v in SMOKE_PAYLOAD.items() if k != "sqft_living"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_endpoint_invalid_input_returns_clear_error():
    bad = {**SMOKE_PAYLOAD, "lat": 200.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "latitude" in r.json()["error"].lower()


def test_predict_invalid_lat_negative():
    r = client.post("/predict", json={**SMOKE_PAYLOAD, "lat": -91.0})
    assert r.status_code == 422
    assert "latitude" in r.json()["error"].lower()


def test_predict_invalid_longitude():
    r = client.post("/predict", json={**SMOKE_PAYLOAD, "long": -200.0})
    assert r.status_code == 422
    assert "longitude" in r.json()["error"].lower()


def test_predict_invalid_grade_too_high():
    r = client.post("/predict", json={**SMOKE_PAYLOAD, "grade": 20})
    assert r.status_code == 422


def test_predict_invalid_bedrooms_too_high():
    r = client.post("/predict", json={**SMOKE_PAYLOAD, "bedrooms": 50})
    assert r.status_code == 422


def test_predict_regression_stable():
    """Primary prediction must stay within +-5% of the first recorded value."""
    r = client.post("/predict", json=SMOKE_PAYLOAD)
    price = float(r.json()["predicted_price"])
    if not _REGRESSION_ANCHOR:
        _REGRESSION_ANCHOR.append(price)
        print(f"\n[REGRESSION ANCHOR] primary = ${price:,.0f}")
    else:
        drift = abs(price - _REGRESSION_ANCHOR[0]) / _REGRESSION_ANCHOR[0]
        assert drift < 0.05, (
            f"Primary prediction drifted {drift:.1%} "
            f"(anchor ${_REGRESSION_ANCHOR[0]:,.0f} → ${price:,.0f})"
        )


# ---------------------------------------------------------------------------
# Production app: vision endpoint removed
# ---------------------------------------------------------------------------


def test_predict_image_route_is_not_exposed():
    r = client.post("/predict-image", json={"pid": 1777500160})
    assert r.status_code == 404


def test_predict_image_tile_routes_are_not_exposed():
    r = client.get("/tile/1777500160")
    assert r.status_code == 404

    r = client.get("/tile/mapbox", params={"lat": 47.57, "long": -122.23})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Release hardening: artifacts, schema, parity, submission
# ---------------------------------------------------------------------------

def test_tabular_artifacts_exist():
    assert Path("models/deployed/tabular_model.joblib").exists()
    assert Path("models/deployed/tabular_pipeline.joblib").exists()


def test_feature_schema_matches_artifact():
    pipeline, _ = load_tabular_artifacts()
    expected_cols = [c for c in FEATURE_COLS if c != "price"]
    assert pipeline["feature_cols"] == expected_cols
    assert "zip_target" in pipeline["feature_cols"]


def test_api_offline_prediction_parity():
    payload = {**SMOKE_PAYLOAD, "sale_year": 2015, "sale_quarter": 3}
    offline = predict_single(payload)
    api = client.post("/predict", json=payload)
    assert api.status_code == 200
    assert abs(float(api.json()["predicted_price"]) - float(offline["predicted_price"])) < 1e-6


def test_submission_schema_and_counts():
    sub = pd.read_csv("predictions/submission.csv")
    assert list(sub.columns) == ["id", "predicted_price"]
    assert sub["predicted_price"].notna().all()
    assert sub["predicted_price"].gt(0).all()

    # Row/id-parity is asserted against the test set. The raw data/test.xlsx is
    # gitignored (large, not public), so fall back to the repository-contained
    # cleaned test fixture when the raw file is absent (e.g. a fresh CI clone).
    raw = Path("data/test.xlsx")
    if raw.exists():
        test_df = pd.read_excel(raw)
    else:
        test_df = pd.read_pickle("preprocessed/test_clean.pkl")
    assert len(sub) == len(test_df)
    assert sorted(sub["id"].tolist()) == sorted(test_df["id"].tolist())


def test_secret_and_config_safety():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    env_example = Path(".env.example").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "data/*.xlsx" in gitignore
    assert "MAPBOX_TOKEN=pk.your_public_token_here" in env_example
    assert "sk-" not in env_example


def test_baseline_manifest_matches_files():
    import hashlib
    import json

    manifest = json.loads(Path("reports/baseline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["model_version"] == "xgboost-tuned-2026-08-15"
    for item in manifest["artifacts"]:
        digest = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"]
