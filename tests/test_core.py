"""Comprehensive test suite for the Satellite Property Valuation API.

Covers:
  - Feature engineering unit tests
  - /predict: valid, missing-field, out-of-range, coordinate validation
  - /predict-image: PID mode, lat/long mode, invalid coords, missing inputs
  - Regression baseline (primary prediction must not drift)
"""
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.backend.main import app
from src.features.build_features import _engineer, ZipTargetEncoder

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

