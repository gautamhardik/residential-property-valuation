"""Smoke test for the production FastAPI service.

Verifies the current live path only:
  GET  /health
  GET  /
  POST /predict
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import app  # noqa: E402

client = TestClient(app)

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

health = client.get("/health")
print("health:", health.status_code, health.json())
assert health.status_code == 200

index = client.get("/")
print("index:", index.status_code, "HTML" if "<!DOCTYPE html>" in index.text else "?")
assert index.status_code == 200

predict = client.post("/predict", json=payload)
body = predict.json()
print("predict:", predict.status_code, json.dumps(body, indent=2))
assert predict.status_code == 200
assert "predicted_price" in body
assert float(body["predicted_price"]) > 0

print("SMOKE OK")
