"""Smoke test for the FastAPI inference service."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import app  # noqa: E402

client = TestClient(app)

r = client.get("/health")
print("health:", r.status_code, r.json())

payload = {
    "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1910, "sqft_lot": 7600,
    "floors": 1.5, "sqft_above": 1560, "yr_built": 1975, "zipcode": 98065,
    "lat": 47.5724, "long": -122.2300, "sqft_living15": 1840, "sqft_lot15": 7620,
}
r = client.post("/predict", json=payload)
print("predict:", r.status_code, json.dumps(r.json(), indent=2))

r = client.get("/")
print("index:", r.status_code, "HTML" if "Satellite Property Valuation" in r.text else "?")
assert r.status_code == 200 and client.post("/predict", json=payload).status_code == 200

r = client.post("/predict-image", json={"pid": 1777500160})
body = r.json()
print("predict-image:", r.status_code, body.get("model_type"), body.get("source"))
assert r.status_code == 200 and body.get("model_type") == "vision_only"
print("SMOKE OK")