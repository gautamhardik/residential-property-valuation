"""FastAPI inference service for the production tabular model.

POST /predict -> primary: tuned tabular XGBoost {predicted_price, model, ...}
GET  /health -> health status
GET  /       -> serves the portfolio UI

The image-model experiments remain documented in the project research notes, but the
production application exposes only the tabular valuation path.

Run:  uvicorn app.backend.main:app --reload  (from project root)
"""
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
   sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.inference.predict import predict_single
from src.inference.artifacts import preload_tabular_artifacts
from src.inference.explain import local_summary, error_band_for

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the production tabular artifacts once at startup."""
    logger.info("Loading XGBoost tabular artifacts...")
    preload_tabular_artifacts()
    logger.info("XGBoost artifacts loaded.")

    yield  # Application runs here

    # Shutdown: nothing to clean up for the in-memory tabular model.


app = FastAPI(
    title="Residential Property Valuation API",
    version="1.0.0",
    lifespan=lifespan,
)


class PropertyInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int | None = 0
    view: int | None = 0
    condition: int | None = 3
    grade: int | None = 7
    sqft_above: int
    sqft_basement: int | None = 0
    yr_built: int
    yr_renovated: int | None = 0
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    sale_year: int | None = 2015
    sale_quarter: int | None = 3


PRIMARY_FIELD_RULES = {
   "bedrooms": ("bedrooms", 1, 20),
   "bathrooms": ("bathrooms", 0.5, 20),
   "sqft_living": ("sqft_living", 200, 10000),
   "sqft_lot": ("sqft_lot", 200, 500000),
   "floors": ("floors", 1, 10),
   "waterfront": ("waterfront", 0, 1),
   "view": ("view", 0, 4),
   "condition": ("condition", 1, 5),
   "grade": ("grade", 1, 13),
   "sqft_above": ("sqft_above", 100, 12000),
   "sqft_basement": ("sqft_basement", 0, 8000),
   "yr_built": ("year built", 1800, 2100),
   "yr_renovated": ("year renovated", 0, 2100),
   "zipcode": ("zipcode", 10000, 99999),
   "lat": ("latitude", -90.0, 90.0),
   "long": ("longitude", -180.0, 180.0),
   "sqft_living15": ("sqft_living15", 200, 10000),
   "sqft_lot15": ("sqft_lot15", 200, 500000),
}


def _err(message: str):
   return JSONResponse(status_code=422, content={"error": message})


def _validate_primary_payload(payload: dict):
   for field, value in payload.items():
       if field not in PRIMARY_FIELD_RULES:
           continue
       label, lower, upper = PRIMARY_FIELD_RULES[field]
       if value is None:
           raise ValueError(f"Please enter a valid value for {label}.")
       if isinstance(value, (int, float)) and not (lower <= float(value) <= upper):
           raise ValueError(f"Please enter a valid value for {label}.")


@app.get("/health")
def health():
   return {"status": "ok", "model": "XGBoost tuned tabular"}


@app.post("/predict")
def predict(payload: PropertyInput):
    try:
        _validate_primary_payload(payload.model_dump())
        result = predict_single(payload.model_dump())
        result["model_role"] = "primary"
        result["status"] = "production"
        result["local_shap"] = local_summary(payload.model_dump())
        result["error_band"] = error_band_for(result["predicted_price"])
        return result
    except ValueError as exc:
        return _err(str(exc))
    except Exception:  # noqa: BLE001
        return _err("Unable to reach the valuation service. Please try again.")


FRONTEND = ROOT / "app" / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
   if not FRONTEND.exists():
        return "<h1>Residential Property Valuation API</h1><p>POST /predict</p>"
   return FRONTEND.read_text(encoding="utf-8")