"""Lightweight CLI inference for the production valuation model.

Example:
 python -m app.cli --bedrooms 3 --bathrooms 2.0 --sqft_living 1910 \
     --sqft_lot 7600 --floors 1.5 --sqft_above 1560 --yr_built 1975 \
     --zipcode 98065 --lat 47.5724 --long -122.2300 \
     --sqft_living15 1840 --sqft_lot15 7620

The vision branch remains archived as research documentation and is intentionally not exposed in the
live app or CLI.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
   sys.path.insert(0, str(ROOT))

from src.inference.predict import predict_single, REQUIRED_FIELDS

FIELDS = REQUIRED_FIELDS + ["sale_year", "sale_quarter"]
TYPES = {
   "bedrooms": int, "bathrooms": float, "sqft_living": int, "sqft_lot": int,
   "floors": float, "waterfront": int, "view": int, "condition": int,
   "grade": int, "sqft_above": int, "sqft_basement": int, "yr_built": int,
   "yr_renovated": int, "zipcode": int, "lat": float, "long": float,
   "sqft_living15": int, "sqft_lot15": int, "sale_year": int, "sale_quarter": int,
}


def main():
   ap = argparse.ArgumentParser(description="Predict home sale price using the production tabular XGBoost model.")
   for f in FIELDS:
       ap.add_argument(f"--{f}", type=TYPES[f], default=None)
   args = ap.parse_args()

   features = {k: getattr(args, k) for k in FIELDS if getattr(args, k) is not None}
   result = predict_single(features)
   print("Production model: tuned XGBoost (tabular + geospatial features)")
   print(f"Estimated price: ${result['predicted_price']:,.0f}")
   print(f"Model: {result['model']}")
   print(f"Top factors ({result['importance_type']}):")
   for f in result["top_factors_gain"][:6]:
       print(f"  {f['feature']}: {f['importance']:.3f}")


if __name__ == "__main__":
   main()