"""Historical research CLI for the vision experiment model (archived, not served).

Loads the persisted torch.jit artifact (models/deployed/vision_price.pt) and
predicts a property price from its satellite tile. The jit model bakes in the
train-time feature/target standardization, so it outputs real USD prices.

This is NOT the production champion - that remains the tuned tabular XGBoost.
Use it to inspect the vision experiment model only.

Example:
  python scripts/predict_image.py --pid 1777500160            # auto-resolve tile path
  python scripts/predict_image.py --image path/to/tile.jpg
  python scripts/predict_image.py --pid 1777500160 --tabular  # side-by-side with champion
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.inference import vision


def tabular_price(pid):
    from src.data.load import load_clean_train
    from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
    from src.inference.artifacts import load_tabular_artifacts
    df = load_clean_train()
    row = df[df["id"] == int(pid)]
    if row.empty:
        return None
    pipeline, model = load_tabular_artifacts()
    X = _engineer(row.iloc[[0]])
    X["zip_target"] = X["zipcode"].map(pipeline["zip_enc"]).fillna(pipeline["global_mean"])
    cols = [c for c in pipeline["feature_cols"] if c != "price"]
    return float(model.predict(X[cols])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None, help="property id (resolves tile path)")
    ap.add_argument("--image", default=None, help="path to a satellite tile image")
    ap.add_argument("--tabular", action="store_true", help="also print the tabular champion prediction")
    args = ap.parse_args()

    if not args.pid and not args.image:
        ap.error("provide --pid or --image")

    if args.pid:
        price = vision.predict_pid(args.pid)
        src = f"tile for property {args.pid}"
    else:
        with Image.open(args.image) as im:
            price = vision.predict_from_image(im)
        src = args.image

    print("VISION EXPERIMENT (secondary model - NOT the production champion):")
    print(f"  source: {src}")
    print(f"  estimated price: ${price:,.0f}")
    print(f"  model: ResNet18 partial fine-tune (layer4 + regression head), model_type=vision_only")

    if args.tabular:
        if not args.pid:
            print("  --tabular requires --pid")
            return
        tab = tabular_price(args.pid)
        print("TABULAR CHAMPION (production model):")
        print(f"  estimated price: ${tab:,.0f}" if tab is not None
              else "  no clean-training row for this pid")


if __name__ == "__main__":
    main()