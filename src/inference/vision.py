"""Vision inference — persistence of the torch.jit CNN price model.

Loads models/deployed/vision_price.pt (TorchScript, CPU-compatible, trained
ResNet18 layer4 partial fine-tune + regression head). It bakes in the train-time
feature/target standardization so the returned value is a real USD price.

Exposed:
  predict_from_image(pil)              -> price from an in-memory PIL image
  predict_pid(pid)                     -> price for a training property id (tile)
  predict_from_latlon(lat, lon, token) -> price from a live Mapbox tile

This is a SECONDARY research service, NOT the production champion. The champion
remains the tuned tabular XGBoost (src.inference.predict.predict_single).
"""
import os
import io
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import APP_MODELS_DIR, IMAGE_TRAIN

VISION_MODEL_FILE = APP_MODELS_DIR / "vision_price.pt"

_MODEL_MEAN = [0.485, 0.456, 0.406]
_MODEL_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MODEL_MEAN, std=_MODEL_STD),
])

_model = None


def _load():
    global _model
    if _model is None:
        if not VISION_MODEL_FILE.exists():
            raise FileNotFoundError(
                f"vision model artifact missing: {VISION_MODEL_FILE} "
                "(run scripts/phase8_serialize.py to build it)")
        _model = torch.jit.load(str(VISION_MODEL_FILE), map_location="cpu")
        _model.eval()
    return _model


def preload_vision_model() -> None:
    """Load the ResNet18 TorchScript model into the module cache.

    Call this once at application startup.  Subsequent inference calls
    will reuse the cached model without touching the filesystem.
    Silently skips if the model artifact does not exist (vision is an
    optional research service).
    """
    try:
        _load()
    except FileNotFoundError:
        pass  # Vision artifact optional — warning logged by caller


def predict_from_image(pil_image: Image.Image) -> float:
    img = pil_image.convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        pred = _load()(x)
    return float(np.asarray(pred).reshape(-1)[0])


def pid_tile_path(pid) -> Path:
    path = IMAGE_TRAIN / f"{pid}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"no satellite tile for property {pid}: {path}")
    return path


def predict_pid(pid) -> float:
    with Image.open(pid_tile_path(pid)).convert("RGB") as im:
        return predict_from_image(im)


def predict_from_latlon(lat: float, lon: float, token: str) -> float:
    """Live Mapbox tile for (lat, lon) -> price. Secondary vision path."""
    from src.data.fetch_images import mapbox_url, _is_valid_image
    import requests

    url = mapbox_url(lat, lon, token)
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200 or not _is_valid_image(resp.content):
        raise RuntimeError(f"Mapbox tile fetch failed (HTTP {resp.status_code})")
    img = Image.open(io.BytesIO(resp.content))
    return predict_from_image(img)