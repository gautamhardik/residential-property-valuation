"""Image/property alignment helper.

Every tile is keyed by property id:  images/train/{id}.jpg
This module builds and validates the manifest and reports coverage honestly.
"""
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import IMAGE_TRAIN, IMAGE_EXT


def image_path_for(pid) -> Path:
    return IMAGE_TRAIN / f"{pid}{IMAGE_EXT}"


def is_valid_image(pid, min_size=200) -> bool:
    p = image_path_for(pid)
    if not p.exists():
        return False
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w, h = im.size
        return min(w, h) >= min_size
    except Exception:
        return False


def build_manifest(ids) -> pd.DataFrame:
    """Return a per-property manifest of image availability and validity."""
    rows = []
    for pid in ids:
        path = image_path_for(pid)
        rows.append({
            "id": pid,
            "has_file": path.exists(),
            "valid": is_valid_image(pid) if path.exists() else False,
        })
    return pd.DataFrame(rows)


def coverage_summary(manifest: pd.DataFrame) -> dict:
    n = len(manifest)
    available = int(manifest["has_file"].sum())
    valid = int(manifest["valid"].sum())
    return {
        "properties": n,
        "images_available": available,
        "images_valid": valid,
        "images_missing": n - available,
        "images_invalid": available - valid,
        "coverage_pct": round(valid / n * 100, 2),
    }


def id_image_dataset(ids):
    """Yield (pid, RGB PIL image) for each valid id with an image. Id-keyed."""
    for pid in ids:
        if is_valid_image(pid):
            with Image.open(image_path_for(pid)).convert("RGB") as im:
                yield pid, im.copy()