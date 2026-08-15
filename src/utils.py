"""Helper utilities shared across the package."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def json_dump(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def json_load(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dataframe_to_nested(df: pd.DataFrame) -> dict:
    """Convert a DataFrame into a JSON-friendly nested dict."""
    return df.to_dict(orient="split")


def haversine_km(lat_a, lon_a, lat_b, lon_b):
    """Great-circle distance in km between two coordinate pairs."""
    lat_a, lon_a, lat_b, lon_b = map(np.radians, (lat_a, lon_a, lat_b, lon_b))
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def stable_str_hash(value) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]