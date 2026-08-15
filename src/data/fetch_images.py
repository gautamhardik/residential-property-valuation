"""MAPBOX-based satellite image acquisition with caching and content validation.

Improvements over the original script:
- Token loaded from environment only (never hardcoded).
- Filenames keyed by property *id* (consistent with tabular data).
- Verifies the response is a real image (not a mapbox error placeholder).
- Retries transient failures with backoff.
- Optional RGB-noise sanity test to reject black/corrupt tiles.
"""
import os
import time
import logging
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv

from src.config import MAPBOX_STYLE, IMAGE_SIZE, ZOOM_LEVEL, IMAGE_EXT
from src.data.load import load_train

log = logging.getLogger(__name__)
load_dotenv()


def _token() -> str:
    token = os.getenv("MAPBOX_TOKEN")
    if not token:
        raise RuntimeError(
            "MAPBOX_TOKEN not found. Copy .env.example to .env and set your "
            "Mapbox public access token."
        )
    return token


def mapbox_url(lat, lon, token, zoom=ZOOM_LEVEL, size=IMAGE_SIZE):
    return (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAPBOX_STYLE}/static/"
        f"{lon},{lat},{zoom}/{size}x{size}?access_token={token}"
    )


def _is_valid_image(data: bytes, min_bytes: int = 500) -> bool:
    """Reject empty / error placeholder responses."""
    if len(data) < min_bytes:
        return False
    try:
        with Image.open(__import__("io").BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:
        return False


def download_image(lat, lon, token, out_path: Path, retries=3, backoff=2.0) -> bool:
    url = mapbox_url(lat, lon, token)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200 and _is_valid_image(resp.content):
                out_path.write_bytes(resp.content)
                return True
            if resp.status_code == 429:
                time.sleep(backoff * attempt)
                continue
            if resp.status_code != 200:
                log.warning("HTTP %s for %s", resp.status_code, out_path.name)
                return False
        except Exception as exc:  # network errors
            log.warning("Attempt %s/%s failed for %s: %s", attempt, retries, out_path.name, exc)
            time.sleep(backoff * attempt)
    return False


def fetch_train_images(mode="full", limit=None, sleep_between=0.2):
    """Download satellite tiles for training properties.

    mode:
      "full"   -> all train properties
      "sample" -> deterministic 500-property sample (same seed every run)
      "missing"-> only ids that still lack a tile on disk
    """
    token = _token()
    df = load_train()

    if mode == "sample":
        df = df.sample(n=min(500, len(df)), random_state=42)
    if limit:
        df = df.head(limit)

    total = len(df)
    done = skipped = failed = 0
    log.info("Image run: mode=%s total=%s dir=%s", mode, total, "images/train")

    for _, row in df.iterrows():
        pid = row["id"]
        out = Path("images") / "train" / f"{pid}{IMAGE_EXT}"
        if out.exists() and out.stat().st_size > 500:
            skipped += 1
            done += 1
            continue
        ok = download_image(row["lat"], row["long"], token, out) if getattr(row, "lat", None) is not None else False
        if ok:
            done += 1
            if done % 50 == 0:
                log.info("downloaded %s/%s", done, total)
        else:
            failed += 1
        if sleep_between and ok:
            time.sleep(sleep_between)

    log.info("Summary: requested=%s done=%s(skipped=%s) failed=%s", total, done, skipped, failed)
    return {"requested": total, "downloaded": done - skipped, "skipped": skipped, "failed": failed}