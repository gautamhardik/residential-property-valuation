"""
fetch_all_tiles.py
==================
Batch-download satellite tiles for ALL train + test properties.

Usage:
    python scripts/fetch_all_tiles.py                   # both splits
    python scripts/fetch_all_tiles.py --split train
    python scripts/fetch_all_tiles.py --split test
    python scripts/fetch_all_tiles.py --dry-run         # count only, no downloads
    python scripts/fetch_all_tiles.py --workers 10      # concurrency (default 20)

Resume:  Already-downloaded tiles are skipped automatically.
Token:   Reads MAPBOX_TOKEN from .env or the environment.
"""
import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
import io

from src.config import (
    DATA_TRAIN, DATA_TEST,
    IMAGE_TRAIN, IMAGE_TEST,
    IMAGE_EXT, MAPBOX_STYLE, IMAGE_SIZE, ZOOM_LEVEL,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_tiles")


# ---- Config ------------------------------------------------------------------

MAX_RETRIES      = 4
BACKOFF_BASE     = 1.5        # seconds; sleep = BACKOFF_BASE ** attempt
RATE_LIMIT_SLEEP = 30         # seconds to pause when Mapbox returns 429
TIMEOUT          = 25         # seconds per HTTP request
MIN_IMAGE_BYTES  = 500        # reject tiles smaller than this


# ---- Helpers -----------------------------------------------------------------

def mapbox_url(lat: float, lon: float, token: str) -> str:
    return (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAPBOX_STYLE}/static/"
        f"{lon},{lat},{ZOOM_LEVEL}/{IMAGE_SIZE}x{IMAGE_SIZE}"
        f"?access_token={token}"
    )


def is_valid_image(data: bytes) -> bool:
    if len(data) < MIN_IMAGE_BYTES:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_todo(split: str) -> list:
    """Return [{pid, lat, lon, out_path}] for tiles not yet on disk."""
    df      = pd.read_excel(DATA_TRAIN if split == "train" else DATA_TEST)
    out_dir = IMAGE_TRAIN if split == "train" else IMAGE_TEST
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {int(p.stem) for p in out_dir.glob(f"*{IMAGE_EXT}")}
    todo = []
    for _, row in df.iterrows():
        pid = int(row["id"])
        if pid not in existing:
            todo.append({
                "pid":      pid,
                "lat":      float(row["lat"]),
                "lon":      float(row["long"]),
                "out_path": out_dir / f"{pid}{IMAGE_EXT}",
            })
    return todo


# ---- Progress ----------------------------------------------------------------

class Stats:
    def __init__(self, total: int):
        self.total      = total
        self.downloaded = 0
        self.failed     = 0
        self._lock      = asyncio.Lock()
        self._start     = time.monotonic()

    async def inc(self, key: str):
        async with self._lock:
            setattr(self, key, getattr(self, key) + 1)

    def eta_str(self) -> str:
        done = self.downloaded + self.failed
        if done == 0:
            return "calculating..."
        rate = done / max(0.1, time.monotonic() - self._start)
        remaining = self.total - done
        m, s = divmod(int(remaining / rate), 60)
        return f"{m}m {s:02d}s"

    def line(self) -> str:
        done    = self.downloaded + self.failed
        pct     = 100 * done / self.total if self.total else 0
        filled  = int(30 * pct / 100)
        bar     = "#" * filled + "-" * (30 - filled)
        return (
            f"\r[{bar}] {done}/{self.total} ({pct:.1f}%)  "
            f"ok={self.downloaded}  fail={self.failed}  ETA {self.eta_str()}  "
        )


# ---- Async downloader --------------------------------------------------------

async def download_one(
    session: aiohttp.ClientSession,
    item: dict,
    token: str,
    stats: Stats,
    sem: asyncio.Semaphore,
    rate_gate: asyncio.Event,
) -> None:
    url = mapbox_url(item["lat"], item["lon"], token)

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_gate.wait()
        async with sem:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)
                ) as resp:
                    if resp.status == 429:
                        log.warning("429 rate-limit — sleeping %ss", RATE_LIMIT_SLEEP)
                        rate_gate.clear()
                        await asyncio.sleep(RATE_LIMIT_SLEEP)
                        rate_gate.set()
                        continue
                    if resp.status != 200:
                        if attempt == MAX_RETRIES:
                            await stats.inc("failed")
                            return
                        await asyncio.sleep(BACKOFF_BASE ** attempt)
                        continue

                    data = await resp.read()
                    if not is_valid_image(data):
                        if attempt == MAX_RETRIES:
                            await stats.inc("failed")
                            return
                        await asyncio.sleep(BACKOFF_BASE ** attempt)
                        continue

                    item["out_path"].write_bytes(data)
                    await stats.inc("downloaded")
                    return

            except (asyncio.TimeoutError, aiohttp.ClientError):
                if attempt == MAX_RETRIES:
                    await stats.inc("failed")
                    return
                await asyncio.sleep(BACKOFF_BASE ** attempt)


async def run_fetch(todo: list, token: str, workers: int) -> Stats:
    stats     = Stats(total=len(todo))
    sem       = asyncio.Semaphore(workers)
    rate_gate = asyncio.Event()
    rate_gate.set()

    connector = aiohttp.TCPConnector(limit=workers + 5)

    async with aiohttp.ClientSession(connector=connector) as session:

        async def progress_printer():
            while True:
                print(stats.line(), end="", flush=True)
                if stats.downloaded + stats.failed >= stats.total:
                    break
                await asyncio.sleep(0.5)
            print()

        tasks = [
            download_one(session, item, token, stats, sem, rate_gate)
            for item in todo
        ]
        await asyncio.gather(progress_printer(), *tasks)

    return stats


# ---- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch-fetch Mapbox satellite tiles")
    ap.add_argument("--split",   choices=["train", "test", "both"], default="both")
    ap.add_argument("--workers", type=int, default=20,
                    help="Concurrent download workers (default 20)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count missing tiles only, do not download")
    args = ap.parse_args()

    token = os.getenv("MAPBOX_TOKEN", "")
    if not token and not args.dry_run:
        log.error("MAPBOX_TOKEN not set — add it to .env")
        sys.exit(1)

    splits   = ["train", "test"] if args.split == "both" else [args.split]
    all_todo = []
    for split in splits:
        todo = build_todo(split)
        log.info("%-5s  %d missing tiles", split, len(todo))
        all_todo.extend(todo)

    log.info("Total to fetch: %d  (Mapbox free tier: 50,000/month)", len(all_todo))
    log.info("All %d requests are within the free tier: %s",
             len(all_todo), len(all_todo) <= 50_000)

    if args.dry_run or not all_todo:
        if not all_todo:
            log.info("Nothing to download — all tiles already present.")
        return

    log.info("Workers: %d | Starting... (Ctrl-C pauses; re-run resumes automatically)",
             args.workers)
    print()

    t0    = time.monotonic()
    stats = asyncio.run(run_fetch(all_todo, token, args.workers))
    elapsed = time.monotonic() - t0

    print()
    log.info("=" * 50)
    log.info("Finished in %.0f s", elapsed)
    log.info("  Downloaded : %d", stats.downloaded)
    log.info("  Failed     : %d", stats.failed)
    log.info("=" * 50)

    # Final coverage report
    print()
    for split in splits:
        df      = pd.read_excel(DATA_TRAIN if split == "train" else DATA_TEST)
        out_dir = IMAGE_TRAIN if split == "train" else IMAGE_TEST
        have    = len(list(out_dir.glob("*.jpg")))
        total   = len(df)
        log.info("%s coverage: %d/%d (%.1f%%)", split, have, total,
                 100 * have / total)

    if stats.failed > 0:
        log.warning("%d tiles failed. Re-run to retry them.", stats.failed)


if __name__ == "__main__":
    main()
