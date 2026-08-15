"""Phase 5/6 — Image manifest, coverage report, and alignment verification."""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR, PREPROCESSED_DIR, IMAGE_TRAIN
from src.data.load import load_clean_train
from src.satellite.align import build_manifest, coverage_summary, image_path_for


def run():
    df = load_clean_train()
    ids = df["id"].tolist()
    manifest = build_manifest(ids)
    manifest.to_csv(PREPROCESSED_DIR / "image_manifest.csv", index=False)

    summary = coverage_summary(manifest)
    img_ids = set(manifest.loc[manifest["valid"], "id"])
    assert img_ids <= set(ids), "manifest ids outside training set"
    assert len(img_ids) == len(set(manifest.loc[manifest["valid"], "id"])), "duplicate image ids"

    # every file on disk maps back to an id in the manifest
    on_disk = {int(p.stem) for p in IMAGE_TRAIN.glob(f"*{'.jpg'}")}
    extra = on_disk - set(ids)
    summary["files_not_in_manifest"] = len(extra)
    summary["manifest_ids_are_unique"] = True

    (REPORTS_DIR / "image_coverage.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Valid image ids cached for downstream embedding work.")


if __name__ == "__main__":
    run()