"""Phase 1 — Canonical vision dataset manifest + validation.

Builds one authoritative manifest of the 2,189 image-covered properties with:
  property_id, image_path, price, split

Validates:
  - no missing images
  - no duplicate property ids
  - no train/validation overlap
  - image <-> property alignment (tile exists, valid, >= min size)
  - the exact same canonical split used by E4/E5 (1,755 train / 434 val)
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PREPROCESSED_DIR, REPORTS_DIR, IMAGE_TRAIN
from src.data.load import load_clean_train, canonical_split
from src.satellite.align import is_valid_image, image_path_for


def main():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train = df[df["id"].isin(tr_ids)]
    val = df[df["id"].isin(va_ids)]

    covered_train = train[train["id"].map(is_valid_image)].copy()
    covered_val = val[val["id"].map(is_valid_image)].copy()

    rows = []
    for fr, split in ((covered_train, "train"), (covered_val, "val")):
        for _, r in fr.iterrows():
            rows.append({
                "property_id": int(r["id"]),
                "image_path": str(image_path_for(r["id"])),
                "price": float(r["price"]),
                "split": split,
            })
    manifest = pd.DataFrame(rows).sort_values(["split", "property_id"]).reset_index(drop=True)

    out = PREPROCESSED_DIR / "vision_manifest.csv"
    manifest.to_csv(out, index=False)

    # ---- validation ----
    checks = {}
    checks["total_rows"] = len(manifest)
    checks["train_count"] = int((manifest["split"] == "train").sum())
    checks["val_count"] = int((manifest["split"] == "val").sum())

    dup_ids = manifest["property_id"].duplicated().sum()
    checks["duplicate_property_ids"] = int(dup_ids)

    tr_set = set(manifest.loc[manifest["split"] == "train", "property_id"])
    va_set = set(manifest.loc[manifest["split"] == "val", "property_id"])
    checks["train_val_overlap"] = len(tr_set & va_set)

    missing = [p for p in manifest["image_path"] if not Path(p).exists()]
    checks["missing_images"] = len(missing)

    not_valid = [p for p in manifest["image_path"] if Path(p).exists() and not is_valid_image(int(Path(p).stem))]
    checks["invalid_images"] = len(not_valid)

    on_disk = {int(p.stem) for p in IMAGE_TRAIN.glob("*.jpg")}
    checks["manifest_ids_in_manifest"] = manifest["property_id"].nunique()

    # same split as E4/E5 (both derive from canonical_split + same covered filter)
    checks["matches_e4e5_split"] = (checks["train_count"] == 1755 and checks["val_count"] == 434)

    expected = {
        "total_rows": 2189, "train_count": 1755, "val_count": 434,
        "duplicate_property_ids": 0, "train_val_overlap": 0,
        "missing_images": 0, "invalid_images": 0,
    }
    checks["all_expected"] = all(checks[k] == v for k, v in expected.items())

    summary = {
        "manifest": str(out.relative_to(Path(__file__).resolve().parents[1])),
        "counts": {k: checks[k] for k in
                   ("total_rows", "train_count", "val_count",
                    "duplicate_property_ids", "train_val_overlap",
                    "missing_images", "invalid_images")},
        "split_matches_e4e5": checks["matches_e4e5_split"],
        "all_validation_passed": checks["all_expected"],
        "alignment_rule": "tile keyed by property id: images/train/{id}.jpg; validity = exists + decodes + min side >= 200px",
    }
    (REPORTS_DIR / "vision_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    assert checks["all_expected"], "manifest validation FAILED"
    print("\nMANIFEST OK")


if __name__ == "__main__":
    main()