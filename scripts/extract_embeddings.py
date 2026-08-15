"""Phase 7 — Extract visual embeddings for all properties that have valid imagery.

Usage:  python scripts/extract_embeddings.py [--encoder resnet18|resnet50|dinov2_vits14]
Caches to preprocessed/embeddings_{encoder}.npz.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PREPROCESSED_DIR
from src.satellite.align import id_image_dataset, is_valid_image
from src.satellite.embeddings import extract, save_cache, load_cache, cache_path
from src.config import EMBEDDING_DIM
from src.data.load import load_clean_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet18", choices=sorted(EMBEDDING_DIM.keys()))
    ap.add_argument("--force", action="store_true", help="Recompute cache even if it exists")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cached = load_cache(args.encoder)
    if cached is not None and not args.force:
        ids, emb = cached
        print(f"CACHE HIT: {len(ids)} embeddings for '{args.encoder}'");
        return

    df = load_clean_train()
    image_ids = [pid for pid in df["id"] if is_valid_image(pid)]
    print(f"Extracting '{args.encoder}' on {len(image_ids)} images ...")
    ids, emb = extract(
        id_image_dataset(image_ids),
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        device=args.device,
    )
    assert len(ids) == len(image_ids) == emb.shape[0], "embedding alignment failed"
    save_cache(ids, emb, args.encoder)
    print(f"Saved {emb.shape[0]}x{emb.shape[1]} embeddings -> {cache_path(args.encoder).name}")


if __name__ == "__main__":
    main()