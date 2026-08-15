"""Visual embedding extraction with disk cache.

- Frozen pretrained encoders:
  - resnet18 (512-d) as the primary visual baseline,
  - resnet50 (2048-d) as an optional stronger comparison,
  - dinov2_vits14 (384-d) as a stronger self-supervised ViT baseline.
- Id-keyed input: images are joined to property ids explictly.
- Deterministic inference in .eval() / torch.no_grad().
- Embeddings cached to preprocessed/embeddings_{encoder}.npz so the CNN runs once.
"""
import numpy as np
import torch
from torchvision import models, transforms
from pathlib import Path

from src.config import PREPROCESSED_DIR, EMBEDDING_DIM, ENCODER_NAME

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def load_encoder(name: str = ENCODER_NAME):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # drop FC -> pooled features
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # drop FC -> pooled features
    elif name == "dinov2_vits14":
        # Uses the official Facebook DINOv2 checkpoint via torch.hub cache.
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    else:
        raise ValueError(f"Unsupported encoder {name}")
    model.eval()
    return model


def extract(pid_image_pairs, encoder_name: str = ENCODER_NAME,
            batch_size: int = 32, device: str = "cpu"):
    """Return (ids list, embeddings ndarray) for the given (pid, PIL image) pairs."""
    model = load_encoder(encoder_name).to(device)
    ids = []
    tensors = []
    chunks = []

    def flush_batch():
        if not tensors:
            return
        batch = torch.stack(tensors).to(device)
        feats = model(batch)
        if feats.ndim > 2:
            feats = feats.view(batch.size(0), -1)
        chunks.append(feats.cpu().numpy().astype(np.float32, copy=False))
        tensors.clear()

    for pid, im in pid_image_pairs:
        ids.append(pid)
        tensors.append(TRANSFORM(im))
        if len(tensors) >= batch_size:
            with torch.no_grad():
                flush_batch()

    with torch.no_grad():
        flush_batch()

    if not chunks:
        emb = np.empty((0, EMBEDDING_DIM[encoder_name]), dtype=np.float32)
    else:
        emb = np.vstack(chunks).astype(np.float32, copy=False)

    if emb.shape[1] != EMBEDDING_DIM[encoder_name]:
        raise ValueError(
            f"Embedding width mismatch for {encoder_name}: "
            f"expected {EMBEDDING_DIM[encoder_name]}, got {emb.shape[1]}"
        )
    return ids, emb


def cache_path(encoder_name: str) -> Path:
    return PREPROCESSED_DIR / f"embeddings_{encoder_name}.npz"


def save_cache(ids, emb, encoder_name: str):
    path = cache_path(encoder_name)
    np.savez(path, ids=np.asarray(ids, dtype=np.int64), embeddings=emb)


def load_cache(encoder_name: str):
    path = cache_path(encoder_name)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["ids"].tolist(), data["embeddings"]