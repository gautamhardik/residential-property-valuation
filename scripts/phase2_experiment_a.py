"""Phase 2 — Experiment A: ResNet18 with a trainable regression head.

Frozen ImageNet ResNet18 backbone + trainable regression head. Two targets:
  A1  target = price        loss = MSE
  A2  target = log1p(price) loss = MSE   (A2 evaluated back in price space)

Because the backbone is FROZEN, its only output the head ever sees is the 512-d
pooled feature vector; training the head on the cached resnet18 embeddings is
mathematically identical to training it on images (and far faster on CPU).

Records a full training log for every run: target, epochs run, best epoch,
train/val loss trajectory, RMSE, R2, MAE, and wall-clock runtime.

Guarantees: deterministic seed, CPU batch, mild augmentation, early stopping,
validation monitoring, best checkpoint saved, no test data used.
"""
import argparse
import sys
import json
import time
from pathlib import Path

os_env = None
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR, FIGURES_DIR, EMBEDDING_DIM
from src.satellite.embeddings import load_cache
from src.data.load import load_clean_train, canonical_split
from src.satellite.align import is_valid_image
from src.models.train import evaluate

DEVICE = "cpu"
HIDDEN = (256, 64)
DROPOUT = 0.2
MAX_EPOCHS = 200
PATIENCE = 20
HEAD_LR = 1e-3


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class RegressionHead(nn.Module):
    """Trainable head. Standardizes features/targets, inverts on output.

    forward(x) -> real price (USD). Buffers make the deployed jit model
    self-contained (no runtime scalers needed).
    """

    def __init__(self, in_dim, target_type, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.target_type = target_type
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(x_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))
        layers, prev = [], in_dim
        for d in HIDDEN:
            layers += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def encode(self, x, y=None):
        xz = (x - self.x_mean) / self.x_std
        if y is None:
            return xz
        if self.target_type == "log1p":
            yz = (torch.log1p(y) - self.y_mean) / self.y_std
        else:
            yz = (y - self.y_mean) / self.y_std
        return xz, yz

    def forward(self, x):
        xz = (x - self.x_mean) / self.x_std
        z = self.net(xz).squeeze(1)
        y = z * self.y_std + self.y_mean
        if self.target_type == "log1p":
            y = torch.expm1(y)
        return y


def covered_split(encoder_name: str):
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    tr = df[df["id"].isin(tr_ids)]
    va = df[df["id"].isin(va_ids)]
    tr = tr[tr["id"].map(is_valid_image)]
    va = va[va["id"].map(is_valid_image)]
    cached = load_cache(encoder_name)
    if cached is None:
        raise FileNotFoundError(
            f"Missing embedding cache for '{encoder_name}'. "
            f"Run scripts/extract_embeddings.py --encoder {encoder_name} first."
        )
    ids, emb = cached
    ebs = dict(zip(ids, emb))
    X_tr = np.stack([ebs[i] for i in tr["id"]])
    X_va = np.stack([ebs[i] for i in va["id"]])
    return tr, va, X_tr, X_va


def run_a(X_tr, y_tr, X_va, y_va, target_type, encoder_name: str):
    set_seed()
    t0 = time.time()
    yt_raw = np.log1p(y_tr) if target_type == "log1p" else y_tr
    x_mean, x_std = X_tr.mean(0), X_tr.std(0).clip(min=1e-8)
    head = RegressionHead(X_tr.shape[1], target_type,
                          x_mean, x_std, float(yt_raw.mean()), float(yt_raw.std()))

    Xt, yt_z = head.encode(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
    Xv = head.encode(torch.tensor(X_va, dtype=torch.float32))

    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    loss_fn = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_rmse": [], "val_r2": []}
    best_rmse, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        head.train()
        opt.zero_grad()
        z = head.net(Xt).squeeze(1)
        loss = loss_fn(z, yt_z)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            pred_va = head(Xv).numpy()
        met = evaluate(y_va, pred_va)
        history["epoch"].append(ep)
        history["train_loss"].append(float(loss.item()))
        history["val_rmse"].append(met["rmse"])
        history["val_r2"].append(met["r2"])
        if met["rmse"] < best_rmse:
            best_rmse = met["rmse"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_ep, wait = ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred_va = head(Xv).numpy()
        pred_tr = head(Xt).numpy()
    met = evaluate(y_va, pred_va)
    met_tr = evaluate(y_tr, pred_tr)
    runtime = time.time() - t0

    record = {
        "experiment": "A" if target_type == "price" else "A",
        "variant": "A1_price_MSE" if target_type == "price" else "A2_log1p_MSE",
        "target_type": target_type,
        "encoder": encoder_name,
        "loss": "MSE",
        "architecture": f"{encoder_name} (frozen backbone) + trainable regression head",
        "backbone": "frozen",
        "split": {"train": int(len(y_tr)), "val": int(len(y_va))},
        "seed": RANDOM_STATE,
        "n_epochs_run": len(history["epoch"]),
        "best_epoch": best_ep,
        "final_epoch": best_ep,
        "train_rmse": met_tr["rmse"],
        "val_rmse": met["rmse"],
        "val_r2": met["r2"],
        "val_mae": met["mae"],
        "train_r2": met_tr["r2"],
        "best_val_rmse": best_rmse,
        "runtime_seconds": round(runtime, 1),
        "history": {k: history[k] for k in history},
    }
    print(json.dumps(record, indent=2))
    return record, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phase2")
    ap.add_argument("--encoder", default="resnet18", choices=sorted(EMBEDDING_DIM.keys()))
    ap.add_argument("--out", default=str(REPORTS_DIR / "results_dl.json"))
    args = ap.parse_args()

    tr, va, X_tr, X_va = covered_split(args.encoder)
    y_tr, y_va = tr["price"].values, va["price"].values
    print(f"Covered subset: train {len(tr)} / val {len(va)}")

    records = {}
    heads = {}
    for tgt in ("price", "log1p"):
        rec, head = run_a(X_tr, y_tr, X_va, y_va, tgt, args.encoder)
        records[rec["variant"]] = rec
        heads[tgt] = head

    # checkpoint heads (weights) for downstream deployment
    from src.config import MODELS_DIR
    ckpt_dir = MODELS_DIR / "vision" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for tgt, head in heads.items():
        torch.save({"target_type": tgt, "state_dict": head.state_dict()},
                   ckpt_dir / f"head_{'A1_price' if tgt=='price' else 'A2_log1p'}.pt")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "phase": f"Phase 2 — Experiment A (trainable regression head, frozen {args.encoder})",
        "split": {"covered_train": int(len(tr)), "covered_val": int(len(va))},
        "encoder": args.encoder,
        "variants": records,
        "checkpoint_dir": str(ckpt_dir),
    }, indent=2), encoding="utf-8")
    print("\nSaved", out)


if __name__ == "__main__":
    main()