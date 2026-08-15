"""Fine-tuning / trained vision model (DL experiment, isolated from the tabular champion).

Scientific question:
    Can a TRAINED ResNet18 visual representation recover valuation signal that frozen
    ImageNet embeddings failed to capture (E4B image-only R2 0.138)?

Experiments (both use the canonical 1,755/434 image-covered split):
  A  ResNet18 (ImageNet) with a TRAINABLE REGRESSION HEAD and frozen backbone.
     Two target variants: price (MSE) and log1p(price) (MSE, evaluated back in price).
     With a frozen backbone the head only ever sees the 512-d pooled features, so
     training it on the cached resnet18 embeddings (preprocessed/embeddings_resnet18.npz)
     is mathematically identical to training it on images.
  B  PARTIAL FINE-TUNING: unfreeze layer4 of the backbone only, low LR, early stopping.
     Gated: only runs if A is stable (best price-space val R2 > 0). If A is unstable /
     overfits badly / produces nonsensical predictions, B is skipped and the failure is
     reported rather than tuned away.

Numerical hygiene (no change to the science):
  - features and targets are standardized per-experiment; RMSE/MAE/R2 are invariant to
    the affine transforms, and the inverse transforms are baked into the head as
    buffers so the deployed torch.jit model outputs real USD prices directly.
  - MSE for the log1p variant is computed in log1p-space (as planned).

Guardrails honored: the XGBoost champion, E1-E5 artifacts, features, submission and
validation methodology are NOT modified by this script.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # required before torch import on this box

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
from src.data.load import load_clean_train, canonical_split
from src.satellite.align import is_valid_image, image_path_for
from src.satellite.embeddings import load_cache
from src.models.train import evaluate

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

DEVICE = "cpu"
HIDDEN = (256, 64)
DROPOUT = 0.2
MAX_EPOCHS_A = 150
MAX_EPOCHS_B = 20
PATIENCE_A, PATIENCE_B = 8, 8
BATCH = 32
HEAD_LR = 1e-3
LAYER4_LR = 1e-5


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class RegressionHead(nn.Module):
    """Trainable head. Standardizes features/targets, inverts on output.

    forward(x) -> real price (USD). Buffers make the deployed jit model self-contained.
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
        """Standardized features; and standardized target-space labels for MSE."""
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


class VisionPriceModel(nn.Module):
    """ResNet18 backbone (avgpooled, no fc) + trainable regression head.

    Handles both exp A (fully frozen backbone) and exp B (layer4 unfrozen).
    """

    def __init__(self, head, freeze=True, unfreeze_layer4=False):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # incl. avgpool, 512-d
        self.head = head
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if unfreeze_layer4:
            for p in self.backbone[7].parameters():  # index 7 == layer4
                p.requires_grad = True

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


def embeddings_subset():
    """Train/val rows + cached frozen-embedding arrays for the covered ids."""
    ids, emb = load_cache("resnet18")
    emb_by = dict(zip(ids, emb))
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    tr = df[df["id"].isin(tr_ids)]
    va = df[df["id"].isin(va_ids)]
    tr = tr[tr["id"].map(is_valid_image)]
    va = va[va["id"].map(is_valid_image)]
    X_tr = np.stack([emb_by[i] for i in tr["id"]])
    X_va = np.stack([emb_by[i] for i in va["id"]])
    return tr, va, X_tr, X_va


def make_head(X_tr, y_tr, target_type):
    x_mean, x_std = X_tr.mean(0), X_tr.std(0).clip(min=1e-8)
    yt = np.log1p(y_tr) if target_type == "log1p" else y_tr
    return RegressionHead(X_tr.shape[1], target_type,
                          x_mean, x_std, float(yt.mean()), float(yt.std())) if yt.std() > 0 else \
        RegressionHead(X_tr.shape[1], target_type, x_mean, x_std, 0.0, 1.0)


def train_head(X_tr, y_tr, X_va, y_va, target_type, label):
    """Experiment A: train only the head on the frozen-backbone embeddings."""
    set_seed()
    head = make_head(X_tr, y_tr, target_type)
    Xt, yt_z = head.encode(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
    Xv = head.encode(torch.tensor(X_va, dtype=torch.float32))

    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    loss_fn = nn.MSELoss()

    best_rmse, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(1, MAX_EPOCHS_A + 1):
        head.train()
        opt.zero_grad()
        z = head.net(Xt).squeeze(1)
        loss = loss_fn(z, yt_z)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            pred_va = head(Xv).numpy()
            pred_tr = head(Xt).numpy()
        met = evaluate(y_va, pred_va)
        if met["rmse"] < best_rmse:
            best_rmse = met["rmse"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_ep, wait = ep, 0
        else:
            wait += 1
            if wait >= PATIENCE_A:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred_va = head(Xv).numpy()
    met = evaluate(y_va, pred_va)
    print(f"  [{label}] exp A head -> val RMSE ${met['rmse']:,.0f}  R2 {met['r2']:.4f}  (best ep {best_ep})")
    return met, head


class ImageDataset(Dataset):
    def __init__(self, rows, train_aug=False):
        self.ids = rows["id"].tolist()
        self.y = rows["price"].values
        base = [transforms.Resize((224, 224))]
        if train_aug:
            base += [transforms.RandomHorizontalFlip(),
                     transforms.RandomRotation(10),
                     transforms.RandomApply(
                         [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.5)]
        base += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
        self.tf = transforms.Compose(base)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(image_path_for(self.ids[i])).convert("RGB") as im:
            x = self.tf(im)
        return x, torch.tensor(self.y[i], dtype=torch.float32)


def run_experiment_b(tr, va, X_tr, target_type, label):
    """Experiment B: partial fine-tuning — unfreeze layer4 of the backbone only."""
    set_seed()
    head = make_head(X_tr, tr["price"].values, target_type)
    model = VisionPriceModel(head, freeze=False, unfreeze_layer4=True)
    train_ds = ImageDataset(tr, train_aug=True)
    val_ds = ImageDataset(va, train_aug=False)
    gen = torch.Generator().manual_seed(RANDOM_STATE)
    tr_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=gen)
    va_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    l4_params = [p for p in model.backbone[7].parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": head_params, "lr": HEAD_LR},
                            {"params": l4_params, "lr": LAYER4_LR}])
    loss_fn = nn.MSELoss()

    def loss_batch(xb, yb):
        # MSE computed in standardized target space (log1p or price), gradients flow to layer4
        feats = model.backbone(xb.to(DEVICE)).flatten(1)
        xz, yz = model.head.encode(feats, yb.to(DEVICE))
        z = model.head.net(xz).squeeze(1)
        return loss_fn(z, yz)

    best_rmse, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(1, MAX_EPOCHS_B + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = loss_batch(xb.to(DEVICE), yb.to(DEVICE))
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                preds.append(model(xb.to(DEVICE)).numpy())
                ys.append(yb.numpy())
        pred_va = np.concatenate(preds)
        met = evaluate(np.concatenate(ys), pred_va)
        if met["rmse"] < best_rmse:
            best_rmse = met["rmse"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep, wait = ep, 0
        else:
            wait += 1
            if wait >= PATIENCE_B:
                break
        if ep % 4 == 0 or ep == 1:
            print(f"    exp B ep {ep:3d} loss {tot/len(tr_dl):.3f}  val RMSE ${met['rmse']:,.0f}  R2 {met['r2']:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    print(f"  [{label}] exp B partial fine-tune -> val RMSE ${best_rmse:,.0f} (best ep {best_ep})")
    return best_rmse, model


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    cells = [("E4\nTabular control", 126_792, 0.872), ("Frozen emb.\n(E4B)", 329_239, 0.138)]
    for r in rows:
        cells.append((r["label"], r["rmse"], r["r2"]))
    labels = [c[0] for c in cells]
    rmses = [c[1] / 1000 for c in cells]
    r2s = [c[2] for c in cells]
    colors = ["#94a3b8", "#94a3b8"] + ["#2563eb"] * len(rows)
    bars = ax.bar(range(len(labels)), rmses, color=colors, edgecolor="white")
    for i, (v, r2) in enumerate(zip(rmses, r2s)):
        ax.text(i, v + 4, f"${v:,.0f}K\nR2 {r2:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE on image val subset (thousands USD)")
    ax.set_ylim(0, max(rmses) * 1.15)
    ax.set_title("Trained ResNet18 head vs frozen embeddings vs tabular control\n(434-property image validation subset)", fontsize=12, pad=12)
    fig.savefig(FIGURES_DIR / "fig_dl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_dl.png")


def build_scripted_model(model):
    """Return the deployable model with a frozen backbone (head weights untouched)."""
    if isinstance(model, RegressionHead):
        model = VisionPriceModel(model, freeze=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    tr, va, X_tr, X_va = embeddings_subset()
    y_tr, y_va = tr["price"].values, va["price"].values
    print(f"Covered subset: train {len(tr)} / val {len(va)}")

    # ---- Experiment A ----
    rows_A, heads_A = [], {}
    for target_type, label in (("price", "A price-MSE"), ("log1p", "A log1p-MSE")):
        met, head = train_head(X_tr, y_tr, X_va, y_va, target_type, label)
        rows_A.append({"label": f"Trained head ({label.split()[1]})", "target": target_type,
                       "rmse": met["rmse"], "r2": met["r2"], "mae": met["mae"],
                       "num_train": int(len(tr)), "num_val": int(len(va))})
        heads_A[target_type] = head

    # ---- stability gate ----
    bestA = max(rows_A, key=lambda r: r["r2"])
    stable = bestA["r2"] > 0.0  # meaningful signal vs predicting the mean
    print(f"\nExperiment A best: {bestA['label']} R2 {bestA['r2']:.4f}  -> stable={stable}")

    rows_B, models_B = [], {}
    if args.experiment in ("B", "both") and stable:
        for target_type, label in (("price", "B price-MSE"), ("log1p", "B log1p-MSE")):
            _, model = run_experiment_b(tr, va, X_tr, target_type, label)
            model.eval()
            va_ds = ImageDataset(va, train_aug=False)
            vdl = DataLoader(va_ds, batch_size=BATCH, shuffle=False)
            preds, ys_true = [], []
            with torch.no_grad():
                for xb, yb in vdl:
                    preds.append(model(xb.to(DEVICE)).numpy())
                    ys_true.append(yb.numpy())
            met = evaluate(np.concatenate(ys_true), np.concatenate(preds))
            rows_B.append({"label": f"Partial FT ({label.split()[1]})", "target": target_type,
                           "rmse": met["rmse"], "r2": met["r2"], "mae": met["mae"],
                           "num_train": int(len(tr)), "num_val": int(len(va))})
            models_B[target_type] = model
    elif args.experiment in ("B", "both"):
        print("Experiment A unstable (best val R2 <= 0). Skipping B per guardrail.")
        print(json.dumps({"status": "unstable_A",
                          "reason": "best_experiment_A_val_r2 <= 0",
                          "results_A": rows_A}, indent=2))

    # ---- select deployed vision model ----
    candidates = rows_A + rows_B
    best = max(candidates, key=lambda r: r["r2"])
    target_type = best["target"]
    print(f"\nSelected vision model: {best['label']}  RMSE ${best['rmse']:,.0f}  R2 {best['r2']:.4f}")

    model = heads_A[target_type] if best in rows_A else models_B[target_type]
    model = build_scripted_model(model.head if hasattr(model, "head") else model)
    # model is a VisionPriceModel-agnostic wrapper: if we get a head directly, wrap it.
    model.eval()

    jit_path = MODELS_DIR / "deployed" / "vision_price.pt"
    jit_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(jit_path))
    print(f"Saved torch.jit vision model -> {jit_path}")

    # sanity-check trained predictions are plausible (non-sensical guard)
    with torch.no_grad():
        demo = torch.zeros(1, 3, 224, 224)
        demo_pred = model(demo).item()
    print(f"  demo prediction (black image): ${demo_pred:,.0f}  (train price mean ${y_tr.mean():,.0f})")

    results = {
        "question": "Can a trained ResNet18 visual representation recover valuation signal frozen ImageNet embeddings missed?",
        "split": {"covered_train": int(len(tr)), "covered_val": int(len(va))},
        "stable_guard": bool(stable),
        "models": [
            {"label": "E4_tabular_control", "input": "tabular", "rmse": 126792.09, "r2": 0.8721},
            {"label": "E4B_image_only_frozen", "input": "image (frozen embeddings)", "rmse": 329239.51, "r2": 0.1375},
            *rows_A, *rows_B,
            {"label": "E5_mm_rn18", "input": "tabular + frozen embeddings", "rmse": 139807.70, "r2": 0.8445},
            {"label": "Final_XGBoost", "input": "full tabular (all 16,110)", "rmse": 103802.76, "r2": 0.9205},
        ],
        "selected": {"label": best["label"], "target": target_type, "rmse": best["rmse"], "r2": best["r2"]},
        "torch_jit": str(jit_path),
        "summary": "frozen-emb R2 0.138 -> trained-head R2 {:.3f}".format(best["r2"]),
    }
    out = REPORTS_DIR / "results_dl.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved", out)
    make_figure(rows_A + rows_B)


if __name__ == "__main__":
    main()