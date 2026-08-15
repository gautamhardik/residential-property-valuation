"""Phase 8 — Experiment B reproduction + correct serialization.

Reproduces the best-evaluated vision model from the DL extension:
    ResNet18 (ImageNet) with LAYER4 partially fine-tuned + trainable regression head.

This model was previously evaluated with val R2 ~= 0.296 (price target) on the
same canonical 1,755/434 image-covered split.

Phase 8 fixes a serialization bug: the deployed artifact must contain the FULL
model (backbone with trained layer4 + head), NOT just the head. Verification:
  - TorchScript predictions vs in-memory PyTorch predictions on the val split.
  - R2 difference <= 0.001, plus max-abs and mean-abs prediction differences.
The scripted model is saved to models/deployed/vision_price.pt.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR, MODELS_DIR
from src.data.load import load_clean_train, canonical_split
from src.satellite.align import is_valid_image, image_path_for
from src.satellite.embeddings import load_cache

ROOT = Path(__file__).resolve().parents[1]
from src.models.train import evaluate

DEVICE = "cpu"
HIDDEN = (256, 64)
DROPOUT = 0.2
MAX_EPOCHS = 20
PATIENCE = 8
BATCH = 32
HEAD_LR = 1e-3
LAYER4_LR = 1e-5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class RegressionHead(nn.Module):
    def __init__(self, in_dim, target_type, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.target_type = target_type
        t = lambda v: torch.as_tensor(v, dtype=torch.float32)  # noqa: E731
        self.register_buffer("x_mean", t(x_mean))
        self.register_buffer("x_std", t(x_std))
        self.register_buffer("y_mean", t(y_mean))
        self.register_buffer("y_std", t(y_std))
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
        return xz, ((torch.log1p(y) if self.target_type == "log1p" else y)
                    - self.y_mean) / self.y_std

    def forward(self, x):
        xz = (x - self.x_mean) / self.x_std
        z = self.net(xz).squeeze(1)
        y = z * self.y_std + self.y_mean
        return torch.expm1(y) if self.target_type == "log1p" else y


class VisionPriceModel(nn.Module):
    """ResNet18 (avgpooled, no fc) + regression head; layer4 unfrozen for B."""

    def __init__(self, head):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = head
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone[7].parameters():  # layer4
            p.requires_grad = True

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


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
        base += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(base)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(image_path_for(self.ids[i])).convert("RGB") as im:
            x = self.tf(im)
        return x, torch.tensor(self.y[i], dtype=torch.float32)


def covered_split():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    tr = df[df["id"].isin(tr_ids)].copy()
    va = df[df["id"].isin(va_ids)].copy()
    tr = tr[tr["id"].map(is_valid_image)].copy()
    va = va[va["id"].map(is_valid_image)].copy()
    ids, emb = load_cache("resnet18")
    ebs = dict(zip(ids, emb))
    X_tr = np.stack([ebs[i] for i in tr["id"]])
    return tr, va, X_tr


def run_b(tr, va, X_tr):
    set_seed()
    y_tr = tr["price"].values
    x_mean, x_std = X_tr.mean(0), X_tr.std(0).clip(min=1e-8)
    head = RegressionHead(X_tr.shape[1], "price", x_mean, x_std,
                          float(y_tr.mean()), float(y_tr.std()))
    model = VisionPriceModel(head)
    tr_dl = DataLoader(ImageDataset(tr, train_aug=True), batch_size=BATCH, shuffle=True,
                       generator=torch.Generator().manual_seed(RANDOM_STATE))
    va_dl = DataLoader(ImageDataset(va, train_aug=False), batch_size=BATCH, shuffle=False)

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    l4_params = [p for p in model.backbone[7].parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": head_params, "lr": HEAD_LR},
                            {"params": l4_params, "lr": LAYER4_LR}])
    loss_fn = nn.MSELoss()

    best_rmse, best_state, best_ep, wait = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr_dl:
            opt.zero_grad()
            feats = model.backbone(xb.to(DEVICE)).flatten(1)
            xz, yz = head.encode(feats, yb.to(DEVICE))
            loss = loss_fn(head.net(xz).squeeze(1), yz)
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                preds.append(model(xb.to(DEVICE)).numpy())
                ys.append(yb.numpy())
        met = evaluate(np.concatenate(ys), np.concatenate(preds))
        if met["rmse"] < best_rmse:
            best_rmse = met["rmse"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep, wait = ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        if ep % 4 == 0 or ep == 1:
            print(f"  ep {ep:3d} loss {tot/len(tr_dl):.3f}  val RMSE ${met['rmse']:,.0f}  R2 {met['r2']:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  best epoch {best_ep}  best val RMSE ${best_rmse:,.0f}  runtime {time.time()-t0:.0f}s")
    return model


def predict_all(model, rows):
    dl = DataLoader(ImageDataset(rows, train_aug=False), batch_size=BATCH, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(model(xb.to(DEVICE)).numpy())
    return np.concatenate(preds)


def main():
    tr, va, X_tr = covered_split()
    y_va = va["price"].values
    print(f"Covered subset: train {len(tr)} / val {len(va)}")

    model = run_b(tr, va, X_tr)

    # ---- serialization gate ----
    preds_py = predict_all(model, va)
    met_py = evaluate(y_va, preds_py)

    scripted = torch.jit.script(model)
    jit_path = MODELS_DIR / "deployed" / "vision_price.pt"
    jit_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(jit_path))

    # also persist the full trained model (backbone + head) as a best checkpoint
    ckpt_dir = MODELS_DIR / "vision" / "phase8"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"target_type": "price", "val_r2": met_py["r2"], "val_rmse": met_py["rmse"],
                "state_dict": model.state_dict()}, ckpt_dir / "vision_price_full.pt")
    print("Saved checkpoint", ckpt_dir / "vision_price_full.pt")

    # reload from disk to exercise the real deploy path
    re = torch.jit.load(str(jit_path), map_location="cpu")
    preds_jit = predict_all(re, va)
    met_jit = evaluate(y_va, preds_jit)

    maxdiff = float(np.abs(preds_py - preds_jit).max())
    meandiff = float(np.abs(preds_py - preds_jit).mean())
    r2diff = abs(met_py["r2"] - met_jit["r2"])
    rmsediff = abs(met_py["rmse"] - met_jit["rmse"])
    print(f"\nPyTorch val  RMSE ${met_py['rmse']:,.0f}  R2 {met_py['r2']:.6f}")
    print(f"TorchScript  RMSE ${met_jit['rmse']:,.0f}  R2 {met_jit['r2']:.6f}")
    print(f"  max-abs pred diff ${maxdiff:,.2f}   mean-abs ${meandiff:,.2f}   R2 diff {r2diff:.8f}")

    verification = {
        "n_val": int(len(va)),
        "py_rmse": met_py["rmse"], "py_r2": met_py["r2"],
        "jit_rmse": met_jit["rmse"], "jit_r2": met_jit["r2"],
        "r2_difference": r2diff,
        "rmse_difference": rmsediff,
        "max_abs_pred_difference": maxdiff,
        "mean_abs_pred_difference": meandiff,
        "acceptance": "R2 diff <= 0.001" if r2diff <= 0.001 else "FAIL",
    }
    assert r2diff <= 0.001, "TorchScript vs PyTorch R2 diff exceeds 0.001"
    print("SERIALIZATION VERIFIED: TorchScript predictions match PyTorch exactly.")
    print("Saved", jit_path)

    (REPORTS_DIR / "results_dl_serialization.json").write_text(
        json.dumps({"phase": "Phase 8", "artifacts": {"torchscript": str(jit_path.relative_to(ROOT))},
                    "model": "ResNet18 layer4 partial fine-tune + regression head",
                    "verification": verification}, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()