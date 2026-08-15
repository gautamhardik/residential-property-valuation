"""Phase 12 — Grad-CAM for the ACTUAL evaluated vision model.

Previous gradcam.py explained a PROXY (frozen encoder + separate XGBoost over
embedding dims). This version explains the model that was actually trained and
evaluated: VisionPriceModel = frozen-ImageNet ResNet18 + partial fine-tune of
layer4 + trainable regression head, predicting price directly.

Grad-CAM shows which image regions are associated with the model's *visual
prediction*. This is correlational description, not causal claim — captions use
"regions associated with the model's visual prediction", never "cause the price".
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
from src.data.load import load_clean_train, canonical_split
from src.satellite.align import image_path_for, is_valid_image
from src.inference.vision import TRANSFORM


def build_model():
    """Rebuild the exact evaluated architecture from the persisted jit artifact."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts dir
    from phase8_serialize import VisionPriceModel, RegressionHead
    from src.inference.vision import VISION_MODEL_FILE
    scripted = torch.jit.load(str(VISION_MODEL_FILE), map_location="cpu")
    head = RegressionHead(512, "price",
                          torch.zeros(512), torch.ones(512),
                          torch.tensor(0.0), torch.tensor(1.0))
    model = VisionPriceModel(head)
    model.load_state_dict(scripted.state_dict())
    model.eval()
    return model, scripted.state_dict()


def gradcam_price(model, x, target_layer):
    acts, grads = {}, {}

    def fwd_hook(mod, inp, outp):
        acts["a"] = outp.detach()

    def bwd_hook(mod, ginp, gout):
        grads["g"] = gout[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    out = model(x)                       # scalar price prediction (or (1,) batch)
    pred = out if out.dim() == 0 else out[0]
    model.zero_grad()
    pred.backward()

    a = acts["a"]
    g = grads["g"]
    alpha = g.mean(dim=(2, 3), keepdim=True)
    cam = (alpha * a).sum(dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    h_fwd.remove()
    h_bwd.remove()
    return cam, float(pred.detach())


def main():
    model, sd = build_model()
    val_r2 = None  # carried in results_dl_serialization.json, not in the jit artifact

    # target layer: layer4 (index 7 in the Sequential backbone)
    backbone = model.backbone
    target_layer = backbone[7]

    df = load_clean_train()
    _, va_ids = canonical_split(df)
    val = df[df["id"].isin(va_ids)]
    cand = val[val["id"].map(is_valid_image)].sort_values("price", ascending=False)
    picks = pd.concat([cand.head(3), cand.tail(3)])  # expensive + affordable

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    meta = []
    model.train(False)
    for _, row in picks.iterrows():
        pid = int(row["id"])
        pil = Image.open(image_path_for(pid)).convert("RGB")
        x = TRANSFORM(pil).unsqueeze(0).requires_grad_(True)
        cam, pred = gradcam_price(model, x, target_layer)

        heat = cv2.resize(cam, (256, 256))
        heat = np.uint8(255 * heat)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        base = np.array(pil.resize((256, 256)))
        base_bgr = base[:, :, ::-1]
        overlay = cv2.addWeighted(base_bgr, 0.6, heat, 0.4, 0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(base); axes[0].axis("off"); axes[0].set_title(f"Satellite tile (id {pid})")
        axes[1].imshow(overlay[:, :, ::-1]); axes[1].axis("off")
        axes[1].set_title("Grad-CAM: regions associated with the model's visual prediction")
        fig.suptitle(
            f"Actual ${row['price']:,.0f} — vision-model prediction ${pred:,.0f}\n"
            f"Correlational only; does NOT claim causation.",
            fontsize=9,
        )
        fname = f"gradcam_vision_{pid}.png"
        fig.savefig(FIGURES_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        meta.append({"id": pid, "price": int(row["price"]),
                     "vision_prediction": round(pred), "file": fname})
        print(f"saved {fname} (actual ${row['price']:,.0f} vs vision ${pred:,.0f})")

    summary = {
        "phase": "Phase 12",
        "explains": "ACTUAL evaluated model: ResNet18 partial fine-tune (layer4 + regression head), price target",
        "val_r2_of_explained_model": None,
        "caveat": "regions are ASSOCIATED with the model's visual prediction; not causal",
        "samples": meta,
    }
    (REPORTS_DIR / "gradcam_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("saved", REPORTS_DIR / "gradcam_summary.json")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()