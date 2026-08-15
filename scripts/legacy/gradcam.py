"""Phase 10b — Correctly-labelled Grad-CAM for the satellite encoder.

What this explains:
  the frozen ResNet18's embeddings were used by the *multimodal price model*
  (XGBoost on 33 tabular + 512 visual dims). We select the embedding dimensions
  that the price model found most predictive, then use Grad-CAM to show which
  image regions drive those specific embedding dimensions.

What this does NOT claim:
  it does NOT attribute the final price prediction itself to image pixels
  (the regressor is tree-based on pooled vectors). The captions say exactly that.
"""
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
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RANDOM_STATE, REPORTS_DIR, FIGURES_DIR
from src.data.load import load_clean_train, canonical_split
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.satellite.align import image_path_for, is_valid_image
from src.satellite.embeddings import load_cache, TRANSFORM

# NOTE (limitation): this Grad-CAM model is a PROXY. It is trained on the
# train-with-image subset only (not the official E5 sample) and is used to find
# which embedding dimensions the price regressor relies on. It does NOT explain
# the final tree prediction itself — see the figure captions and the report.
# It uses its own independent gradient-boost config (it is a separate image-
# proxying regressor, unrelated to the champion model's tuning artifact).
XGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.08, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)
N_TOP_DIMS = 24
TAB_COUNT = 33  # FEATURE_COLS minus 'price'


def top_visual_dimensions():
    """Return indices + weights of the embedding dims the price model relies on."""
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train = df[df["id"].isin(tr_ids)]
    ids18, emb18 = load_cache("resnet18")
    emb_by = dict(zip(ids18, emb18))
    sub = train[train["id"].isin(set(ids18))]
    y = sub["price"].values

    X_tab = _engineer(sub)
    gmean, enc = fit_target_encoder(X_tab)
    X_tab["zip_target"] = X_tab["zipcode"].map(enc).fillna(gmean)
    cols = [c for c in FEATURE_COLS if c != "price"]
    X_all = np.hstack([X_tab[cols].values, np.array([emb_by[p] for p in sub["id"]])])
    model = XGBRegressor(**XGB_PARAMS).fit(X_all, y)

    imp = model.feature_importances_
    emb_imp = imp[TAB_COUNT:]
    top = np.argsort(emb_imp)[::-1][:N_TOP_DIMS]
    weights = emb_imp[top]
    weights = weights / weights.sum()
    return top.tolist(), weights.tolist()


def gradcam(encoder, x, dim_idx, dim_weights, target_layer):
    acts, grads = {}, {}

    def fwd_hook(mod, inp, outp):
        acts["a"] = outp.detach()

    def bwd_hook(mod, ginp, gout):
        grads["g"] = gout[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    out = encoder(x)
    e = out.flatten(1)
    target = (dim_weights * e[0, dim_idx]).sum()
    encoder.zero_grad()
    target.backward()

    a = acts["a"]                       # (1,C,H,W)
    g = grads["g"]
    alpha = g.mean(dim=(2, 3), keepdim=True)          # per-channel importance
    cam = (alpha * a).sum(dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    h_fwd.remove()
    h_bwd.remove()
    return cam


def run():
    from src.satellite.embeddings import load_encoder
    dim_idx, dim_weights = top_visual_dimensions()
    print(f"Top visual dims (by price-model importance): {dim_idx}")

    encoder = load_encoder("resnet18")
    # Sequential(repr of children[:-1]) → ['0'=conv1...'7'=layer4, '8'=avgpool]
    layer4 = encoder[7]

    df = load_clean_train()
    _, va_ids = canonical_split(df)
    val = df[df["id"].isin(va_ids)]
    cand = val[val["id"].map(is_valid_image)].sort_values("price", ascending=False)
    picks = pd.concat([cand.head(3), cand.tail(3)])  # expensive + affordable

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    meta = []
    encoder.train(False)
    for _, row in picks.iterrows():
        pid = int(row["id"])
        img_path = image_path_for(pid)
        pil = Image.open(img_path).convert("RGB")
        x = TRANSFORM(pil).unsqueeze(0).requires_grad_(True)
        cam = gradcam(encoder, x, dim_idx, torch.tensor(dim_weights, dtype=torch.float32), layer4)

        heat = cv2.resize(cam, (256, 256))
        heat = np.uint8(255 * heat)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        base = np.array(pil.resize((256, 256)))
        base_bgr = base[:, :, ::-1]
        overlay = cv2.addWeighted(base_bgr, 0.6, heat, 0.4, 0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(base); axes[0].axis("off"); axes[0].set_title(f"Satellite tile (id {pid})")
        axes[1].imshow(overlay[:, :, ::-1]); axes[1].axis("off")
        axes[1].set_title("Grad-CAM: regions driving price-relevant embedding dims")
        fig.suptitle(
            f"Actual price ${row['price']:,.0f} — Grad-CAM explains the frozen encoder's\n"
            f"price-relevant visual dims, NOT the tree model's prediction.",
            fontsize=9,
        )
        fname = f"gradcam_{pid}.png"
        fig.savefig(FIGURES_DIR / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        meta.append({"id": pid, "price": int(row["price"]), "file": fname,
                     "explains": "frozen encoder embedding dims important to the price regressor"})
        print(f"saved {fname} (price ${row['price']:,.0f})")

    (REPORTS_DIR / "gradcam_summary.json").write_text(
        json.dumps({"top_visual_dims": dim_idx, "samples": meta}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(4)
    run()