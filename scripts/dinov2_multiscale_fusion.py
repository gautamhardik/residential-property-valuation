"""Multi-view DINOv2 satellite embeddings + fusion experiment.

This is a practical image upgrade over the single-tile baseline: for each valid
property image, generate several views (original, center-crop, wider crop,
mirror) and average the DINOv2 embeddings before concatenating with tabular
features. The goal is to extract stronger parcel-level context without changing
any of the champion tabular logic.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR, RANDOM_STATE
from src.data.load import canonical_split, load_clean_train
from src.features.build_features import FEATURE_COLS, _engineer, fit_target_encoder
from src.models.train import evaluate, make_champion_estimator
from src.satellite.align import is_valid_image, image_path_for


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BASE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def build_views(im: Image.Image):
    rgb = im.convert("RGB")
    w, h = rgb.size
    views = [
        rgb,
        rgb.resize((224, 224)),
    ]

    crop_w = max(1, int(w * 0.85))
    crop_h = max(1, int(h * 0.85))
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    region = rgb.crop((left, top, left + crop_w, top + crop_h)).resize((224, 224))
    views.append(region)

    wide = rgb.crop((0, 0, max(1, w - 10), h)).resize((224, 224))
    views.append(wide)
    views.append(ImageOps.mirror(rgb).resize((224, 224)))

    unique, seen = [], set()
    for view in views:
        key = (view.size, view.mode)
        if key not in seen:
            unique.append(view)
            seen.add(key)
    return unique


def load_dino_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    return model


def image_embedding(model, pid, transform_fn=BASE_TF):
    path = image_path_for(pid)
    with Image.open(path).convert("RGB") as im:
        feats = []
        for view in build_views(im):
            x = transform_fn(view).unsqueeze(0)
            with torch.no_grad():
                z = model(x)
            if z.ndim > 1:
                z = z[:, 0]
            feats.append(z.cpu().numpy()[0])
        return np.mean(np.vstack(feats), axis=0).astype(np.float32)


def build_tabular(train_df, val_df):
    X_tr = _engineer(train_df)
    X_va = _engineer(val_df)
    global_mean, enc = fit_target_encoder(X_tr)
    for fr in (X_tr, X_va):
        fr["zip_target"] = fr["zipcode"].map(enc).fillna(global_mean)
    cols = [c for c in FEATURE_COLS if c != "price"]
    return X_tr[cols].values, X_va[cols].values


def run():
    df = load_clean_train()
    tr_ids, va_ids = canonical_split(df)
    train = df[df["id"].isin(tr_ids)].copy()
    val = df[df["id"].isin(va_ids)].copy()

    valid_train = train[train["id"].map(is_valid_image)]
    valid_val = val[val["id"].map(is_valid_image)]
    print(f"Image-covered subset: train={len(valid_train)} val={len(valid_val)}")

    model = load_dino_model()
    emb_tr = []
    emb_va = []
    for pid in valid_train["id"].tolist():
        emb_tr.append(image_embedding(model, int(pid)))
    for pid in valid_val["id"].tolist():
        emb_va.append(image_embedding(model, int(pid)))
    X_tr_tab, X_va_tab = build_tabular(valid_train, valid_val)
    E_tr = np.stack(emb_tr)
    E_va = np.stack(emb_va)

    y_tr = valid_train["price"].values
    y_va = valid_val["price"].values

    rows = []
    tab_model = make_champion_estimator().fit(X_tr_tab, y_tr)
    tab_met = evaluate(y_va, tab_model.predict(X_va_tab))
    rows.append({"experiment": "E4_multiview_tabular_only", **tab_met})

    fusion_model = make_champion_estimator().fit(np.hstack([X_tr_tab, E_tr]), y_tr)
    fusion_met = evaluate(y_va, fusion_model.predict(np.hstack([X_va_tab, E_va])))
    rows.append({"experiment": "E6_multiview_dinov2_fusion", **fusion_met})

    base = rows[0]["rmse"]
    for row in rows:
        row["rmse_improvement_pct"] = (base - row["rmse"]) / base * 100.0
        row["r2_change"] = row["r2"] - rows[0]["r2"]

    out = {
        "encoder": "dinov2_vits14_multiview",
        "split": {"covered_train": int(len(valid_train)), "covered_val": int(len(valid_val))},
        "results": rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "results_dinov2_multiview.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "results_dinov2_multiview.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved", REPORTS_DIR / "results_dinov2_multiview.json")


if __name__ == "__main__":
    run()
