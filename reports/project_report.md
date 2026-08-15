# Project Report — Satellite Property Valuation

**Date:** 2026-08-14
**Author:** Data Science Team
**Status:** Final (portfolio release)

---

## 1. Executive Summary

**Problem.** Estimate residential sale prices in King County, WA, from structured housing/geospatial
attributes and satellite imagery.

**Method.** A controlled regression study. Engineered tabular + geospatial features feed a
leakage-safe pipeline and a tuned **XGBoost** model (random-holdout RMSE **$103.8K**, R² **0.9205**).
A separate, identically-split experiment tests whether satellite imagery adds incremental signal —
first with frozen ImageNet ResNet18/50 embeddings, then with task-trained ResNet18 variants.

**Result.** Satellite imagery did **not** add value under any tested representation. Frozen embeddings
degraded the tabular control (R² 0.872 → 0.844); a trained visual head (R² 0.099/0.113) and a partial
layer4 fine-tune (R² 0.296) stayed far below the tabular control (0.872) on the same 434 validation
properties. Per a pre-registered decision gate, the vision branch was stopped and **XGBoost remained
the production champion**.

**Outcome.** Deployed: tuned XGBoost (primary) + clearly-labelled ResNet18 vision research endpoint
(secondary). This is an honest negative result: strong baseline → controlled multimodal test →
pre-registered stop → production selection.

---

## 2. Business Question

**Why was satellite imagery considered?** Conventional housing attributes (size, beds, bathrooms,
location) miss the *context* of a property — the character of the surrounding built and natural
environment. Remote-sensing imagery is a plausible, low-cost proxy for that context. The business
question is therefore:

> *Does remotely-sensed imagery provide predictive value for residential price estimation beyond
> conventional housing and geospatial variables?*

We answer with a controlled, identically-split experiment — not by cherry-picking a config that
"makes the images useful." The tool used to answer it is a property-valuation system for King County,
WA.

---

## 3. Dataset

**Task + target.** Estimate the sale price (`price`, USD) of residential properties from (a) 21
structured columns about the home/lot (bedrooms, sqft, grade, condition, waterfront, year built,
zipcode, latitude/longitude, ...) and (b) a 256×256 px satellite tile of each property's
neighborhood. The raw dataset simulates a Kaggle-style competition: `data/test.xlsx` (5,404 rows)
carries **no labels**.

**Raw → clean.** `data/train.xlsx` 16,209×23, `data/test.xlsx` 5,404×22. After the audit below, 16,110 clean
training rows feed every experiment.

**Satellite imagery.** Mapbox `satellite-v9`, 256×256 px, zoom 18, tiles keyed by property id, cached
on disk. Coverage is a **convenience sample**: 13.6% of properties (2,189/16,110) have a usable tile.
See §8 for the coverage-selection analysis.

---

## 4. Data Quality & Leakage

### 4.1 Audit (Phase 1)

`reports/data_audit.json`:

| Check | Result | Action |
|---|---|---|
| Duplicate ids in train | 99 | Repeat sales of the same property → keep the **most recent** transaction (`keep='last'`) |
| Duplicate ids in test | 8 | Same property, two rows → both predicted individually |
| Train/test overlapping ids | 70 | Same physical property; test has **no** labels → no target leakage |
| Coordinate validity | 100% valid lat/lon | Used everywhere as a locational signal |
| Duplicate coordinate pairs | 360 train / 47 test | Different homes at same parcel → distinct records |
| Price in test | No | Confirmed, no leakage vector |
| Dates | 2014–2015 both splits | Parsed to `sale_year`, `sale_quarter` |

The audit preceded any modeling, so all decisions were made without peeking at validation labels.

### 4.2 Target-encoding leakage audit (explicit)

`zip_target` trace:

```text
zip_target
  ↓  source:      zipcode
  ↓  transform:   smoothed mean of price per ZIP (James–Stein, λ=20)
  ↓  uses target? YES (price) — but only TRAINING labels
  ↓  when?        Fitted inside each experiment's training split (and inside each tuning fold)
  ↓  val/test?    Never — val/test rows never contribute to the encoding
  ↓  unseen ZIP?  Falls back to the global training mean
  ↓  leakage?     NONE for the holdout estimate
```

All other engineered features are deterministic functions of raw columns (no target). Train/test
overlap (70 ids) is benign because the test set carries no labels. **Verdict: leakage-safe.** The
0.9205 R² was re-verified after switching tuning to fold-safe encoding (§7); the numbers are unchanged.

---

## 5. Feature Engineering

From the raw columns (`src/features/build_features.py`):

- **Structural aggregations:** `total_sqft`, `basement_frac`, `above_frac`.
- **Ratios:** `lot_living_ratio`, `living_per_bedroom`.
- **Interactions / geospatial:** `lat_long_interaction`, `dist_to_center_km` (Haversine distance to
  downtown Seattle).
- **Spatial aggregation:** `zip_target` — smoothed target-encoded ZIP-mean of price (James–Stein,
  λ=20), fitted **only on the training split** (leakage analysis in §4.2).
- **Temporal:** `sale_year`, `sale_quarter` for the 2014→2015 trend.

---

## 6. Baseline Models

All models share identical splits, seeds, and preprocessing.

| Experiment | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| E1 original 5-feature set | LinearRegression | \$233.4k | \$148.6k | 0.598 |
| E1 original 5-feature set | RandomForest | \$162.5k | \$85.7k | 0.805 |
| E1 original 5-feature set | XGBoost | \$168.2k | \$85.8k | 0.791 |
| E2 full raw tabular | LinearRegression | \$195.1k | \$123.8k | 0.719 |
| E2 full raw tabular | RandomForest | \$131.2k | \$69.4k | 0.873 |
| E2 full raw tabular | XGBoost | \$115.1k | \$64.5k | 0.902 |
| E3 engineered tabular | LinearRegression | \$153.0k | \$95.8k | 0.827 |
| E3 engineered tabular | RandomForest | \$124.2k | \$67.8k | 0.886 |
| **E3 engineered tabular** | **XGBoost** | **\$110.6k** | **\$62.2k** | **0.910** |

(`reports/results_tabular.csv` — 9 models, 3 experiments × 3 algorithms, identical protocol.)

**Broader model space (leak-free OOF stack).** `reports/results_ensemble.csv` compares the tuned
families and a stack (OOF 3-fold predictions feed a Ridge stacker):

| Model | RMSE | R² |
|---|---|---|
| Tuned XGBoost (champion) | \$103.8k | 0.9205 |
| Tuned CatBoost | \$102.8k | 0.922 |
| LightGBM | \$111.9k | 0.908 |
| OOF-Ridge stack | \$101.6k | 0.924 |

The stack has the best *in-distribution* number but leans heavily on CatBoost (≈0.63 weight) and
inherits its generalization weakness — reported as evidence the space was explored, not deployed.

---

## 7. Final XGBoost

### 7.1 Tuning (fold-safe)

**Method (`scripts/tune_models.py`):** `RandomizedSearchCV` — 30 random iterations, 3-fold CV inside
the training fold, scored on neg RMSE. The canonical validation split is never touched during search.
`zip_target` is fitted inside a `Pipeline` so each CV fold encodes using **that fold's training
targets only**. Best models re-fit on the full training fold and are scored once on the untouched
holdout.

| Model | Selected params (abridged) | RMSE | MAE | R² |
|---|---|---|---|---|
| RandomForest | depth 43, n_est 208, max_features 0.5 | \$118.9k | \$66.2k | 0.896 |
| **XGBoost** | lr .087, max_depth 4, min_child_weight 5, λ 3.42, colsample .78, subsample .78 | **\$103.8k** | **\$61.2k** | **0.921** |
| CatBoost | depth 4, iters 366, lr .119, subsample .92 | \$103.6k | \$61.9k | 0.921 |

**Final model = tuned XGBoost on engineered tabular features** — holdout RMSE **$103,802.8**, R²
**0.9205** (config in `reports/tuned_best.json`, consumed by every downstream phase).

### 7.2 Why XGBoost, not the marginally better CatBoost?

Model selection is *generalization-aware*, not leaderboard-driven. CatBoost and the OOF-Ridge stack
beat XGBoost on the in-distribution random holdout, but over-fit the target-encoded `zip_target` and
**collapsed on out-of-sample tests**: CatBoost out-of-time R² 0.881 and spatial R² **0.598**, versus
XGBoost's 0.893 and 0.809. `scripts/tune_models.py` applies a **temporal tie-break** (within 1% random-holdout
RMSE → pick the better out-of-time model), which is what keeps XGBoost champion.

---

## 8. Satellite Experiment

### 8.1 Coverage & selection bias

`src/data/fetch_images.py` (Mapbox Static Images, retries/backoff on 429s, content validation, disk
caching). Coverage: **13.6%** (2,189/16,110).

**Coverage-selection analysis (`reports/coverage_bias.json`):** the image-covered subset (n=2,189) is
compared against the rest (n=13,921) on price, grade, size, bathrooms, coordinates and waterfront.
All |Cohen's d| < 0.15 (price d = −0.008, grade d = −0.039, waterfront 0.87% vs 0.68%; both subsets
span the same 70 ZIPs). The convenience sample is **attribute-neutral**, so the satellite experiment
is not confounded by a biased image sample.

### 8.2 Frozen embeddings (E4/E5)

Subset = the 2,189 image-covered properties, split with the **same canonical train/val ids**, identical
metric, and — for XGBoost — the exact tuned configuration. The tabular control (E4) stacks imagery
against the strongest tabular model available.

| Experiment | Family | RMSE | R² | ΔRMSE vs tabular control |
|---|---|---|---|---|
| E4 tabular control | XGB | \$126.8k | 0.872 | — |
| E4 tabular control | RF | \$150.3k | 0.820 | — |
| E4B image-only (ResNet18) | XGB | \$329.2k | 0.138 | +159% (worse) |
| E4B image-only (ResNet18) | RF | \$329.5k | 0.136 | +159% (worse) |
| E5 tabular + ResNet18 embeddings | XGB | \$139.8k | 0.844 | +10.3% (worse) |
| E5 tabular + ResNet18 embeddings | RF | \$164.1k | 0.786 | +29.4% (worse) |
| E5B tabular + ResNet50 embeddings | XGB | \$147.0k | 0.828 | +15.9% (worse) |
| E5B tabular + ResNet50 embeddings | RF | \$159.4k | 0.798 | +25.7% (worse) |

(`reports/results_multimodal.csv`; PCA-denoised variants in `reports/results_multimodal_pca.csv` — PCA never
recovers the loss, confirming the embeddings add noise rather than signal.)

**Scoped conclusion:** frozen ImageNet ResNet18/50 embeddings did **not** provide incremental value
over the tabular/geospatial baseline under the evaluated setup.

### 8.3 Task-trained visual representations (the DL extension)

The frozen-embedding gate motivates a stronger test: would *task-specific training* of a CNN recover
visual signal that generic ImageNet features missed? Run on the same 2,189 image-covered properties,
canonical split (1,755 train / 434 val), no test data:

| Experiment | Model | Input | RMSE | R² |
|---|---|---|---|---|
| A1 | ResNet18 + trained head | Image | \$336.6k | 0.099 |
| A2 | ResNet18 + trained head (log-price) | Image | \$334.0k | 0.113 |
| B | ResNet18 partial fine-tune (layer4) | Image | \$297.5k | **0.296** |
| (control) | XGBoost tabular | Tabular | \$126.8k | 0.872 |

**Protocol.** A1/A2: frozen ImageNet ResNet18 + trainable MLP head (512→256→64→1), MSE on `price`
and `log1p(price)` (evaluated back in price space). Frozen backbone ⇒ head trained on cached 512-d
pooled features (identical to training on images). B: partial fine-tune of layer4 (LR 1e-5) + head
(1e-3), same split/standardization, early stopping on validation RMSE, seed 42. Feature/target
standardization is baked into the head as buffers so the deployed model outputs USD directly.

---

## 9. Why the Vision Branch Was Rejected

**The negative result, stated plainly.**

1. **Frozen embeddings hurt.** Adding ResNet18 embeddings to the tuned tabular model degraded it
   (E4 0.872 → E5 0.844); image-only models were near-useless (R² ≈ 0.14).
2. **Task-specific training did not recover the signal.** A trained visual head (A1 0.099 / A2
   0.113) and a partial fine-tune (B 0.296) remain far below the tabular control (0.872) on the same
   434 properties.
3. **Pre-registered decision gate → Case 1 (STOP).** Because even the trained representation did not
   approach the tabular signal, no fusion model (E6), ViT, ResNet50 fine-tuning, PCA sweeps, or TTA
   were attempted. The model space was not expanded to chase a win; the negative result is the result.
   (See `reports/results_dl_eval.json`, `reports/results_dl_final.json`.)

**Interpretation.** Under the available coverage (13.6%) and single 256px Mapbox tile per property,
the visual modality provides limited incremental information beyond structured/geospatial features.
A property-level valuation signal is not captured by scene-level ImageNet features of one low-res
tile. Higher resolution / multi-angle / multi-temporal imagery is an open future direction — not an
unfinished requirement of this study.

---

## 10. Error Analysis

`reports/error_analysis.json` (validation n = 3,222). Overall: MAE $61.2k, RMSE $103.8k, R² 0.9205;
median relative error ≈ **8%**.

| Segment | Metric | Behavior |
|---|---|---|
| Price: Low/Mid/High/Luxury | RMSE | Luxury (n=806, mean \$990k): RMSE \$178.6k, bias −\$30.2k → tail under-priced; low band RMSE \$54.7k, bias +\$16.8k |
| Size: Small→XL | RMSE | XL homes (mean \$874k): RMSE \$166.2k; small/medium ≈ \$63k |
| Condition 1–2 (rare, n≤30) | med rel err | 22–25% vs ~8% for condition 3–5 |
| Waterfront (n=21) | RMSE | \$262.3k, bias −\$124.4k — smallest, most valuable class; worst segment |
| Location (16 bands) | RMSE | Central-neighborhood luxury rings worst; southern bands best |
| Has satellite image | RMSE | 92.1k (n=434) vs 105.5k (n=2,788) — small-sample residual, not a population difference |

Waterfront (n=21), the luxury tail, and rare-condition homes are small-sample segments; aggregate
metrics hide them, which is why they are reported explicitly. **Where the final XGBoost model
struggles:** expensive, rare, and extreme homes — the luxury tail, waterfront, and condition-1/2
properties.

---

## 11. Spatial Robustness

Same tuned XGBoost, same engineered features, same metric; **only the split differs**
(`reports/split_strategy.json`, `reports/temporal_validation.json`):

| Split | Train / Val n | RMSE | R² |
|---|---|---|---|
| Random 80/20 (primary) | 12,888 / 3,222 | \$103.8k | **0.921** |
| Out-of-time (2015 Mar–May) | 12,554 / 3,556 | \$117.5k | **0.893** |
| Spatial holdout (KMeans cell) | 15,656 / 454 | \$95.4k | **0.809** |

**Interpretation.** Random validation estimates performance when the model can exploit geographic
similarity between training and validation observations; the spatial holdout provides a more demanding
geographic-generalization test. The gap is attributable to the split, not a model/feature difference.

Caveats, stated plainly:

- The out-of-time leg trains only on sales ≤ 2015-02-28 and predicts 2015-03→05 — no temporal
  interleaving — yet still explains 89% of future variance.
- The spatial-holdout **RMSE is lower in absolute terms** only because that small holdout (n=454, ~2
  KMeans cells) covers a lower-variance price region; **R² is the more informative robustness
  comparison**, and it is the value quoted in every headline.
- 99.6% of random-split validation homes lie within 1 km of a training home, so the random-split
  number largely reflects interpolation within known neighborhoods. For any out-of-sample claim, quote
  0.893 (time) or 0.809 (space), never the 0.921 interpolation number.

---

## 12. Explainability

### 12.1 Global SHAP (production XGBoost)

`scripts/shap_analysis.py` (TreeExplainer on the tuned XGBoost, explained on an untouched 300-row
validation sample; `reports/shap_summary.json`). **Global SHAP importance** = mean |SHAP value|:

| Feature | Mean \|SHAP\| |
|---|---:|
| zip_target | \$78.0k |
| grade | \$62.2k |
| sqft_living | \$50.9k |
| dist_to_center_km | \$48.2k |
| lat | \$26.8k |

SHAP describes **feature contributions to the model**, not causal drivers of price. (The API/CLI
report a *different* but equally valid quantity — XGBoost gain importance — labelled as such in the
response; SHAP is the report's primary method.)

### 12.2 Grad-CAM (vision model)

Since §9 froze the vision branch with a negative result, explainability is provided for the *actual
evaluated* model (B: layer4 fine-tuned, predicts price directly) via `scripts/phase12_gradcam.py`.
It shows which image regions are **associated with the model's visual prediction** — correlational
description only, never "these regions cause the property to be expensive."

The saliency is diffuse across the parcel, consistent with the embeddings being uninformative; the
fine-tuned model's high-error/high-cross predictions (e.g. predicted ~$2.5M vs actual $3.4M) are
themselves evidence of how little visual signal the imagery carries.

---

## 13. Deployment

### 13.1 Production (primary)

```text
POST /predict  →  engineered features  →  tuned XGBoost  →  property price   [PRIMARY]
```

- Full-train retrain (16,110 rows) → 5,404-row `predictions/submission.csv` (1:1 mirror of
  `data/test.xlsx`, id multiset identical, 0 nulls).
- FastAPI service (`app/backend/main.py`: `/predict`, `/health`, demo UI), CLI (`app/cli.py`), and
  Python API (`src/inference.predict.predict_single`). Inference re-invokes the same feature
  engineering and persisted encoder, so the deployed model is the exact final model.
- Freeze snapshot: `reports/baseline_snapshot.json` (SHA-256 checksums of model/pipeline/submission
  verified unchanged; RMSE $103,802.8 / R² 0.9205).

### 13.2 Vision research endpoint (secondary)

```text
POST /predict-image  →  TorchScript ResNet18 (partial fine-tune)  →  vision_only  [SECONDARY]
```

The vision model lost the gate (§9) and is **not** the production champion. It is serialized as a
CPU-compatible TorchScript artifact (`models/deployed/vision_price.pt`) containing the full model
(backbone + trained layer4 + head) and served strictly as a labelled research service
(`model_type: vision_only`), so the two models can never be confused. CLI:
`python -m app.cli --pid <id>`. Frontend: dedicated "Vision experiment" tab with an explicit
NOT-the-champion notice.

Serialization verified: TorchScript and PyTorch produce identical predictions (max-abs difference
$0.00, R² diff 0.0000), and inference runs in eval mode with no training at request time.

---

## 14. Limitations & Future Work

**Dataset.** King County only; 2014–2015 sales; single regional market — no claim of geographic or
temporal generality.

**Satellite.** ~13.6% usable coverage (attribute-neutral on observables, but not random); single
source (Mapbox `satellite-v9`); fixed 256 px at zoom 18; one scale.

**Vision.** Scene-level ImageNet features of a single tile do not capture property-level valuation
signals under the evaluated setup. A different representation — street-level or roof imagery,
building footprints, multi-angle/multi-temporal data, higher resolution — is the obvious future
direction, and was deliberately not chased beyond the pre-registered stop.

**Evaluation.** The random-split headline reflects interpolation within known neighborhoods; the
out-of-time (0.893) and spatial-holdout (0.809) numbers are the defensible out-of-sample claims.

**Business.** A sale price is not intrinsic value; predictions are point estimates, not appraisals;
no causal claims are made about any feature.

**Future work.** Higher-quality/multi-temporal remote-sensing data with a remote-sensing-specific
representation — not more architectures on the same 2,189 tiles.

---

## Appendix — Reproduction

```bash
pip install -r requirements.txt
cp .env.example .env        # set a Mapbox public token (only needed to re-fetch imagery)
python scripts/data_audit.py
python scripts/tabular_baselines.py
python scripts/split_validation.py
python scripts/temporal_validation.py
python scripts/tune_models.py
python scripts/ensemble_experiment.py
python scripts/satellite_coverage.py
python scripts/extract_embeddings.py   # or reuse cached preprocessed/embeddings_*.npz
python scripts/multimodal_experiment.py
python scripts/multimodal_pca.py                   # PCA-denoising ablation (§8.2)
python scripts/final_model.py
python scripts/error_analysis.py
python scripts/shap_analysis.py
python scripts/make_figures.py                   # 5 portfolio figures (from committed artifacts)
python scripts/make_notebooks.py                 # regenerates the clean narrative notebooks
python scripts/phase1_manifest.py                # canonical 1,755/434 vision split + manifest
python scripts/phase2_experiment_a.py            # A1/A2 trainable-head experiments
python scripts/phase3_4_eval_gate.py             # fair E4/E5/A1/A2 table + decision gate
python scripts/phase8_serialize.py               # full-model TorchScript (backbone+layer4+head) + verify
python scripts/phase12_gradcam.py                # Grad-CAM from the actual evaluated model
python scripts/phase13_14_final_table.py         # authoritative final experiment table + conclusion
python scripts/phase16_qa.py                     # full automated QA cascade
```

Every script is seeded, idempotent, and writes its artifact under `reports/`. Satellite embeddings
and the split are cached under `preprocessed/`, so the multimodal phase re-runs offline. The vision
model (`models/deployed/vision_price.pt`) is served via the API's `POST /predict-image`, the CLI's
`--pid`, and the frontend's Vision tab (see `app/backend/main.py`, `app/cli.py`,
`src/inference/vision.py`).

**External requirements.** Raw `data/train.xlsx`/`test.xlsx` must be supplied (not committed); a
Mapbox access token is required only to *re-fetch* imagery (cached tiles and embeddings are
committed); the committed `preprocessed/` and `models/` artifacts are sufficient for inference.