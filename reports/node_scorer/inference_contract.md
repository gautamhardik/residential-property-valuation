# Inference Contract — XGBoost tabular champion (Node migration source of truth)

Authoritative reference: the Python implementation in `app/backend/main.py`,
`src/inference/{predict,explain,artifacts}.py`, `src/features/build_features.py`.
This document fixes every behavior the Node scorer must reproduce. Nothing here has
been changed; it is the audit required by Phase 1 of the Node migration.

## 0. Artifacts

| Path | Contents |
|---|---|
| `models/deployed/tabular_model.joblib` | fitted `XGBRegressor`, 535 trees, 33 features |
| `models/deployed/tabular_pipeline.joblib` | dict: `zip_enc`, `global_mean`, `feature_cols`, `model_type` |
| `reports/val_predictions_best_tabular.csv` | 20% holdout `predicted`/`price`, source of the error band |

Pipeline values (authoritative):
- `global_mean = 538518.8458100558`
- `feature_cols` = 33 names (order below).
- `zip_enc` = 70 ZIP codes -> float16-able target mean (James–Stein, smoothing=20).
- `model_type = "xgboost"`.

Model values (authoritative, read from `tabular_model.json`):
- objective `reg:squarederror`, `max_depth = 4`, 535 dense binary trees (full depth-4,
  31 nodes each).
- `base_score = 538518.9` (== global_mean as float32; `boost_from_average = 1`).
- `num_class=0`, `num_target=1`, single-output gbtree.

## 1. Raw input contract (API `PropertyInput`)

Defaults applied when omitted: `waterfront=0, view=0, condition=3, grade=7,
sqft_basement=0, yr_renovated=0, sale_year=2015, sale_quarter=3`.
`to_row` requires bedrooms/bathrooms/sqft_living/sqft_lot/floors/sqft_above/yr_built/
zipcode/lat/long/sqft_living15/sqft_lot15; the optional/quality fields fall back to
defaults. Field validation ranges live in `PRIMARY_FIELD_RULES` (backend) and are NOT
part of the model math.

## 2. Feature engineering (must be replicated bit-for-bit in Node)

Given the single-row DataFrame, `_engineer(df)` computes (all floats are IEEE-754
float64 until fed to the model; the model *splits* in float32):

- `age = max(ref - yr_built, 0)` where `ref = sale_year` (single row) — default 2015.
- `renovated = 1 if yr_renovated > 0 else 0`.
- `renovation_age = max(ref - yr_renovated, 0) if renovated else 0`.
- `total_sqft = sqft_living + sqft_basement`.
- `basement_frac = sqft_basement / (sqft_living + 1e-6)`.
- `above_frac = sqft_above / (sqft_living + 1e-6)`.
- `living_per_bedroom = sqft_living / (bedrooms + 1.0)`.
- `lot_living_ratio = sqft_lot / (sqft_living + 1e-6)`.
- `has_basement = 1 if sqft_basement > 0 else 0`.
- `lat_long_interaction = lat * long`.
- `dist_to_center_km = haversine_km(lat, long, 47.6062, -122.3321)`
  (radius `6371.0`, standard great-circle — see `src/utils.py`).
- `zip_freq = 1.0` for single-row inference (value_counts(normalize=True) of the single row).
- `zip_target = zip_enc.get(zipcode, global_mean)` (fallback only for ZIPs absent from
  the 70 training ZIPs).

Feature ORDER (33, must not change) — `feature_cols`:
```
[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition,
 grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long,
 sqft_living15, sqft_lot15, sale_year, sale_quarter, age, renovated, renovation_age,
 total_sqft, basement_frac, above_frac, living_per_bedroom, lot_living_ratio,
 has_basement, lat_long_interaction, dist_to_center_km, zip_freq, zip_target]
```
`feature_types`:
```
[int, float, int, int, float, int, int, int, int, int, int, int, int, int,
 float, float, int, int, int, int, int, int, int, int, float, float, float, float,
 int, float, float, float, float]
```
`zip_target` is float; all denominators use the exact constants above.

## 3. Prediction (verified exact)

- Each tree is a full binary tree; node `n`:
  - if `left_children[n] == -1` (== `right_children[n] == -1`) it is a LEAF with value
    `base_weights[n]`.
  - internal node: `feat = split_indices[n]; thr = split_conditions[n]`.
    - missing (feature absent / NaN): go left iff `default_left[n] == 1`.
    - present: `float32(feat_value) < float32(thr)` -> left, else right.
- Raw prediction = sequential **float32** accumulation:
  `acc = float32(base_score); for each tree leaf v: acc = float32(acc + float32(v))`.
  Left values already include learning_rate; no extra scaling.
- `predicted_price = round(raw, 2)`.

Verified: canonical example raw = `557597.375` -> `557597.38`; manual float32 walk
matches `xgb.Booster.predict` exactly (verified 0/535 tree-route divergences with
float32 comparison).

## 4. LOCAL SHAP / contributions (native `pred_contribs`)

- `booster.predict(dmat, pred_contribs=True)` returns 34 values:
  `[c_1..c_33, base]`, `prediction_contrib = base + sum(c_i)`.
- For canonical, `base = 538904.625`, and `base + sum(c) = 557597.125` (float32;
  reconstructs prediction within ~0.25/0.185 of `predict` — QA tolerance is < 1.0).
- `local_summary` returns `expected_value = round(base,2)`,
  `total_contribution = round(sum(c),2)`,
  `predicted_price = round(base + sum(c), 2)`, and top +/- contributions rounded to
  cents, filtered to nonzero.
- FEATURE_LABELS: human-readable map in `src/inference/explain.py`.

> Node must reproduce per-feature `c_i` (so top factors and base agree with Python to
> within tolerance), via a faithful TreeSHAP (path-dependent, interventional) that
> satisfies `base + sum(c_i) == prediction`.

## 5. Error band (not model math)

Reads the holdout CSV; buckets predictions with `pd.cut` on the unique quantiles
`[0,.2,.4,.6,.8,1]`; returns median absolute error + count for the segment containing
`predicted_price`. Reproducible in Node from the same CSV.

## 6. API response contract

`POST /predict` (also reached under `/api/predict` after the `/api` prefix strip)
returns (see `app/backend/main.py`):
```json
{
  "predicted_price": 557597.38,
  "model": "XGBoost (tuned) on engineered tabular features",
  "model_role": "primary",
  "status": "production",
  "importance_type": "xgboost_gain",
  "top_factors_gain": [{"feature": "...", "importance": 0.0}],  // 8 factors, gain
  "local_shap": {"expected_value": 0.0, "total_contribution": 0.0,
                 "predicted_price": 0.0, "top_positive": [], "top_negative": [],
                 "method": "TreeSHAP on the deployed model, evaluated at request time"},
  "error_band": {"segment_label": "...", "typical_error": 0.0, "n": 0, "method": "...", "note": "..."}
}
```

## 7. Canonical fixture (authoritative)

Input (the Phase-16 QA property):
```json
{"bedrooms":3,"bathrooms":2.0,"sqft_living":1910,"sqft_lot":7600,"floors":1.5,
 "waterfront":0,"view":0,"condition":3,"grade":7,"sqft_above":1560,"sqft_basement":0,
 "yr_built":1975,"yr_renovated":0,"zipcode":98065,"lat":47.5724,"long":-122.2300,
 "sqft_living15":1840,"sqft_lot15":7620,"sale_year":2015,"sale_quarter":3}
```
Expected: `predicted_price = 557597.38`; SHAP base = `538904.625`; error band uses the
holdout segment containing 557597.38.