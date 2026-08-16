# Vercel Runtime Dependency Audit

Follow-up to the 813.56 MB Vercel bundle failure. This audit traces the *actual*
production import graph and classifies every dependency, so the reduction is
evidence-based — no packages removed by guessing.

Legend — A: REQUIRED at runtime · B: tests · C: training · D: research/vision · E: transitive · F: unused

---

## 1. Classification of current requirements (pre-fix status)

| Dependency | Class | Imported by (production path unless noted) |
|---|---|---|
| `numpy` | **A** | `src/utils`, `src/features/build_features`, `src/inference/predict`, `src/inference/explain` |
| `pandas` | **A** | `src/features/build_features`, `src/inference/predict`, `src/inference/explain` |
| `scipy` | **A/E** | transitive of scikit-learn / xgboost (not imported directly) |
| `scikit-learn` | **A/E** | `xgboost.sklearn.XGBRegressor` base (the artifact is an XGBRegressor) |
| `xgboost` | **A** | `src/inference/explain` (native TreeSHAP `pred_contribs`) |
| `joblib` | **A** | `src/inference/artifacts` (loads the persisted champion) |
| `fastapi` | **A** | `app/backend/main.py` |
| `pydantic` | **A** | `app/backend/main.py` (request schema) |
| `uvicorn` | **A/B** | local run only (Vercel supplies its own server) |
| `httpx` | **B** | `fastapi.testclient` (tests only) |
| `pytest` | **B** | tests |
| `shap` | **D (research)** | was `src/inference/explain` — REPLACED by XGBoost-native TreeSHAP (see §2) |
| `torch` | **D** | research/vision only — not in production graph |
| `torchvision` | **D** | research/vision only |
| `Pillow` | **D** | research/vision only |
| `opencv-python` | **D** | research/vision only |
| `matplotlib` | **D** | research/report figures |
| `seaborn` | **D** | research/report figures |
| `lightgbm` | **C/D** | training baselines |
| `catboost` | **C/D** | training baselines |
| `requests` | **D** | Mapbox tile download (research) |
| `python-dotenv` | **D** | Mapbox tile download (research) |
| `openpyxl` | **C** | training (xlsx loading) |
| `jupyter` / `ipykernel` / `nbformat` | **D** | notebooks |

## 2. The cause of the 813.56 MB bundle

- The failing build installed a **research-heavy `requirements.txt`** (torch + torchvision +
  opencv + Pillow ≈ 800 MB) into the Python function.
- `.vercelignore` correctly excluded repository files (73 research/vision files); it cannot
  shrink installed **Python packages**, which is what dominated the bundle.
- The single largest *runtime-justifiable* surprise was **`shap`**, which pulls
  `llvmlite` (115 MB) + `numba` (11 MB) only for TreeSHAP acceleration.

## 3. Production dependency reduction (evidence-based)

### `shap` removed (≈ 127 MB) — replaced by XGBoost-native TreeSHAP
- Production `local_summary` now uses `Booster.predict(DMatrix, pred_contribs=True)`.
- **Verified bit-identical:** max |shap_contribution − native_contribution| = **0.0**;
  base value 538904.625 == shap `expected_value`; explanation total and per-feature
  contributions match exactly for the canonical property.
- Result: identical `/predict` `local_shap` output with **no API/schema change**, while
  dropping `shap`, `numba`, `llvmlite` (≈ 127 MB), `cloudpickle`, `slicer`, `tqdm` (transitive).
- `shap` remains in `requirements-research.txt` for `scripts/shap_analysis.py`.

### Full removed-from-runtime set
`torch`, `torchvision`, `Pillow`, `opencv-python`, `matplotlib`, `seaborn`,
`lightgbm`, `catboost`, `requests`, `python-dotenv`, `openpyxl`, `jupyter`,
`ipykernel`, `nbformat`, `shap` (+ transitive `numba`/`llvmlite`).

### Production files & what each installs
- `requirements-vercel.txt` — **authoritative minimal** set (no test deps), installed on Vercel.
- `requirements.txt` — minimal runtime + CI test deps (pytest/httpx); read by CI and by
  `@vercel/python` for the Vercel install step.
- `requirements-research.txt` — full research/training env (dev machine only).

## 4. Production import graph (traced)

```
api/index.py ──────────────────────────────────────────────► Vercel `app`
  └─ app/backend.main
       ├─ src.inference.predict ─► src.inference.artifacts ─► joblib
       │                            src.features.build_features
       └─ src.inference.explain  ─► src.inference.artifacts ─► joblib
                                      src.features.build_features
                                      xgboost (native TreeSHAP)
All: numpy, pandas, scikit-learn (transitive), scipy (transitive), xgboost, joblib
API: fastapi, pydantic, uvicorn(local)
```
**No** torch / torchvision / transformers / opencv / Pillow / vision / satellite /
notebook / training / image-processing code is reachable from production.

## 5. Final sizes (measured, Python 3.12, fresh install of `requirements-vercel.txt`)

| Package | MB | Required? | Reason |
|---|---|---|---|
| scipy (.libs incl.) | 83.3 | Yes (E) | transitive of scipy/scikit-learn |
| xgboost | 55.1 | Yes | production model runtime |
| pandas | 33.0 | Yes | feature engineering |
| scikit-learn | 25.9 | Yes (E) | XGBRegressor base |
| numpy (+libs) | 39.6 | Yes | core |
| other (fastapi, pydantic, joblib…) | ~35 | Yes | API + model I/O |
| ~~shap / numba / llvmlite~~ | ~~127~~ | Removed | replaced by native TreeSHAP |
| **Total** | **271.4 MB** | | (was 401 MB incl. shap; 813.56 MB pre-trim) |

## 6. Verification

- Clean Python 3.12 venv, `requirements-vercel.txt` only: `shap`/`torch` **not importable**,
  model loads, `predict_single` → **557597.38** (exact parity), `local_summary` → identical
  SHAP values, `error_band_for` → n=644, feature importance unchanged.
- Local regression: `pytest` **29/29**; `phase16_qa.py` **31/31 PASS**. Champion untouched.