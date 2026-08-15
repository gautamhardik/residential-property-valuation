# Vercel Readiness Audit (Phase 1 — read-only)

Date: 2026-08-15. This audit was performed **before** any deployment changes and
informs the phases that follow. Nothing in this document implies a change was made.

Legend: **PASS** = production-safe as-is · **WARNING** = needs attention / documented risk ·
**BLOCKER** = will fail or break production if not fixed.

---

## 1. Application entrypoints

| Item | Finding | Verdict |
|---|---|---|
| Product app | `app/backend/main.py` → FastAPI `app` | PASS |
| Local dev launcher | `app/run.py` (free-port uvicorn launcher) uses `app.backend.main:app` | WARNING (see frontend path below) |
| Docker launcher | `Dockerfile` CMD uses `app.backend.main:app` | WARNING |
| CLI inference | `app/cli.py` → `src.inference.predict.predict_single` (no HTTP) | PASS |

## 2. Frontend entrypoint & API call path

- `app/frontend/index.html` is a single self-contained page (inline CSS/JS, fonts via CDN).
- It is served by the FastAPI `GET /` route reading the file from `ROOT/app/frontend/index.html`.
- The form posts to `fetch('/predict', …)` — already **same-origin relative** (no host/port). PASS.
- However Vercel Python functions live under `/api/*`. The frontend must call the
  API under the `/api` prefix on Vercel. **PASS on relative-path principle, WARNING on path schema.**

## 3. Production inference path (tabulated from imports)

```
app.backend.main -> src.inference.predict.predict_single
                 -> src.inference.artifacts (load_tabular_artifacts)
                 -> src.features.build_features (_engineer)
                 -> src.config, src.utils
app.backend.main -> src.inference.explain (local_summary, error_band_for)  [shap, pandas]
```
No training/research module is imported during request handling (`torch`, `notebooks`, etc. are not in the runtime chain).

## 4. Model artifact

| Item | Finding | Verdict |
|---|---|---|
| Production model | `models/deployed/tabular_model.joblib` (0.81 MB, tuned XGBoost) | PASS |
| Pipeline | `models/deployed/tabular_pipeline.joblib` (zip encoder, global mean, feature cols, model_type) | PASS |
| Path resolution | `src/inference/artifacts.py` uses `src.config.APP_MODELS_DIR` (project-root anchored) — no absolute Windows path | PASS |
| Loading | Cached once at startup (`preload_tabular_artifacts`); per-request loads avoided | PASS |
| Champion unchanged | No retraining; RMSE $103.8K / R² 0.9205 preserved | PASS |

## 5. Startup / import-time side effects

| Item | Finding | Verdict |
|---|---|---|
| `src/config.py` runs `.mkdir(parents=True, exist_ok=True)` for 6 dirs **at import time** | Vercel lambda filesystem is read-only (RW only under `/tmp`); the unguarded `mkdir` would raise `OSError` on import → the whole deployment would crash | **BLOCKER** |

## 6. Environment variables / secrets

| Item | Finding | Verdict |
|---|---|---|
| `.env` | present locally, gitignored (`!` checked) | PASS |
| `.env.example` | placeholder-only Mapbox token | PASS |
| Runtime env needs | Production inference/API reads **no** environment variables (no `os.environ` in runtime chain) | PASS |
| Hardcoded secrets | none in source/model/config | PASS |
| Stray `models/deployed/vision_price.pt` (43 MB) | A **vision** checkpoint committed inside the production `deployed/` dir | WARNING (exclude from deploy; body copy is vision research) |

## 7. Filesystem assumptions / path handling

| Item | Finding | Verdict |
|---|---|---|
| Absolute Windows paths | Repo hygiene pass earlier removed them; `git grep "C:\\Users"` → none in runtime | PASS |
| Relative CSV for error band | `src/inference/explain.py` reads `reports/val_predictions_best_tabular.csv` by **relative** path (CWD-dependent) | WARNING (anchor to project root) |
| CWD assumption | Vercel function CWD is the project root, so relative works today | WARNING |

## 8. Runtime dependency set (what the API actually imports)

Imports traced transitively: `numpy, pandas, scipy, scikit-learn (defensive via joblib/shap), xgboost, joblib, shap, fastapi, pydantic, uvicorn (local only)`.

`requirements.txt` currently also installs (NOT needed at runtime): `torch, torchvision, Pillow, opencv-python, matplotlib, seaborn, requests, python-dotenv, httpx, lightgbm, catboost, openpyxl, jupyter, ipykernel, nbformat`. Installing these on Vercel bloats the image, slows cold start, and risks build failure.

| Item | Finding | Verdict |
|---|---|---|
| Split runtime vs training deps | Not split — one fat `requirements.txt` | **BLOCKER** (Vercel image size / cold start risk) |

## 9. Bundle / deployment size

Tracked repository ≈ **129.8 MB**. The true runtime need is ≈ **1.0 MB**.
Large tracked, deploy-irrelevant items:
- `preprocessed/*.npz` (~72 MB embeddings) + `*.pkl`
- `models/deployed/vision_price.pt` (43 MB) + `models/vision/**`
- `reports/figures/*.png` (~4 MB), notebooks, predictions, scripts, tests
- `images/` (~422 MB) and `data/*.xlsx` (~2.5 MB) are gitignored → absent from a fresh clone.

| Item | Finding | Verdict |
|---|---|---|
| Lambda size limit (50 MB default) | Exceeded without exclusions | **BLOCKER** |
| No `.vercelignore` | Does not exist | **BLOCKER** |

## 10. Vercel configuration

| Item | Finding | Verdict |
|---|---|---|
| `vercel.json` | Does not exist | **BLOCKER** (no API route / function wiring) |
| `api/index.py` | Does not exist | **BLOCKER** |

## 11. CORS

- The frontend and API are same-origin (served by one deployment) → no CORS needed. No CORS middleware exists. PASS.

## 12. Database / background / long-running

- No database (correct). No background workers. Single cold-called request path. PASS.

## 13. Request-time cost (explainability)

- `local_summary` builds a TreeSHAP `TreeExplainer` once (lazy, cached) and runs it per request.
- `error_band_for` reads + buckets an 0.09 MB holdout CSV lazily once.
- Both are wrapped so explanation failures degrade to `None` and never block the prediction. PASS (performance budget noted in Phase 13).

## 14. API hardening

- `POST /predict` has a pydantic schema, numeric-bounds validation, human-readable messages, generic error fallback (no stack trace leak), deterministic output. PASS.
- `GET /health` is lightweight (no inference). PASS.

## 15. CI

- `.github/workflows/ci.yml` installs `requirements.txt` + pytest, runs `pytest -q`.
- Tests import only runtime modules (+pandas/fastapi/pytest) → a trimmed `requirements.txt` keeps CI green. PASS with the planned split.
- No test depends on `data/*.xlsx` (gitignored) or `images/`. PASS.

---

## Roll-up

- BLOCKER: (a) import-time `mkdir` side effect in `src/config.py`; (b) fat `requirements.txt`; (c) no `vercel.json`; (d) no `api/index.py`; (e) no `.vercelignore` + 129.8 MB bundle > 50 MB.
- WARNING: (a) frontend must call API under `/api`; (b) `app/run.py` + `Dockerfile` launcher module; (c) relative holdout-CSV path in `explain.py`; (d) 43 MB vision `.pt` inside `models/deployed`.
- PASS: model + pipeline are small, anchored, unchanged; no env/secrets in runtime; no DB; same-origin; CORS-free; CI independent of datasets; API already hardened.

These findings are implemented across Phases 2–15; see `reports/vercel_deployment_audit.md`.