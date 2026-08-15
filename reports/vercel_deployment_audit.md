# Vercel Deployment Audit (final)

Companion to `reports/vercel_readiness_audit.md` (phase 1, read-only). This report
documents what changed, the final architecture, every verification run, and the
remaining manual steps. The trained XGBoost champion, its feature logic, and its
verified metrics were **not** modified.

---

## 1. Original architecture

- One FastAPI app (`app/backend/main.py`) serving `POST /predict`, `GET /health`, and `GET /`
  (reads `app/frontend/index.html`).
- Frontend POSTed to same-origin `/predict`.
- A single fat `requirements.txt` (incl. torch/torchvision/jupyter/etc.).
- No Vercel wiring, no `api/`, no `.vercelignore`, no `vercel.json`.
- `src/config.py` ran `mkdir` on 6 dirs **at import time** (fails on Vercel's read-only lambda FS).
- 129.8 MB tracked bundle (embeddings, vision checkpoints, figures, notebooks).

## 2. Final architecture

```
vercel.json  ->  api/index.py (ASGI: StripApiPrefix)
                  └── app/backend/main.py   (single source of truth)
                        ├── POST /predict  -> src.inference.predict  -> tuned XGBoost
                        │                    src.inference.explain (TreeSHAP + error band)
                        ├── GET  /health
                        └── GET  /   (serves app/frontend/index.html)
Routes: /api/* , / , /predict , /health  -> api/index.py
```

All three targets (static frontend, FastAPI backend, tuned XGBoost) ship in **one** Vercel
deployment. Same-origin throughout → no CORS, no second backend domain.

## 3. Files added

| File | Purpose |
|---|---|
| `api/index.py` | Vercel ASGI entrypoint; strips `/api` prefix onto `app.backend.main.app`; delegates lifespan (model preload) |
| `vercel.json` | `@vercel/python` build of `api/index.py` + routes `/api/*`, `/`, `/predict`, `/health`; `maxLambdaSize 50mb` |
| `.vercelignore` | excludes research/heavy files from the bundle |
| `requirements-research.txt` | research/training deps, never deployed |
| `reports/vercel_readiness_audit.md` | Phase-1 audit |
| `reports/vercel_deployment_audit.md` | this report |

## 4. Files modified

| File | Change | Why |
|---|---|---|
| `src/config.py` | guarded import-time `mkdir` with `try/except OSError` | Vercel FS is read-only; inference never writes these dirs |
| `app/frontend/index.html` | `fetch('/predict') → fetch('/api/predict')` | same-origin `/api` path under one deployment |
| `app/run.py` | launcher `app.backend.main:app → api.index:app` | local dev mirrors deployed `/api` paths |
| `Dockerfile` | `CMD` → `uvicorn api.index:app` | consistent mounted `/api` paths in containers |
| `src/inference/explain.py` | holdout CSV path anchored to `PROJECT_ROOT` | CWD-independent path resolution |
| `requirements.txt` | trimmed to runtime-only | image size / cold start / build reliability |
| `tests/test_core.py` | +3 wrapper tests (`/api/health`, `/api/predict` parity, 404) | CI guards the deployment path |
| `.gitignore` | added `.vercel/` | ignore local Vercel state |
| `README.md` | added "Deploy to Vercel" section | document deployment + local-parity run |

## 5. Files removed

None. Research/provenance artifacts are retained (per non-negotiable rule 12); they are
excluded from the deployment bundle, not deleted from the repo.

## 6. Runtime dependencies (`requirements.txt`, verify)

`numpy, pandas, scipy, scikit-learn, xgboost, joblib, shap, fastapi, uvicorn, pydantic`
(+ `pytest`, `httpx` for CI). `torch/torchvision/Pillow/opencv/matplotlib/seaborn/lightgbm/
catboost/requests/python-dotenv/openpyxl/jupyter/ipykernel/nbformat` moved to
`requirements-research.txt`.

## 7. Deployment dependencies

`@vercel/python` (project build request to Vercel). No database, no external service.

## 8. Environment variables

None required by production inference (verified: no `os.environ` in the runtime import
chain). `.env` is gitignored and `.vercelignore`'d; `.env.example` stays a placeholder.

## 9. Model artifact

- `models/deployed/tabular_model.joblib` (0.81 MB) + `tabular_pipeline.joblib` — kept.
- Path is project-root anchored (`src.config.APP_MODELS_DIR`). No absolute Windows path.
- Loaded once per warm instance (startup lifespan preload). Checksum unchanged (QA baseline).
- 43 MB `models/deployed/vision_price.pt` (vision research) excluded from deploy via `.vercelignore`.

## 10. Bundle-size assessment

- Deploy-required set = **1.00 MB** (model 0.81 MB + HTML + source + tiny holdout CSV).
- `.vercelignore` removes ~128 MB of research/heavy material, keeping the lambda well under 50 MB.

## 11. Security assessment

- No secrets in source/model/config; none required at runtime. No CORS exposure.
- No database. `.env` never deployed. No mapbox token needed by production.
- Non-negotiable constraints respected: no retrain, no champion change, no vision model
  reintroduced, no Next.js, no framework workaround.

## 12. API assessment

- `POST /api/predict`: pydantic schema, numeric-bounds validation, human-readable 422 messages,
  deterministic schema, generic fallback (no stack trace), no path leak.
- `GET /api/health`: lightweight, no inference.
- Verified 404 for unknown `/api/*`.

## 13. Frontend assessment

- Same-origin `fetch('/api/predict', …)`; loading state, reset, example property, validation,
  result + SHAP + error-band rendering all present in the existing, unchanged UI.
- Root `/` serves the page; `index.html` contains the `/api/predict` call.

## 14. Fresh-clone result

An isolated 1.00 MB clean tree (only deploy files, no `data/`, `images/`, `.env`, notebooks,
embeddings) ran the wrapper with a clean import root:
`/health` 200, `/predict` 200 price **557597.38** (exact parity), SHAP + error band (n=644)
present, root HTML served with `/api/predict`. **PASS** — no hidden dependency on ignored/untracked files.

## 15. Vercel local simulation

`uvicorn api.index:app` local run (mirrors Vercel wiring):
`/api/health` 200 · `/api/predict` 200 price 557597.38, 8 factors, SHAP + band ·
`/` serves HTML (title + `/api/predict`) · `/predict` native 200 · invalid `lat=200` → 422
"Please enter a valid value for latitude." · unknown `/api/nope` → 404. **PASS.**

## 16. CI result

`.github/workflows/ci.yml` installs trimmed `requirements.txt` + pytest, runs `pytest -q`:
**29/29 passed** locally (CI-equivalent env is Python 3.11). QA script: **31/31 PASS**.
Smoke: **SMOKE OK**. Model/RMSE/R² unchanged; vision endpoints absent.

## 17. Regression test result

`pytest -q` → **29/29**. `phase16_qa.py` → **ALL QA CHECKS PASS (31/31)**. `smoke_api.py` →
**SMOKE OK**. API/offline parity intact (557597.38 == 557597.38). Champion checksum unchanged
(manifest hashes pass). No vision/satellite functionality reintroduced.

## Remaining limitations & manual steps

- **Real Vercel deploy not executed here** — pushing/deploying requires your Vercel account.
  The deployment path is fully simulated and fresh-clone-verified, but the first `vercel --prod`
  is a manual step. Expected to build cleanly from `api/index.py` + trimmed requirements.
- XGBoost loads the serialized champion with a benign "serialized with an older config"
  warning (non-fatal; parity is exact). If you later retrain, prefer `Booster.save_model`.
- Pin a Python runtime on Vercel (3.11) if you want to match CI exactly.

---

## VERCEL READINESS: 9/10

**Justification:** All five phase-1 blockers were resolved (import-time `mkdir`, fat
`requirements.txt`, no `vercel.json`, no `api/index.py`, oversized bundle). The complete
single-deployment path — static frontend → `/api/*` → FastAPI → feature engineering →
tuned XGBoost → JSON — is verified end-to-end: 29/29 tests, 31/31 QA, SMOKE OK, exact
offline/API parity, and a **fresh-clone run from a 1.00 MB bundle with zero local
dependencies**. Research artifacts stay in the repo but out of the bundle, and the champion
model is untouched.

The 1-point deduction: a real `vercel --prod` has not been executed in this environment
(no Vercel credentials), so the very last step remains manual; every preceding deployment
invariant is already proven. Once that push succeeds, readiness is 10/10.

**Blockers:** none remaining.
**Warnings:** benign xgboost load warning; Vercel Python version not pinned.
**Changes made:** see tables 3–4.
**Tests passed:** 29/29 pytest · 31/31 QA · SMOKE OK · fresh-clone PASS · Vercel local sim PASS.
**Remaining manual steps:** run `vercel` / `vercel --prod` with your account; (optional) pin Python 3.11.