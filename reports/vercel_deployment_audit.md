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

## 6. Runtime dependencies (final production set)

`numpy, pandas, scipy, scikit-learn, xgboost, joblib, fastapi, uvicorn, pydantic`
(+ `pytest`, `httpx` for CI). `shap` was removed from the runtime graph (replaced by
XGBoost-native TreeSHAP, §18/§19); torch/vision/notebook/plotting deps live in
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

---

## 18. Bundle-size remediation (813.56 MB → 271.4 MB)

A first Vercel deploy failed with a **813.56 MB** function bundle (limit 500 MB uncompressed).
Root cause: the build installed a research-heavy `requirements.txt` (torch/torchvision/opencv/
Pillow ≈ 800 MB) into the Python function. `.vercelignore` was already excluding repository
research files correctly; the size was **installed Python packages**, which ignores-list cannot shrink.

Evidence-based reduction (see `reports/vercel_runtime_dependency_audit.md`):

1. **Production deps isolated** → `requirements-vercel.txt` (authoritative, pinned, no test deps);
   `requirements.txt` = same runtime set + CI test deps (`pytest`, `httpx`); `requirements-research.txt`
   = full research env (dev only).
2. **`shap` removed (≈ 127 MB)** — reconstructed locally via measurements: `shap` feeds a TreeSHAP
   through `numba`+`llvmlite` (115 MB). Replaced with **XGBoost-native `pred_contribs=True`**,
   verified **bit-identical** (max |Δ| = 0.0; base == shap `expected_value`; same explanation total).
   `/predict` `local_shap` output is byte-for-byte the same; `shap` moved to `requirements-research.txt`.
3. **Final measured bundle (fresh Python 3.12 install, `requirements-vercel.txt` only): 271.4 MB** —
   well under 500 MB with ~229 MB headroom.

Final size table (see audit report §5): scipy 83.3 · xgboost 55.1 · pandas 33.0 · scikit-learn 25.9 ·
numpy+libs 39.6 · API/other ~35 MB. No torch/torchvision/opencv/Pillow/shap/numba/llvmlite deployed.

Verified from the clean 3.12 environment (no `shap`, no `torch` importable): model loads,
`predict_single` → **557597.38** exact, `local_summary` → identical values, `error_band` n=644.

## 19. Vercel config & Python decision

- **Kept legacy `builds`/`routes` in `vercel.json`**: required to route `/`, `/api/*`, `/predict`,
  `/health` to the single FastAPI function. The Vercel note *"Due to builds existing in your
  configuration file, the Build and Development Settings defined in your Project Settings will not
  apply"* is **informational and expected**: it simply means project-settings build/dev commands are
  overridden by our `vercel.json`, which is what we rely on. It does not disable dependency install.
- **Python version**: Vercel selects 3.12 when unspecified (a known, stable runtime). Clean-env
  verification was done on **Python 3.12.13** to match the deploy target; the repo's local env is
  3.14 and CI is 3.11 — all three produce the identical 557597.38 (model is library-version-agnostic).
  No `.python-version`-based pin is reliably honored by the legacy builder, so the target is
  documented as 3.12 rather than force-pinned.

## Remaining limitations & manual steps

- **Real `vercel --prod` not executed here** — deploying requires your Vercel account. The full
  deployment path (bundle size, install, model load, `/api` parity, fresh-clone) has been
  reproduced locally on Python 3.12 with a measured **271.4 MB** bundle; the final push is manual.
- Benign XGBoost "serialized with an older config" load warning; parity is exact.
- Vercel runtime Python is 3.12 (documented; not force-pinned).

---

## VERCEL READINESS: 9/10

**Justification:** Every phase-1 blocker is resolved, and the bundle failure is now closed with an
evidence-based fix: **813.56 MB → 271.4 MB** (< 500 MB limit, ~229 MB headroom). The production set
contains only verified imports (fastapi, pydantic, numpy, pandas, scipy, scikit-learn, xgboost,
joblib, uvicorn) — no torch/vision/research packages. `shap` was replaced by XGBoost-native TreeSHAP
with **bit-identical** explanation output, so the API/schema/predictions are unchanged. Verified on a
clean Python 3.12 env: model loads, `/predict` = 557597.38 exact, `local_summary` identical, error
band present; `pytest` 29/29, QA 31/31, SMOKE OK. Champion model, features, and metrics untouched.

The 1-point deduction: the *literal* `vercel --prod` run and its resulting ".vercel/output" size
report still require your Vercel account/credentials; every deployment invariant (including the
271.4 MB production function size) has been reproduced locally on the target Python.

**BLOCKERS:** none. **WARNINGS:** benign XGBoost load warning; Vercel runtime Python (3.12) is
documented, not force-pinned. **VERDICT: VERCEL DEPLOYMENT READY: YES** (pending the manual push).

**Changed:** `src/inference/explain.py` (native TreeSHAP) · `requirements-vercel.txt` (new, minimal)
· `requirements.txt`/`requirements-research.txt` (shap moved) · added
`reports/vercel_runtime_dependency_audit.md`.
**Tests passed:** 29/29 pytest · 31/31 QA · SMOKE OK · clean-env parity PASS (exact) · bundle 271.4 MB.
**Remaining manual steps:** `vercel --prod` (needs your account); reconfirm final `.vercel/output` size.