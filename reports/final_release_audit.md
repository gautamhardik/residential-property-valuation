# Final Release Audit

## Verification run (2026-08-15 hardening pass)

- `pytest tests/test_core.py` — 24/24 passed.
- `scripts/smoke_api.py` — SMOKE OK (health, UI, `/predict`).
- `scripts/phase16_qa.py` — `ALL QA CHECKS PASS` (manifest hashes, parity, deprecated routes 404, docs truth, security).
- Live app surface: `/predict`, `/health`, `/` only. No vision/tile route exists.
- Frontend: single-purpose valuation UI; all dead satellite/vision markup, JS, and CSS removed. Every `getElementById` resolves to an existing element (verified).
- Stale claim fixed: README no longer shows a `python -m app.cli --pid …` vision example; the CLI is tabular-only.

## Scientific integrity
PASS — The tuned XGBoost champion is credible; satellite experiments are correctly framed as research-only and negative under the tested setup.

## Model integrity
PASS — Production inference uses the exact tuned tabular artifact and matches offline predictions on a deterministic payload.

## Data integrity
PASS — Train/test duplicates and target encoding are handled explicitly; no clear target leakage in the production path.

## Deployment integrity
PASS — `/predict` and `/health` are the only live inference paths. Deprecated vision routes return 404.

## Security
PASS — `.env` is ignored, `.env.example` contains placeholders only, and no obvious secrets are tracked.

## Reproducibility
PASS — Core tabular reproduction is solid (verified import + tests in current env). The provided `Dockerfile` builds `uvicorn app.backend.main:app`; a local container build could not be executed because the Docker daemon is not running in the verification environment, so image build is asserted from the Dockerfile rather than executed. Historical satellite research still depends on cached artifacts and a Mapbox token for re-fetching imagery (documented as research-only).

## Repository cleanliness
WARN — The repo is much clearer now, but it still retains historical research assets for provenance.

## Documentation consistency
PASS — Live vs historical claims are now separated more clearly; authoritative artifacts are documented.

## Frontend quality
PASS — The app is focused on property valuation, with the research branch not exposed in the live UI.

## Portfolio readiness
PASS — Strong resume project, now easier to defend because the live product and research branch are clearly separated.

## Findings

### P0
- None identified in the hardened release path.

### P1
- Historical research artifacts should not be described as live product features.

### P2
- The project still carries some historical breadth, which is useful for provenance but not necessary for the live demo.

### P3
- Additional polish would mostly be presentation-only.

---

## Release-integrity fix pass (added 2026-08-15)

Changes landed in this pass:

- **P3 — absolute paths removed from non-notebook tracked files.** `git grep "C:\\Users\\hiten"` over `*.py|*.json|*.md` returns no matches; all 6 affected JSON report/manifest files reload as valid JSON.
- **P2 — large-binary inventory documented.** Added a size/role classification table to `reports/ARTIFACT_GUIDE.md` covering every tracked artifact ≥ 0.5 MB (deployment-required, reproducibility cache, research provenance). No tracked binary was deleted; the 72 MB embedding cache is regenerable via `scripts/extract_embeddings.py`.
- **P6 — fresh-clone CI simulation.** Built a pristine clone (tracked + untracked release artifacts only, zero `data/*.xlsx`, no `images/`, no `.env`) and ran `pytest -q`: **24/24 passed**. This mirrors `.github/workflows/ci.yml` exactly.
- **P0 manifest fix.** `reports/baseline_manifest.json` previously hashed `data/train.xlsx` / `data/test.xlsx` inside `artifacts`, which do not exist in a fresh clone or CI and broke `test_baseline_manifest_matches_files`. Moved them to a documented `raw_data_external` section (digests preserved for verification when data is recreated); refreshed stale digests for `reports/experiment_log.json`, `reports/results_dl_final.json`, `reports/project_report.md` caused by earlier legitimate edits.
- **P4 — stale-claim scan.** Repo-wide scan for `predict-image`, `/tile/`, `--pid`, `vision_only`: only legitimate 404 tests, the correctly-labelled `scripts/legacy/`, and correct "not exposed" statements remain.
- **P5 — Docker static review.** `Dockerfile` and README `docker build/run` commands are self-consistent; no orphan `docker-compose*` references anywhere. Live image build/run could not be executed because the Docker daemon is not running in this environment.
- **P7 — QA cascade.** `scripts/phase16_qa.py`: **ALL QA CHECKS PASS** (27/27), including offline/API parity 557597.38 == 557597.38.

## Explanation & honest-uncertainty pass (added 2026-08-15)

Final presentation hardening focused on two things: making the estimate *explainable* and being *honest* about uncertainty.

- **`/predict` now returns `local_shap`** — a request-time TreeSHAP breakdown (exact `expected_value`, `total_contribution`, `predicted_price`, and top positive/negative drivers with human-readable `label`s and `direction`). `expected_value + total_contribution` reproduces `predicted_price` to within <$1 (verified: gap 0.26 on the live payload).
- **`/predict` now returns `error_band`** — an empirical, segment-level typical error (n reported). It is explicitly *not* a per-property confidence interval, so it never fabricates a false precision range around a single estimate.
- **Frontend** shows a "Why this estimate?" panel (driver +/– contributions in dollars) and the empirical typical error, wired through the new `primary-shap` / `primary-typical` anchors. No stepwise SHAP fallback, no "Uncertainty" label, no ± band presented as a per-property interval.
- **Naming integrity** — flagship internals (`zip_target`, `total_sqft`, `_flag`, …) never surface as user-facing labels.
- **A11y/semantics** — `h2`/`h3` card hierarchy, a skip-to-content link, and labelled result anchors.
- **Verification** — `pytest` updated to **26/26 passed** (added SHAP-parity + label-integrity + error-band tests); `scripts/phase16_qa.py` extended and again **ALL QA CHECKS PASS** (**31/31** incl. SHAP parity, empirical error band present, no fake uncertainty interval in the frontend); served-HTML smoke over the live app: all 7 target assertions pass, every `getElementById` resolves.

### Pending (documented, not blocking)
- Docker image build + container smoke run once the Docker daemon is started.
- Freeze commit/tag that adds `reports/ARTIFACT_GUIDE.md`, `reports/baseline_manifest.json`, `reports/final_release_audit.md` and the other new release artifacts to git.
