# Package Guide

## What this repo contains

- `app/` — production FastAPI app and frontend demo
- `src/` — importable data, feature, model, and inference code
- `scripts/` — reproducible research and training phases
- `reports/` — canonical results, figures, and audit artifacts
- `models/deployed/` — final production artifacts
- `archive/historical_research/` — preserved non-production context

## Canonical production path

- `POST /predict`
- tuned XGBoost
- engineered tabular + geospatial features

## Canonical research path

- satellite coverage analysis
- frozen and trained ResNet18 experiments
- final negative conclusion for imagery under the tested setup

## Reproduction

Use `README.md` and `reports/ARTIFACT_GUIDE.md` as the source of truth for the
final release artifacts.
