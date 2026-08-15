# Legacy scripts (historical, superseded)

These scripts were part of earlier iterations of the project. They are **not**
part of the canonical reproduction pipeline (see `README.md` "Quick start") and
are preserved purely for provenance. They are not maintained and may not run
unchanged against the current package layout.

| Script | Superseded by |
|---|---|
| `finetune_cnn.py` | `scripts/phase2_experiment_a.py` (Experiment A) + `scripts/phase8_serialize.py` (Experiment B) |
| `image_subset_compare.py` | `scripts/phase3_4_eval_gate.py` (fair E4/E5/A1/A2 eval) |
| `predict_image.py` | Historical vision-only CLI / research artifact (not part of the live app) |
| `gradcam.py` | `scripts/phase12_gradcam.py` (Grad-CAM on the actual evaluated model) |

Treat anything under this directory as historical evidence, not living code.