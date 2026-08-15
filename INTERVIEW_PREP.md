# Interview Prep

## Strong answers

- Why XGBoost? It generalized better than the alternatives when temporal and spatial holdouts were considered.
- Why did satellite imagery not win? The tested coverage, resolution, and ResNet18 setup did not add enough signal.
- Is `zip_target` leakage? No; it is fit on training folds only.
- What is the production model? Tuned XGBoost on engineered tabular features.
- Is the vision branch deployed? No; it is archived research only.

## Questions to be ready for

- Why not LightGBM/CatBoost?
- Why is spatial R² lower?
- Why only 13.6% image coverage?
- How do SHAP and Grad-CAM differ from causality?
- How do you know API inference matches offline inference?
- What would you improve first?

## Key caution

Never claim satellite imagery is useless in general. The defensible claim is
that the tested representation did not add enough incremental signal for this
dataset and setup.
