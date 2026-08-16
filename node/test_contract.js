"use strict";
const fs = require("fs");
const path = require("path");
const scorer = require("./scorer.js");

const ROOT = path.resolve(__dirname, "..");
const model = scorer.loadModel(
  path.join(ROOT, "models/deployed/tabular_model.json"),
  path.join(ROOT, "models/deployed/tabular_pipeline.json"),
  path.join(ROOT, "models/deployed/tabular_meta.json"),
);

const canonical = {
  bedrooms: 3, bathrooms: 2.0, sqft_living: 1910, sqft_lot: 7600, floors: 1.5,
  waterfront: 0, view: 0, condition: 3, grade: 7, sqft_above: 1560, sqft_basement: 0,
  yr_built: 1975, yr_renovated: 0, zipcode: 98065, lat: 47.5724, long: -122.2300,
  sqft_living15: 1840, sqft_lot15: 7620, sale_year: 2015, sale_quarter: 3,
};

const res = scorer.predictContract(model, canonical);
console.log(JSON.stringify(res, null, 2));

// Expected values from the Python oracle (contract.py).
let failures = 0;
function eq(name, got, exp) {
  const ok = Object.is(got, exp);
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: got=${got} expected=${exp}`);
}
eq("predicted_price", res.predicted_price, 557597.38);
eq("model_role", res.model_role, "primary");
eq("status", res.status, "production");
eq("importance_type", res.importance_type, "xgboost_gain");
eq("top_factors_gain[0].feature", res.top_factors_gain[0]?.feature, "grade");
eq("top_factors_gain[0].importance", res.top_factors_gain[0]?.importance, 0.26828283071517944);
eq("local_shap.expected_value", res.local_shap?.expected_value, 538904.62);
eq("local_shap.total_contribution", res.local_shap?.total_contribution, 18692.5);
eq("local_shap.predicted_price", res.local_shap?.predicted_price, 557597.12);
eq("local_shap.top_positive[0].feature", res.local_shap?.top_positive?.[0]?.feature, "dist_to_center_km");
eq("local_shap.top_negative[0].feature", res.local_shap?.top_negative?.[0]?.feature, "grade");
eq("error_band.segment_label", res.error_band?.segment_label, "$512K\u2013$686K");
eq("error_band.typical_error", res.error_band?.typical_error, 46637.89);
eq("error_band.n", res.error_band?.n, 644);

// validation checks
const bad = scorer.predictContract(model, { ...canonical, bedrooms: 99 });
console.log("TODO bad-range expected to 422 (validated in handler, not here)");
const missingErr = scorer.validateInput({ ...canonical, sqft_living: undefined });
console.log("missing-field error:", JSON.stringify(missingErr));
eq("validateInput missing", missingErr !== null, true);
const outOfRange = scorer.validateInput({ ...canonical, bedrooms: 99 });
eq("validateInput out-of-range", outOfRange !== null, true);

console.log(failures === 0 ? "CONTRACT: PASS" : `CONTRACT: FAIL (${failures})`);
process.exit(failures === 0 ? 0 : 1);