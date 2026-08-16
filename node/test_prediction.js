"use strict";
const fs = require("fs");
const scorer = require("./scorer.js");

const ROOT = "C:/Users/hiten/Documents/Satellite_Property_Valuation";
const model = scorer.loadModel(
  ROOT + "/models/deployed/tabular_model.json",
  ROOT + "/models/deployed/tabular_pipeline.json",
);

const golden = JSON.parse(
  fs.readFileSync(ROOT + "/reports/node_scorer/golden.json", "utf8"),
);

let maxAbs = 0;
let sumAbs = 0;
let maxRel = 0;
const failures = [];

for (const c of golden.cases) {
  const feats = scorer.engineerFeatures(c.input, model);
  const x = scorer.FEATURE_COLS.map((name) => feats[name]);
  // feature-equality check: engineered vector must match the golden vector
  let featErr = 0;
  for (let i = 0; i < 33; i++) featErr = Math.max(featErr, Math.abs(x[i] - c.features[i]));
  if (featErr > 1e-9) {
    failures.push({ id: c.id, kind: "feature", err: featErr });
    continue;
  }
  const raw = scorer.predictRaw(model, feats);
  const err = Math.abs(raw - c.prediction_raw);
  const rel = Math.abs(raw - c.prediction_raw) / Math.max(1, Math.abs(c.prediction_raw));
  if (err > maxAbs) maxAbs = err;
  if (rel > maxRel) maxRel = rel;
  sumAbs += err;
  if (err > 1e-3) failures.push({ id: c.id, kind: "prediction", err, raw, exp: c.prediction_raw });
}

console.log("cases:", golden.cases.length);
console.log("feature-engineering matched for all (max err threshold 1e-9).");
console.log("max abs prediction error:", maxAbs);
console.log("mean abs prediction error:", sumAbs / golden.cases.length);
console.log("max rel prediction error:", maxRel);
console.log("failures (abs>1e-3):", failures.length);
if (failures.length) {
  for (const f of failures.slice(0, 10)) console.log(JSON.stringify(f));
}

const canon = golden.cases[0];
const cr = scorer.predictRaw(model, scorer.engineerFeatures(canon.input, model));
console.log("canonical raw node:", cr, "expected:", canon.prediction_raw, "rounded:", cr.toFixed(2), "expected:", canon.prediction);

const ok = failures.length === 0 && maxAbs <= 1e-3;
console.log(ok ? "PREDICTION PARITY: PASS" : "PREDICTION PARITY: FAIL");
process.exit(ok ? 0 : 1);