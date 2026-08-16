"use strict";
const fs = require("fs");
const scorer = require("./scorer.js");

const ROOT = "C:/Users/hiten/Documents/Satellite_Property_Valuation";
const model = scorer.loadModel(
  ROOT + "/models/deployed/tabular_model.json",
  ROOT + "/models/deployed/tabular_pipeline.json",
);
const golden = JSON.parse(fs.readFileSync(ROOT + "/reports/node_scorer/golden.json", "utf8"));

let maxContrib = 0;
let maxBase = 0;
let maxRecon = 0;
const failures = [];
const t0 = Date.now();
let roundMismatch = 0;
let rankMismatch = 0;
const rankMismatchCases = [];
for (const c of golden.cases) {
  const feats = scorer.engineerFeatures(c.input, model);
  const r = scorer.predictContribs(model, feats);
  for (let i = 0; i < 33; i++) {
    maxContrib = Math.max(maxContrib, Math.abs(r.contribs[i] - c.shap_contribs[i]));
    if (Math.round(r.contribs[i] * 100) / 100 !== Math.round(c.shap_contribs[i] * 100) / 100) {
      roundMismatch++;
    }
  }
  maxBase = Math.max(maxBase, Math.abs(r.base - c.shap_base));
  const recon = Math.abs(r.base + r.contribs.reduce((a, b) => a + b, 0) - c.prediction_raw);
  maxRecon = Math.max(maxRecon, recon);

  // ranking parity: top-5 positive and top-5 negative indices by contribution
  const posIdx = (arr) =>
    arr
      .map((v, i) => [i, v])
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([i]) => i)
      .join(",");
  const negIdx = (arr) =>
    arr
      .map((v, i) => [i, v])
      .filter(([, v]) => v < 0)
      .sort((a, b) => a[1] - b[1])
      .slice(0, 5)
      .map(([i]) => i)
      .join(",");
  if (posIdx(r.contribs) !== posIdx(c.shap_contribs) || negIdx(r.contribs) !== negIdx(c.shap_contribs)) {
    rankMismatch++;
    rankMismatchCases.push(c.id);
  }
if (Math.abs(r.base - c.shap_base) > 1e-5) {
    failures.push({ id: c.id, kind: "base", diff: Math.abs(r.base - c.shap_base) });
  }
}
const ms = Date.now() - t0;
console.log("cases:", golden.cases.length, "elapsed:", ms + "ms");
console.log("max raw contrib error:", maxContrib);
console.log("max base error:", maxBase);
console.log("max reconstruct error (pred_contribs vs predict):", maxRecon);
console.log("round-to-cents mismatches:", roundMismatch, "/", golden.cases.length * 33);
console.log("top-factor ranking mismatches:", rankMismatch, "cases", rankMismatchCases.slice(0, 10));
console.log("base failures:", failures.length);

const canon = golden.cases[0];
const cr = scorer.predictContribs(model, scorer.engineerFeatures(canon.input, model));
console.log("canonical base node:", cr.base, "expected:", canon.shap_base);
console.log("canonical contribs[15] (lat):", cr.contribs[15], "expected:", canon.shap_contribs[15]);

const ok = failures.length === 0 && roundMismatch === 0 && rankMismatch === 0 && maxBase <= 1e-5;
console.log(ok ? "SHAP DISPLAY PARITY: PASS" : "SHAP DISPLAY PARITY: FAIL");
process.exit(ok ? 0 : 1);