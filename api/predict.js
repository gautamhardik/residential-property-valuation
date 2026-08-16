"use strict";
/**
 * Vercel serverless function: POST /api/predict
 *
 * Production Node scorer. Reproduces the deployed XGBoost tabular model exactly
 * (see node/scorer.js) with no Python ML dependencies, and returns a response
 * byte-identical to the legacy Python FastAPI /predict handler.
 */
const path = require("path");
const { loadModel, predictContract, validateInput } = require("../node/scorer.js");

const MODEL_DIR = path.join(__dirname, "..", "models", "deployed");
let _model = null;
function getModel() {
  if (!_model) {
    _model = loadModel(
      path.join(MODEL_DIR, "tabular_model.json"),
      path.join(MODEL_DIR, "tabular_pipeline.json"),
      path.join(MODEL_DIR, "tabular_meta.json"),
    );
  }
  return _model;
}

function parseBody(req) {
  const b = req.body;
  if (b !== undefined && b !== null) {
    if (Buffer.isBuffer(b)) return JSON.parse(b.toString("utf8") || "{}");
    if (typeof b === "string") return JSON.parse(b || "{}");
    if (typeof b === "object") return b;
  }
  return {};
}

module.exports = (req, res) => {
  if (req.method !== "POST") {
    res.status(404).json({ error: "Not Found" });
    return;
  }
  let body;
  try {
    body = parseBody(req);
  } catch (e) {
    res.status(422).json({ error: "Invalid JSON body." });
    return;
  }
  const err = validateInput(body);
  if (err) {
    res.status(422).json({ error: err });
    return;
  }
  try {
    const result = predictContract(getModel(), body);
    res.status(200).json(result);
  } catch (e) {
    res.status(422).json({ error: "Unable to reach the valuation service. Please try again." });
  }
};