"use strict";
/**
 * Dependency-free Node.js reproduction of the production XGBoost tabular scorer.
 *
 * Mirrors `src/inference/predict.py` + `src/features/build_features.py` and the
 * tree arithmetic of `xgb.Booster.predict`. Loads two exported artifacts:
 *   - tabular_model.json     (XGBoost native model export, raw array form)
 *   - tabular_pipeline.json  (feature_cols, global_mean, zip_enc)
 *
 * Prediction parity with Python is guaranteed by:
 *   - identical feature engineering (IEEE-754 float64)
 *   - float32 split comparison via Math.fround
 *   - sequential float32 accumulation matching xgboost's prediction
 */

const fs = require("fs");

// Exponential of the model (reg:squarederror) -> identity; kept for clarity.
// 1e-6 denominators and the Seattle center must match Python exactly.
const EPS = 1e-6;
const SEATTLE_LAT = 47.6062;
const SEATTLE_LON = -122.3321;
const EARTH_R = 6371.0;

const FEATURE_COLS = [
  "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
  "waterfront", "view", "condition", "grade", "sqft_above",
  "sqft_basement", "yr_built", "yr_renovated", "zipcode",
  "lat", "long", "sqft_living15", "sqft_lot15", "sale_year", "sale_quarter",
  "age", "renovated", "renovation_age",
  "total_sqft", "basement_frac", "above_frac",
  "living_per_bedroom", "lot_living_ratio", "has_basement",
  "lat_long_interaction", "dist_to_center_km",
  "zip_freq", "zip_target",
];

// Required fields relative to input defaults (mirror predict.to_row / DEFAULT_VALUES).
const DEFAULT_VALUES = {
  waterfront: 0, view: 0, condition: 3, grade: 7,
  sqft_basement: 0, yr_renovated: 0, sale_year: 2015, sale_quarter: 3,
};

// Human-readable labels (mirror src/inference/explain.py FEATURE_LABELS).
const FEATURE_LABELS = {
  bedrooms: "Bedrooms", bathrooms: "Bathrooms", sqft_living: "Living area",
  sqft_lot: "Lot size", floors: "Floors", waterfront: "Waterfront",
  view: "View quality", condition: "Condition", grade: "Grade",
  sqft_above: "Above-ground area", sqft_basement: "Basement area",
  yr_built: "Year built", yr_renovated: "Year renovated", zipcode: "ZIP code",
  lat: "Latitude", long: "Longitude", sqft_living15: "Avg. nearby living area",
  sqft_lot15: "Avg. nearby lot size", sale_year: "Sale year",
  sale_quarter: "Sale quarter", age: "Property age", renovated: "Recently renovated",
  renovation_age: "Years since renovation", total_sqft: "Total built area",
  basement_frac: "Basement share", above_frac: "Above-ground share",
  living_per_bedroom: "Living area per bedroom", lot_living_ratio: "Lot-to-living ratio",
  has_basement: "Has basement", lat_long_interaction: "Location interaction",
  dist_to_center_km: "Distance to city center", zip_freq: "Neighborhood frequency",
  zip_target: "Neighborhood location signal",
};

function f32(x) {
  return Math.fround(x);
}

function haversineKm(latA, lonA, latB, lonB) {
  const toRad = (d) => (d * Math.PI) / 180;
  const aL = toRad(latA), aN = toRad(lonA);
  const bL = toRad(latB), bN = toRad(lonB);
  const dl = bL - aL;
  const dn = bN - aN;
  const a =
    Math.pow(Math.sin(dl / 2), 2) +
    Math.cos(aL) * Math.cos(bL) * Math.pow(Math.sin(dn / 2), 2);
  return EARTH_R * 2 * Math.asin(Math.sqrt(a));
}

/**
 * Engineered 33-feature vector (float64) for a single property.
 * Mirrors `_engineer` + `zip_target` for a single-row frame.
 */
function engineerFeatures(input, model) {
  const saleYear = input.sale_year !== undefined ? input.sale_year : 2015;
  const f = {};
  for (const c of FEATURE_COLS) f[c] = null;

  f.bedrooms = input.bedrooms;
  f.bathrooms = input.bathrooms;
  f.sqft_living = input.sqft_living;
  f.sqft_lot = input.sqft_lot;
  f.floors = input.floors;
  f.waterfront = input.waterfront !== undefined ? input.waterfront : 0;
  f.view = input.view !== undefined ? input.view : 0;
  f.condition = input.condition !== undefined ? input.condition : 3;
  f.grade = input.grade !== undefined ? input.grade : 7;
  f.sqft_above = input.sqft_above;
  f.sqft_basement = input.sqft_basement !== undefined ? input.sqft_basement : 0;
  f.yr_built = input.yr_built;
  f.yr_renovated = input.yr_renovated !== undefined ? input.yr_renovated : 0;
  f.zipcode = input.zipcode;
  f.lat = input.lat;
  f.long = input.long;
  f.sqft_living15 = input.sqft_living15;
  f.sqft_lot15 = input.sqft_lot15;
  f.sale_year = saleYear;
  f.sale_quarter = input.sale_quarter !== undefined ? input.sale_quarter : 3;

  const ref = saleYear;
  f.age = Math.max(ref - f.yr_built, 0);
  f.renovated = f.yr_renovated > 0 ? 1 : 0;
  f.renovation_age = f.renovated === 1 ? Math.max(ref - f.yr_renovated, 0) : 0;
  f.total_sqft = f.sqft_living + f.sqft_basement;
  f.basement_frac = f.sqft_basement / (f.sqft_living + EPS);
  f.above_frac = f.sqft_above / (f.sqft_living + EPS);
  f.living_per_bedroom = f.sqft_living / (f.bedrooms + 1.0);
  f.lot_living_ratio = f.sqft_lot / (f.sqft_living + EPS);
  f.has_basement = f.sqft_basement > 0 ? 1 : 0;
  f.lat_long_interaction = f.lat * f.long;
  f.dist_to_center_km = haversineKm(f.lat, f.long, SEATTLE_LAT, SEATTLE_LON);
  f.zip_freq = 1.0; // single-row frame: value_counts(normalize=True) == 1.0
  if (model) {
    const z = String(f.zipcode);
    f.zip_target = model.zipEnc[z] !== undefined ? model.zipEnc[z] : model.globalMean;
  } else {
    f.zip_target = model != null ? model.globalMean : 0;
  }
  return f;
}

/**
 * Load the exported model + pipeline into a compact in-memory form.
 * Returns { featureCols, featureIndex, baseScore, globalMean, zipEnc, trees }.
 */
function loadModel(modelJsonPath, pipelineJsonPath, metaJsonPath) {
  const m = JSON.parse(fs.readFileSync(modelJsonPath, "utf8"));
  const p = JSON.parse(fs.readFileSync(pipelineJsonPath, "utf8"));
  const meta = metaJsonPath ? JSON.parse(fs.readFileSync(metaJsonPath, "utf8")) : null;

  const trees = m.learner.gradient_booster.model.trees.map((t) => ({
    splitIndex: Int32Array.from(t.split_indices),
    splitCond: Float32Array.from(t.split_conditions),
    leaf: Float32Array.from(t.base_weights),
    left: Int32Array.from(t.left_children),
    right: Int32Array.from(t.right_children),
    defaultLeft: Int32Array.from(t.default_left),
  }));

  const baseScore =
    parseFloat(m.learner.learner_model_param.base_score.replace(/[\[\]]/g, ""));
  const featureIndex = {};
  FEATURE_COLS.forEach((c, i) => (featureIndex[c] = i));

  return {
    featureCols: FEATURE_COLS,
    featureIndex,
    baseScore,
    globalMean: p.global_mean,
    zipEnc: p.zip_enc,
    trees,
    treeViews: m.learner.gradient_booster.model.trees.map(treeViews),
    meta,
  };
}

/**
 * Predict raw (float32-accumulated) price for an engineered float64 feature vector.
 * `features` may be an object (name->value) or a 33-length array.
 */
function predictRaw(model, features) {
  const x = new Float32Array(33);
  if (Array.isArray(features)) {
    for (let i = 0; i < 33; i++) x[i] = f32(features[i]);
  } else {
    for (let i = 0; i < 33; i++) {
      const c = FEATURE_COLS[i];
      const v = features[c];
      x[i] = v === null || v === undefined || Number.isNaN(v)
        ? NaN
        : f32(v);
    }
  }

  let acc = f32(model.baseScore);
  for (const t of model.trees) {
    let n = 0;
    for (;;) {
      if (t.left[n] === -1) break; // leaf
      const fv = x[t.splitIndex[n]];
      let goLeft;
      if (fv === undefined || Number.isNaN(fv)) {
        goLeft = t.defaultLeft[n] === 1;
      } else {
        goLeft = fv < t.splitCond[n];
      }
      n = goLeft ? t.left[n] : t.right[n];
    }
    acc = f32(acc + t.leaf[n]);
  }
  return acc;
}

const UNSEEN = -999.0; // f32(-999)
const NR = 8;
const KPI = 3.141592653589793238462643383279502884;
const MIN_CHILD = 1e-12;

function legendrePoly(n, x) {
  let p0 = 1.0;
  if (n === 0) return p0;
  let p1 = x;
  if (n === 1) return p1;
  for (let k = 2; k <= n; k++) {
    const pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
    p0 = p1;
    p1 = pk;
  }
  return p1;
}
function legendreDeriv(n, x, pn) {
  const nd = n;
  return (nd * (x * pn - legendrePoly(n - 1, x))) / (x * x - 1.0);
}
function makeQuadratureRule() {
  const nodes = new Array(NR);
  const weights = new Array(NR);
  for (let i = 0; i < NR; i++) {
    const theta = (KPI * (i + 0.75)) / (NR + 0.5);
    let x = Math.cos(theta);
    for (let it = 0; it < 64; it++) {
      const pn = legendrePoly(NR, x);
      const dpn = legendreDeriv(NR, x, pn);
      const dx = pn / dpn;
      x -= dx;
      if (Math.abs(dx) < 1e-15) break;
    }
    const pn = legendrePoly(NR, x);
    const dpn = legendreDeriv(NR, x, pn);
    const w = 2.0 / ((1.0 - x * x) * dpn * dpn);
    const s = 0.5 * (x + 1.0);
    const ws = 0.5 * w;
    const outIdx = NR - 1 - i;
    nodes[outIdx] = f32(s * s);
    weights[outIdx] = f32(2.0 * s * ws);
  }
  return { nodes, weights };
}
const QRULE = makeQuadratureRule();

function branchWeight(cover, parentCover) {
  if (parentCover <= 0.0) return f32(0.5);
  const weight = cover / parentCover;
  if (weight < MIN_CHILD) return f32(MIN_CHILD);
  return f32(weight);
}

function extractQuadratureDelta(hVals, pEnter, pExit) {
  let acc = f32(0.0);
  const nodes = QRULE.nodes;
  if (pEnter !== 1.0) {
    const alphaEnter = f32(pEnter - 1.0);
    for (let i = 0; i < NR; i++) {
      const denom = f32(1.0 + f32(alphaEnter * nodes[i]));
      acc = f32(acc + f32(f32(alphaEnter * hVals[i]) / denom));
    }
  }
  if (pExit !== 1.0) {
    const alphaExit = f32(pExit - 1.0);
    for (let i = 0; i < NR; i++) {
      const denom = f32(1.0 + f32(alphaExit * nodes[i]));
      acc = f32(acc - f32(f32(alphaExit * hVals[i]) / denom));
    }
  }
  return acc;
}

// Precompute per-tree float32 views once at load time.
function treeViews(tr) {
  const splitCond = tr.split_conditions.map((v) => f32(v));
  const leaf = tr.base_weights.map((v) => f32(v));
  const sh = tr.sum_hessian.map((v) => f32(v));
  return {
    splitIndex: tr.split_indices,
    splitCond,
    leaf,
    left: tr.left_children,
    right: tr.right_children,
    defaultLeft: tr.default_left,
    sh,
    nn: tr.split_indices.length,
  };
}

/**
 * Per-feature additive SHAP contributions + bias, faithful port of xgboost's
 * QuadratureTreeSHAP. Returns { contribs: number[33], base: number }.
 * `feats` is the engineered object (name->float64 value).
 */
function predictContribs(model, feats) {
  const x = new Array(33);
  for (let i = 0; i < 33; i++) {
    const c = FEATURE_COLS[i];
    const v = feats[c];
    x[i] = v === null || v === undefined || Number.isNaN(v) ? NaN : f32(v);
  }

  const contrib = new Float32Array(33);
  const pathProb = new Array(33).fill(UNSEEN);
  let rootMeanSum = f32(0.0);
  let treeContrib = new Float32Array(33);

  const runNode = (tv, nidx, cVals, wProd, outH) => {
    if (tv.left[nidx] === -1) {
      const leafScale = f32(f32(wProd) * tv.leaf[nidx]);
      for (let i = 0; i < NR; i++) {
        outH[i] = f32(f32(cVals[i] * leafScale) * QRULE.weights[i]);
      }
      return;
    }
    const left = tv.left[nidx];
    const right = tv.right[nidx];
    const leftW = branchWeight(tv.sh[left], tv.sh[nidx]);
    const rightW = branchWeight(tv.sh[right], tv.sh[nidx]);
    const lacksLeft = x[tv.splitIndex[nidx]] === NaN || Number.isNaN(x[tv.splitIndex[nidx]]);
    const goesLeft = lacksLeft
      ? tv.defaultLeft[nidx] === 1
      : x[tv.splitIndex[nidx]] < tv.splitCond[nidx];

    const rightH = new Array(NR).fill(f32(0.0));
    visitChild(tv, nidx, left, leftW, goesLeft, cVals, wProd, outH);
    visitChild(tv, nidx, right, rightW, !goesLeft, cVals, wProd, rightH);
    for (let i = 0; i < NR; i++) outH[i] = f32(outH[i] + rightH[i]);
  };

  const visitChild = (tv, splitNode, childNode, childW, satisfies, cVals, wProd, outH) => {
    const sidx = tv.splitIndex[splitNode];
    const pOld = pathProb[sidx];
    let pE;
    if (pOld === UNSEEN) {
      pE = satisfies ? f32(1.0 / childW) : f32(0.0);
    } else {
      pE = satisfies ? f32(pOld / childW) : f32(0.0);
    }
    const cChild = cVals.slice();
    const alphaE = f32(pE - 1.0);
    for (let i = 0; i < NR; i++) {
      cChild[i] = f32(cChild[i] * f32(1.0 + f32(alphaE * QRULE.nodes[i])));
    }
    if (pOld !== UNSEEN) {
      const alphaOld = f32(pOld - 1.0);
      if (alphaOld !== 0.0) {
        for (let i = 0; i < NR; i++) {
          cChild[i] = f32(cChild[i] / f32(1.0 + f32(alphaOld * QRULE.nodes[i])));
        }
      }
    }
    pathProb[sidx] = pE;
    runNode(tv, childNode, cChild, f32(f32(wProd) * childW), outH);
    const pEnter = pE;
    const pExit = pOld === UNSEEN ? f32(1.0) : pOld;
    treeContrib[sidx] = f32(treeContrib[sidx] + extractQuadratureDelta(outH, pEnter, pExit));
    pathProb[sidx] = pOld;
  };

  const fillRootMean = (tv, nidx) => {
    if (tv.left[nidx] === -1) return tv.leaf[nidx];
    const left = tv.left[nidx];
    const right = tv.right[nidx];
    const parent = tv.sh[nidx];
    const lm = fillRootMean(tv, left);
    const rm = fillRootMean(tv, right);
    if (parent === 0.0) return f32(f32(0.5) * f32(lm + rm));
    return f32(f32(f32(lm * tv.sh[left]) + f32(rm * tv.sh[right])) / parent);
  };

  for (const tv of model.treeViews) {
    for (let i = 0; i < 33; i++) pathProb[i] = UNSEEN;
    treeContrib = new Float32Array(33);
    if (tv.left[0] !== -1) {
      const cInit = new Array(NR).fill(f32(1.0));
      const hVals = new Array(NR).fill(f32(0.0));
      runNode(tv, 0, cInit, f32(1.0), hVals);
    }
    for (let i = 0; i < 33; i++) contrib[i] = f32(contrib[i] + treeContrib[i]);
    rootMeanSum = f32(rootMeanSum + f32(fillRootMean(tv, 0)));
  }

  const contribs = new Array(33);
  for (let i = 0; i < 33; i++) contribs[i] = contrib[i];
  const base = f32(rootMeanSum + f32(model.baseScore));
  return { contribs, base };
}

function round2(x) {
  // Python-compatible round-half-to-even (banker's rounding), matching builtins.round(x, 2).
  const m = 100;
  const sc = x * m;
  const fl = Math.floor(sc);
  const frac = sc - fl;
  if (frac === 0.5) {
    return fl % 2 === 0 ? fl / m : (fl + 1) / m;
  }
  return Math.round(sc) / m;
}

/**
 * Local SHAP summary (mirrors src/inference/explain.local_summary).
 * Returns the object or null on failure.
 */
function localSummary(model, input) {
  try {
    const feats = engineerFeatures(input, model);
    const { contribs, base } = predictContribs(model, feats);
    let s = 0;
    for (let i = 0; i < 33; i++) s += contribs[i];
    const names = FEATURE_COLS;
    const item = (i, direction) => ({
      feature: names[i],
      label: FEATURE_LABELS[names[i]] || names[i].replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      contribution: round2(contribs[i]),
      direction,
    });
    const posOrder = Array.from({ length: 33 }, (_, i) => i).sort((a, b) => contribs[b] - contribs[a]);
    const negOrder = Array.from({ length: 33 }, (_, i) => i).sort((a, b) => contribs[a] - contribs[b]);
    const positives = posOrder.filter((i) => contribs[i] > 0).slice(0, 5).map((i) => item(i, "up"));
    const negatives = negOrder.filter((i) => contribs[i] < 0).slice(0, 5).map((i) => item(i, "down"));
    const total = round2(s);
    const predCont = round2(base + s);
    return {
      expected_value: round2(base),
      total_contribution: total,
      predicted_price: predCont,
      top_positive: positives,
      top_negative: negatives,
      method: model.meta ? model.meta.local_shap_method : "TreeSHAP on the deployed model, evaluated at request time",
    };
  } catch (e) {
    return null;
  }
}

/**
 * Empirical error band for a price (mirrors src/inference/explain.error_band_for).
 * Returns the object or null.
 */
function errorBand(model, price) {
  if (!model.meta || !model.meta.error_band) return null;
  const { edges, segments } = model.meta.error_band;
  if (price == null) return null;
  // pd.cut(include_lowest=True): first bin closed on both ends, others left-open/right-closed.
  for (let k = 0; k < segments.length; k++) {
    const seg = segments[k];
    const lo = edges[k];
    const hi = edges[k + 1];
    const inBin = k === 0 ? price >= lo && price <= hi : price > lo && price <= hi;
    if (inBin) {
      return {
        segment_label: seg.segment_label,
        typical_error: seg.typical_error,
        n: seg.n,
        method: model.meta.error_band.method,
        note: model.meta.error_band.note,
      };
    }
  }
  return null;
}

/**
 * Full production prediction response (mirrors predict_single + the /predict handler).
 * Throws an Error with a friendly message on validation failure.
 */
function predictService(model, input) {
  const feats = engineerFeatures(input, model);
  const raw = predictRaw(model, feats);
  const price = round2(raw);
  const meta = model.meta || {};
  return {
    predicted_price: price,
    model: meta.model || "XGBoost (tuned) on engineered tabular features",
    importance_type: meta.importance_type || "xgboost_gain",
    top_factors_gain: meta.top_factors_gain || [],
  };
}

// Field validation rules (mirror app/backend/main.py PRIMARY_FIELD_RULES).
const PRIMARY_FIELD_RULES = {
  bedrooms: ["bedrooms", 1, 20],
  bathrooms: ["bathrooms", 0.5, 20],
  sqft_living: ["sqft_living", 200, 10000],
  sqft_lot: ["sqft_lot", 200, 500000],
  floors: ["floors", 1, 10],
  waterfront: ["waterfront", 0, 1],
  view: ["view", 0, 4],
  condition: ["condition", 1, 5],
  grade: ["grade", 1, 13],
  sqft_above: ["sqft_above", 100, 12000],
  sqft_basement: ["sqft_basement", 0, 8000],
  yr_built: ["year built", 1800, 2100],
  yr_renovated: ["year renovated", 0, 2100],
  zipcode: ["zipcode", 10000, 99999],
  lat: ["latitude", -90.0, 90.0],
  long: ["longitude", -180.0, 180.0],
  sqft_living15: ["sqft_living15", 200, 10000],
  sqft_lot15: ["sqft_lot15", 200, 500000],
};

// Required raw fields (mirror predict.REQUIRED_FIELDS; optional fields defaulted).
const REQUIRED_FIELDS = [
  "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
  "sqft_above", "yr_built", "zipcode", "lat", "long",
  "sqft_living15", "sqft_lot15",
];

/**
 * Validate a raw input. Mirrors _validate_primary_payload + required-field checks.
 * Returns null if valid, else an error message string.
 */
function validateInput(input) {
  if (typeof input !== "object" || input === null) {
    return "Please provide a JSON object with property features.";
  }
  for (const f of REQUIRED_FIELDS) {
    const v = input[f];
    if (v === undefined || v === null || v === "") {
      const rule = PRIMARY_FIELD_RULES[f];
      return `Please enter a valid value for ${rule ? rule[0] : f}.`;
    }
  }
  for (const field of Object.keys(input)) {
    const rule = PRIMARY_FIELD_RULES[field];
    if (!rule) continue;
    const label = rule[0];
    const value = input[field];
    if (value === null) return `Please enter a valid value for ${label}.`;
    if ((typeof value === "number") && !(rule[1] <= value && value <= rule[2])) {
      return `Please enter a valid value for ${label}.`;
    }
  }
  return null;
}

/**
 * Full response body for a request (mirrors the /predict handler).
 * Assumes input already validated.
 */
function predictContract(model, input) {
  const result = predictService(model, input);
  result.model_role = "primary";
  result.status = "production";
  result.local_shap = localSummary(model, input);
  result.error_band = errorBand(model, result.predicted_price);
  return result;
}

module.exports = {
  FEATURE_COLS,
  engineerFeatures,
  loadModel,
  predictRaw,
  predictContribs,
  localSummary,
  errorBand,
  predictService,
  predictContract,
  validateInput,
  DEFAULT_VALUES,
};