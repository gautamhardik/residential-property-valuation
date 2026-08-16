"use strict";
const handler = require("../api/predict.js");

let statusCode = null, jsonOut = null;
const res = { status(c) { statusCode = c; return this; }, json(o) { jsonOut = o; } };

let req = { method: "GET", body: "{}" };
handler(req, res);
console.log("GET /api/predict ->", statusCode, JSON.stringify(jsonOut));

statusCode = null; jsonOut = null;
req = { method: "POST", body: JSON.stringify({ nope: 1 }) };
handler(req, res);
console.log("POST missing fields ->", statusCode, JSON.stringify(jsonOut));

const canonical = {
  bedrooms: 3, bathrooms: 2.0, sqft_living: 1910, sqft_lot: 7600, floors: 1.5,
  waterfront: 0, view: 0, condition: 3, grade: 7, sqft_above: 1560, sqft_basement: 0,
  yr_built: 1975, yr_renovated: 0, zipcode: 98065, lat: 47.5724, long: -122.2300,
  sqft_living15: 1840, sqft_lot15: 7620, sale_year: 2015, sale_quarter: 3,
};
statusCode = null; jsonOut = null;
req = { method: "POST", body: Buffer.from(JSON.stringify(canonical)) };
handler(req, res);
console.log("POST canonical ->", statusCode);
console.log("predicted_price =", jsonOut && jsonOut.predicted_price);
console.log("expected_value  =", jsonOut && jsonOut.local_shap && jsonOut.local_shap.expected_value);
process.exit(0);