"use strict";
/** Vercel serverless function: GET /api/health (also aliased to /health). */
module.exports = (req, res) => {
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.status(404).json({ error: "Not Found" });
    return;
  }
  res.status(200).json({ status: "ok", model: "XGBoost tuned tabular" });
};