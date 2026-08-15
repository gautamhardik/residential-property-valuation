"""Phase 16 — Final QA cascade. Read-only verification; nothing is retrained."""
import json
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = "PASS"
FAIL = "FAIL"


def check(label, cond, detail=""):
    print(f"[{PASS if cond else FAIL}] {label}  {detail}")
    return cond


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    all_ok = True
    print("=== PHASE 16 QA CASCADE ===")

    # 1. Champion frozen (read-only against baseline_snapshot.json)
    freeze = json.loads((Path("reports") / "baseline_snapshot.json").read_text(encoding="utf-8"))
    aliases = {"submission.csv": "predictions/submission.csv"}
    mismatches = []
    for name, want in freeze["checksums_sha256_16"].items():
        got = sha16(aliases.get(name, name))
        if got != want:
            mismatches.append((name, got, want))
    all_ok &= check("champion checksums identical (6 artifacts)", not mismatches,
                    str([m[0] for m in mismatches]) if mismatches else "")
    all_ok &= check("champion metrics asserted", "RMSE ~= $103,802.8" in freeze["champion"]["asserted"],
                    freeze["champion"]["asserted"])

    # 2. Vision model integrity
    ser = json.loads((Path("reports") / "results_dl_serialization.json").read_text(encoding="utf-8"))
    v = ser["verification"]
    all_ok &= check("serialization R2_diff <= 0.001", v["r2_difference"] <= 0.001,
                    f"diff={v['r2_difference']:.6f}")
    all_ok &= check("serialization max-abs pred diff < $0.01",
                    v["max_abs_pred_difference"] < 0.01,
                    f"${v['max_abs_pred_difference']:,.2f}")
    all_ok &= check("val n == 434 (no test leakage)", v["n_val"] == 434, f"n={v['n_val']}")
    all_ok &= check("deployed torchscript artifact exists",
                    Path("models/deployed/vision_price.pt").exists())

    dl = json.loads((Path("reports") / "results_dl.json").read_text(encoding="utf-8"))
    seeds = {r.get("seed") for r in dl["variants"].values()}
    all_ok &= check("Experiment A deterministic (seed 42)", seeds == {42}, str(seeds))
    both_done = {"A1_price_MSE", "A2_log1p_MSE"} <= set(dl["variants"].keys())
    all_ok &= check("A1 and A2 both recorded with history", both_done)

    # 3. Research integrity / decision gate
    ev = json.loads((Path("reports") / "results_dl_eval.json").read_text(encoding="utf-8"))
    gate = ev["decision_gate"]
    all_ok &= check("gate fired Case 1 (STOP)", gate["decision"].startswith("Case 1"), gate["decision"])
    all_ok &= check("gate computes same best-A R2", abs(gate["best_experiment_a_r2"] - 0.11260274) < 1e-6,
                    f"{gate['best_experiment_a_r2']:.6f}")
    all_ok &= check("E4 control on same 434", abs(gate["e4_r2"] - 0.87208922) < 1e-6,
                    f"{gate['e4_r2']:.6f}")

    final_ = json.loads((Path("reports") / "results_dl_final.json").read_text(encoding="utf-8"))
    oc = final_["conclusion"]
    all_ok &= check("conclusion = Outcome A (negative)", oc["outcome"].startswith("OUTCOME A"))
    all_ok &= check("E6 not executed (gated)", oc["e6"]["executed"] is False)

    # 4. Serving
    import src.inference.vision as vinf
    price = vinf.predict_pid(1777500160)
    all_ok &= check("vision inference loads & predicts", price > 0, f"${price:,.0f}")

    # 5. Documentation
    report = (Path("reports") / "project_report.md").read_text(encoding="utf-8")
    all_ok &= check("report has DL extension section", "## 8. Satellite Experiment" in report
                    and "the DL extension" in report)
    nums = [int(m.group(1)) for m in re.finditer(r"^## (\d+)\. ", report, re.M)]
    all_ok &= check("report sections sequential 1..14", nums == list(range(1, 15)), str(nums))
    all_ok &= check("report has Appendix Reproduction", "## Appendix — Reproduction" in report)
    readme = (Path("README.md").read_text(encoding="utf-8"))
    all_ok &= check("README states vision_only secondary service", "vision_only" in readme)
    all_ok &= check("README fine-tune section present", "What about fine-tuning" in readme)
    log = json.loads((Path("reports") / "experiment_log.json").read_text(encoding="utf-8"))
    all_ok &= check("experiment_log has 13 phases", len(log["phases"]) == 13, str(len(log["phases"])))

    # 6. Portfolio readiness
    figs = ["fig_architecture.png", "fig_model_comparison.png", "fig_generalization.png",
            "fig_shap_importance.png", "fig_gradcam.png"]
    missing_figs = [f for f in figs if not (Path("reports/figures") / f).exists()]
    all_ok &= check("exactly the 5 portfolio figures present", not missing_figs, str(missing_figs))
    all_ok &= check("README embeds architecture figure", "reports/figures/fig_architecture.png" in readme)
    all_ok &= check("README embeds model-comparison figure",
                    "reports/figures/fig_model_comparison.png" in readme)
    legacy_readme = Path("scripts/legacy/README.md")
    all_ok &= check("scripts/legacy archive exists with README", legacy_readme.exists())
    gitignore = (Path(".gitignore").read_text(encoding="utf-8"))
    all_ok &= check(".gitignore excludes .env and data", "\n.env\n" in gitignore
                    and "data/*.xlsx" in gitignore and "catboost_info/" in gitignore)
    ft = json.loads((Path("reports") / "results_dl_final.json").read_text(encoding="utf-8"))
    table_rows = ft["table"]
    all_ok &= check("final table has 6 rows incl. Production", len(table_rows) == 6,
                    str([r.get("exp") for r in table_rows]))
    has_prod = any(r.get("exp") == "Production" for r in table_rows)
    all_ok &= check("final table includes Production row", has_prod)
    all_ok &= check("every final-table row has source provenance",
                    all("source" in r for r in table_rows))

    # 7. Security hygiene (no real secrets committed in source files)
    safe_exts = {".py", ".md", ".txt", ".json", ".cfg", ".toml", ".yml", ".yaml", ".html", ".css", ".js"}
    secret_pats = [
        re.compile(r"pk\.eyJ[A-Za-z0-9_-]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
    ]
    leaks = []
    for p in sorted(Path(".").rglob("*")):
        if not p.is_file() or p.suffix not in safe_exts:
            continue
        if any(part in {"data", "images", "venv", ".venv", "env", "__pycache__", "legacy"} for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in secret_pats:
            if pat.search(txt):
                leaks.append(f"{p} ~ {pat.pattern[:12]}")
    all_ok &= check("no committed secret-like tokens in source", not leaks, str(leaks))

    print()
    print("ALL QA CHECKS PASS" if all_ok else "QA FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())