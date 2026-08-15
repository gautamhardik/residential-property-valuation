"""Phase 5c — Image-coverage selection bias.

Compares the image-covered subset (n = 2,189) against properties without imagery
(n = 13,921) on the distributions that matter, quantifying whether the covered set
is a random sub-sample of the data or a systematically different population.

Coverage is a *convenience sample* (Mapbox tiles that survived fetch/rate-limits in
row order), so distributional differences change how the satellite experiment below
must be read. Statistics: mean/median per group, Cohen's d effect size, and a
two-sample t-test (log-price where heavy skew) / z-test for proportions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPORTS_DIR, FIGURES_DIR
from src.data.load import load_clean_train
from src.satellite.align import is_valid_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp if sp > 0 else 0.0)


def run():
    df = load_clean_train()
    df["has_img"] = df["id"].map(is_valid_image)
    covered = df[df["has_img"]]
    non = df[~df["has_img"]]
    print(f"covered={len(covered)}  non-covered={len(non)}  (pct covered={len(covered)/len(df)*100:.2f}%)")

    numeric_cols = ["price", "grade", "sqft_living", "sqft_lot", "bathrooms", "lat", "long"]
    rows = {}
    for col in numeric_cols:
        a = covered[col].to_numpy(dtype=float)
        b = non[col].to_numpy(dtype=float)
        statcol = np.log(a + 1) if col == "price" else a
        statcol_b = np.log(b + 1) if col == "price" else b
        t, p = stats.ttest_ind(statcol, statcol_b, equal_var=False)
        rows[col] = {
            "covered_mean": round(float(a.mean()), 2),
            "noncovered_mean": round(float(b.mean()), 2),
            "covered_median": round(float(np.median(a)), 2),
            "noncovered_median": round(float(np.median(b)), 2),
            "diff": round(float(a.mean() - b.mean()), 2),
            "cohens_d": round(cohens_d(statcol, statcol_b), 3),
            "p_value": float(p),
        }

    # categorical
    for col in ("waterfront",):
        pa = float(covered[col].mean())
        pb = float(non[col].mean())
        se = np.sqrt(pa * (1 - pa) / len(covered) + pb * (1 - pb) / len(non))
        z = (pa - pb) / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        rows[col] = {
            "covered_prop": round(pa, 4),
            "noncovered_prop": round(pb, 4),
            "diff": round(pa - pb, 4),
            "z": round(float(z), 3),
            "p_value": float(p),
        }

    # geography / zip
    rows["zipcode"] = {
        "covered_unique_zips": int(covered["zipcode"].nunique()),
        "noncovered_unique_zips": int(non["zipcode"].nunique()),
        "covered_top5": covered["zipcode"].value_counts().head(5).to_dict(),
        "noncovered_top5": non["zipcode"].value_counts().head(5).to_dict(),
    }

    summary = {
        "covered_n": int(len(covered)),
        "noncovered_n": int(len(non)),
        "coverage_pct": round(len(covered) / len(df) * 100, 2),
        "sample_type": "convenience sample (Mapbox fetch in row order, 429/rate-limit filtered)",
        "comparisons": rows,
        "interpretation": (
            "Effect sizes (Cohen's d) below ~0.2 are negligible; the groups differ "
            "appreciably when |d| >= 0.2-0.5. Always contrast with the satellite "
            "experiment (which is internally fair: same subset, same split, same model)."
        ),
    }
    (REPORTS_DIR / "coverage_bias.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # figure: price distributions
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(covered["price"] + 1), bins=40, alpha=0.6, density=True,
            label=f"covered (n={len(covered)})")
    ax.hist(np.log10(non["price"] + 1), bins=40, alpha=0.6, density=True,
            label=f"non-covered (n={len(non)})")
    ax.set_xlabel("log10(price)")
    ax.set_title("Price distribution: image-covered vs non-covered properties")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coverage_bias_price.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    run()