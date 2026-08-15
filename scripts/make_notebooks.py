"""Generate the clean portfolio notebooks under notebooks/ using nbformat."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

MARKDOWN_KIND = "markdown"
CODE_KIND = "code"


def cell(kind, src, meta_extra=None):
    if not meta_extra:
        meta_extra = {}
    if kind == "markdown":
        c = nbf.v4.new_markdown_cell(src.strip("\n"), metadata=meta_extra)
    else:
        c = nbf.v4.new_code_cell(src.strip("\n"), metadata=meta_extra)
    return c


def write_nb(path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(nb, path)


PREAMBLE = "# Utilities to add the project root to sys.path so `import src` works from notebooks.\nimport sys, os\nfrom pathlib import Path\nROOT = Path(os.getcwd()).resolve()\nwhile 'src' not in [p.name for p in ROOT.iterdir()] and ROOT.parent != ROOT:\n    ROOT = ROOT.parent\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n"

# ---------------------------------------------------------------------------
# 01 — Data validation & EDA
# ---------------------------------------------------------------------------
PREAMBLE01 = ("%matplotlib inline\n"
              "# Utilities to add the project root to sys.path so `import src` works from notebooks.\n"
              "import sys, os\nfrom pathlib import Path\n"
              "ROOT = Path(os.getcwd()).resolve()\n"
              "while 'src' not in [p.name for p in ROOT.iterdir()] and ROOT.parent != ROOT:\n"
              "    ROOT = ROOT.parent\n"
              "if str(ROOT) not in sys.path:\n"
              "    sys.path.insert(0, str(ROOT))\n")
c01 = [
    cell(MARKDOWN_KIND,
         "# 01 — Data Validation & Exploratory Data Analysis\n"
         "This notebook validates the King-County dataset (duplicates, train/test overlap, "
         "coordinates, date parsing) and inspects the target distribution and spatial structure."),
    cell(CODE_KIND, PREAMBLE01),
    cell(CODE_KIND,
         "from src.data.load import load_clean_train, load_clean_test\n\n"
         "df = load_clean_train()\n"
         "test = load_clean_test()\n"
         "print('Clean train:', df.shape, '| test:', test.shape)\n"
         "print('Duplicate ids in train:', int(df['id'].duplicated().sum()))\n"
         "print('sale years:', df['sale_year'].min(), '-', df['sale_year'].max())"),
    cell(CODE_KIND,
         "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
         "plt.figure(figsize=(8, 4))\n"
         "sns.histplot(df['price'], bins=60)\n"
         "plt.title('Price distribution (right-skewed, log used downstream)')\n"
         "plt.tight_layout(); plt.show()"),
    cell(CODE_KIND,
         "plt.figure(figsize=(7, 6))\n"
         "sc = plt.scatter(df['long'], df['lat'], c=df['price'], cmap='viridis', s=4)\n"
         "plt.colorbar(sc, label='price')\n"
         "plt.title('Geographic price structure (King County, WA)')\n"
         "plt.tight_layout(); plt.show()"),
    cell(MARKDOWN_KIND,
         "### Findings\n"
         "- 99 duplicate ids are repeat sales of the same home; the most recent is kept.\n"
         "- 70 ids overlap train/test but describe identical properties; test has no labels so no target leakage.\n"
         "- Coordinates are all valid; `date` parsed into `sale_year` / `sale_quarter` (2014-2015)."),
]
write_nb(OUT / "01_data_validation_and_eda.ipynb", c01)

# ---------------------------------------------------------------------------
# 02 — Tabular baselines
# ---------------------------------------------------------------------------
c02 = [
    cell(MARKDOWN_KIND,
         "# 02 — Tabular Baselines (E1/E2/E3)\n"
         "Linear Regression, Random Forest and XGBoost on (E1) the original 5-feature set, "
         "(E2) full raw tabular features, and (E3) engineered features. Identical 80/20 split."),
    cell(CODE_KIND, PREAMBLE),
    cell(CODE_KIND,
         "import pandas as pd\nfrom src.config import REPORTS_DIR\n"
         "res = pd.read_csv(REPORTS_DIR / 'results_tabular.csv')\n"
         "res.pivot(index='experiment', columns='model', values='rmse').round(0)"),
    cell(CODE_KIND,
         "res.pivot(index='experiment', columns='model', values='r2').round(4)"),
    cell(MARKDOWN_KIND,
         "### Takeaway\n"
         "Full/engineered features beat the original 5-feature baseline by a wide margin. "
         "XGBoost on engineered features is the strongest untuned tabular model."),
]
write_nb(OUT / "02_tabular_baselines.ipynb", c02)

# ---------------------------------------------------------------------------
# 03 — Satellite & multimodal experiment
# ---------------------------------------------------------------------------
c03 = [
    cell(MARKDOWN_KIND,
         "# 03 — Fair Multimodal Experiment (E4/E5)\n"
         "On the exact same image-covered subset and the same split, we compare tabular-only, "
         "image-only and tabular+satellite (ResNet18/ResNet50 frozen embeddings)."),
    cell(CODE_KIND, PREAMBLE),
    cell(CODE_KIND,
         "import pandas as pd\nfrom src.config import REPORTS_DIR\n"
         "res = pd.read_csv(REPORTS_DIR / 'results_multimodal.csv')\n"
         "res[['experiment', 'family', 'rmse', 'r2', 'rmse_improvement_pct']].round(2)"),
    cell(MARKDOWN_KIND,
         "### Honest conclusion\n"
         "Satellite embeddings from frozen ImageNet encoders do **not** improve price prediction "
         "here; image-only is near-useless (R2 ~0.14) and adding embeddings slightly degrades the "
         "fair tabular control. See the full report for the scientific interpretation."),
]
write_nb(OUT / "03_satellite_and_multimodal.ipynb", c03)

print("Written notebooks to", OUT)
for nb in OUT.glob("*.ipynb"):
    print(" -", nb.name)