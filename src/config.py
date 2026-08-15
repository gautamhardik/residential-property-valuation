"""Global configuration: paths, seeds, dataset constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_TRAIN = DATA_DIR / "train.xlsx"
DATA_TEST = DATA_DIR / "test.xlsx"

IMAGE_DIR = PROJECT_ROOT / "images"
IMAGE_TRAIN = IMAGE_DIR / "train"
IMAGE_TEST = IMAGE_DIR / "test"

PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
APP_MODELS_DIR = MODELS_DIR / "deployed"

for _d in (PREPROCESSED_DIR, PREDICTIONS_DIR, REPORTS_DIR, FIGURES_DIR,
           MODELS_DIR, APP_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Seeds ---
RANDOM_STATE = 42
N_SPLITS = 5                 # folds request (used if KFold is added)
TEST_SIZE = 0.2              # validation hold-out share

# --- Data ---
TARGET = "price"
SALE_REFERENCE_YEAR = 2015   # 2015-01-01 reference for age features
SEATTLE_CENTER = (47.6062, -122.3321)

# --- Imagery ---
MAPBOX_STYLE = "satellite-v9"
IMAGE_SIZE = 256
ZOOM_LEVEL = 18
IMAGE_EXT = ".jpg"

# --- CNN/ViT encoders ---
ENCODER_NAME = "resnet18"    # "resnet18" | "resnet50" | "dinov2_vits14"
EMBEDDING_DIM = {"resnet18": 512, "resnet50": 2048, "dinov2_vits14": 384}
MIN_TRIALS = 30              # RandomizedSearchCV iterations per model