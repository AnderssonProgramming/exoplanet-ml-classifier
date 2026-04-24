"""
Project-wide constants for the exoplanet-ml-classifier.

All column names, file paths, model names, metric names,
and hyperparameter grids are defined here and imported elsewhere.
No other logic lives in this module.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH = Path("data/raw/cumulative_koi.csv")
PROCESSED_DATA_PATH = Path("data/processed/koi_processed.csv")
FIGURES_PATH = Path("reports/figures/")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset — KOI column names (exact NASA Exoplanet Archive field names)
# ---------------------------------------------------------------------------
TARGET_COLUMN = "koi_pdisposition"
POSITIVE_CLASS = "CANDIDATE"
NEGATIVE_CLASS = "FALSE POSITIVE"

LEAKAGE_COLUMNS = [
    "koi_score",
    "koi_disposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_comment",
]

ID_COLUMNS = ["kepid", "kepoi_name", "kepler_name"]

NUMERIC_FEATURES = [
    "koi_period",
    "koi_time0bk",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
    "koi_model_snr",
]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # fraction of the remaining train set

# ---------------------------------------------------------------------------
# Model hyperparameter grids for GridSearchCV
# ---------------------------------------------------------------------------
LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
}

KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced", None],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "scale_pos_weight": [1, 3],
}

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [500],
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CV_FOLDS = 5
PRIMARY_METRIC = "f1"
CLASSIFICATION_THRESHOLD = 0.5
MODEL_NAMES = [
    "logistic_regression",
    "knn",
    "random_forest",
    "xgboost",
    "mlp",
]
