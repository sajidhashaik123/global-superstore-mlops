from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Global Superstore dataset
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v1"
    / "global_superstore_clean.csv"
)

DATASET_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v1"
    / "manifest.json"
)

# Existing model versions (read-only)
BASELINE_MODEL_DIR = PROJECT_ROOT / "models" / "lab7" / "v1"
RETRAINED_MODEL_DIR = PROJECT_ROOT / "models" / "lab12" / "v2"

# Experiment 13 directories
LAB13_MODEL_DIR = PROJECT_ROOT / "models" / "lab13"
LAB13_REPORT_DIR = PROJECT_ROOT / "reports" / "lab13"
LAB13_LOG_DIR = LAB13_REPORT_DIR / "logs"
LAB13_DOC_DIR = PROJECT_ROOT / "docs" / "lab13"

# Experiment 12 evaluation report
LAB12_COMPARISON_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "lab12"
    / "retraining_comparison.json"
)

# Target definition
TARGET_COLUMN = "Profitability"
PROFIT_COLUMN = "Profit"
PROFITABILITY_THRESHOLD = 0

# Model versions
BASELINE_MODEL_VERSION = "lab7-v1"
RETRAINED_MODEL_VERSION = "lab12-v2"

# Initial lifecycle decision based on Experiment 12 evaluation
INITIAL_PRODUCTION_MODEL = BASELINE_MODEL_VERSION
RETRAINED_MODEL_STATUS = "candidate_not_promoted"

# Model settings
RANDOM_STATE = 42
MAX_ITER = 1000

# Six numerical input features
NUMERICAL_FEATURES = [
    "Sales",
    "Quantity",
    "Discount",
    "Shipping Cost",
    "Order Year",
    "Order Month",
]

# Ten categorical input features
CATEGORICAL_FEATURES = [
    "Ship Mode",
    "Segment",
    "City",
    "State",
    "Country",
    "Market",
    "Region",
    "Category",
    "Sub-Category",
    "Order Priority",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Deployment settings for the separate Lab 13 service
API_HOST = "0.0.0.0"
API_PORT = 8000

DOCKER_IMAGE_NAME = "global-superstore-mlops-e2e"
DOCKER_IMAGE_TAG = "v1"

KUBERNETES_NAMESPACE = "default"
KUBERNETES_DEPLOYMENT_NAME = "global-superstore-lab13"
KUBERNETES_SERVICE_NAME = "global-superstore-lab13-service"
KUBERNETES_NODE_PORT = 30081