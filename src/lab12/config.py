from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v1"
    / "global_superstore_clean.csv"
)

# Original production model (do not overwrite)
BASELINE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "lab7"
    / "v1"
)

# Experiment 12 model artifacts
LAB12_MODEL_DIR = PROJECT_ROOT / "models" / "lab12"

# Experiment 12 reports
LAB12_REPORT_DIR = PROJECT_ROOT / "reports" / "lab12"

# Monitoring logs
MONITORING_LOG_DIR = PROJECT_ROOT / "reports" / "lab12" / "logs"

# Target definition
TARGET_COLUMN = "Profitability"
PROFIT_COLUMN = "Profit"

# Target rule: Profit > 0 means profitable
PROFITABILITY_THRESHOLD = 0

# Chronological periods
REFERENCE_START = "2011-01-01"
REFERENCE_END = "2012-12-31"

MONITOR_START = "2013-01-01"
MONITOR_END = "2013-12-31"

RETRAIN_START = "2011-01-01"
RETRAIN_END = "2013-12-31"

EVALUATION_START = "2014-01-01"
EVALUATION_END = "2014-12-31"

# Model configuration
RANDOM_STATE = 42
MAX_ITER = 1000

# The original model's 16 input features
NUMERICAL_FEATURES = [
    "Sales",
    "Quantity",
    "Discount",
    "Shipping Cost",
    "Order Year",
    "Order Month",
]

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