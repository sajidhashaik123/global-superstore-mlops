
from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
TRAIN_PATH = (
    PROJECT_ROOT / "data" / "splits" / "lab2" / "80_20" / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT / "data" / "splits" / "lab2" / "80_20" / "test.csv"
)

# Lab 7 output directories
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "lab7"
MODEL_DIR = PROJECT_ROOT / "models" / "lab7" / "v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "lab7"

# Target column
TARGET_COLUMN = "Profitability"

# Numerical input features
NUMERICAL_FEATURES = [
    "Sales",
    "Quantity",
    "Discount",
    "Shipping Cost",
    "Order Year",
    "Order Month"
]

# Categorical input features
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
    "Order Priority"
]

# Combined input features
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Reproducibility
RANDOM_STATE = 42
