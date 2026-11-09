import json
import hashlib
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.lab13.config import (
    DATASET_PATH,
    DATASET_MANIFEST_PATH,
    BASELINE_MODEL_DIR,
    RETRAINED_MODEL_DIR,
    LAB13_REPORT_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PROFIT_COLUMN,
    PROFITABILITY_THRESHOLD,
    BASELINE_MODEL_VERSION,
    RETRAINED_MODEL_VERSION,
    INITIAL_PRODUCTION_MODEL,
)


def calculate_sha256(file_path):
    """Calculate the SHA256 checksum of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path):
    """Save a JSON report."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def validate_dataset():
    """Validate the Global Superstore dataset and its required columns."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not DATASET_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {DATASET_MANIFEST_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)
        # Feature engineering from Order Date
    dataset["Order Date"] = pd.to_datetime(
        dataset["Order Date"],
        errors="raise"
    )

    dataset["Order Year"] = dataset["Order Date"].dt.year
    dataset["Order Month"] = dataset["Order Date"].dt.month
    manifest = load_json(DATASET_MANIFEST_PATH)

    required_columns = FEATURE_COLUMNS + [PROFIT_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if dataset[required_columns].isnull().any().any():
        raise ValueError("Missing values found in required dataset columns.")

    if not pd.api.types.is_numeric_dtype(dataset[PROFIT_COLUMN]):
        raise ValueError("Profit column must contain numeric values.")

    dataset[TARGET_COLUMN] = (
        dataset[PROFIT_COLUMN] > PROFITABILITY_THRESHOLD
    ).astype(int)

    result = {
        "dataset": "Global Superstore",
        "dataset_path": str(DATASET_PATH),
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_definition": "1 if Profit > 0, otherwise 0",
        "missing_values_in_required_columns": 0,
        "dataset_sha256": calculate_sha256(DATASET_PATH),
        "manifest_available": True,
        "manifest_dataset": manifest.get("dataset", "Not specified"),
        "status": "passed",
    }

    return dataset, result


def verify_model_artifacts(model_dir, expected_version):
    """Verify that a model version has both required artifacts."""

    preprocessor_path = model_dir / "preprocessor.joblib"
    classifier_path = model_dir / "classifier.joblib"
    metadata_path = model_dir / "training_metadata.json"

    for path in [preprocessor_path, classifier_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact missing: {path}")

    preprocessor = joblib.load(preprocessor_path)
    classifier = joblib.load(classifier_path)
    metadata = load_json(metadata_path)

    return {
        "model_version": expected_version,
        "preprocessor": preprocessor,
        "classifier": classifier,
        "metadata": metadata,
        "preprocessor_sha256": calculate_sha256(preprocessor_path),
        "classifier_sha256": calculate_sha256(classifier_path),
        "metadata_sha256": calculate_sha256(metadata_path),
        "status": "verified",
    }