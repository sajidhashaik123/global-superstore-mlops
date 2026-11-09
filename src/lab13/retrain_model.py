
import json
import hashlib
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.lab13.config import (
    PROJECT_ROOT,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    MAX_ITER,
)


# Experiment 13 training dataset
TRAINING_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lab12"
    / "retraining_2011_2013.csv"
)

# New Lab13 model version
MODEL_VERSION = "lab13-v3"

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "lab13"
    / "v3"
)


def calculate_sha256(file_path):
    """Calculate SHA256 checksum of the training dataset."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def retrain_model():
    """Train and save a new Lab13 profitability model."""

    print("\nEXPERIMENT 13 - MODEL RETRAINING")
    print("-" * 45)

    if not TRAINING_DATA.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_DATA}"
        )

    print("Loading Global Superstore training dataset...")

    df = pd.read_csv(TRAINING_DATA)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError(
            "Missing values found in training data."
        )

    X_train = df[FEATURE_COLUMNS]
    y_train = df[TARGET_COLUMN]

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError(
            "Profitability target must contain only 0 and 1."
        )

    print("Training records:", len(df))
    print("Input features:", len(FEATURE_COLUMNS))

    print("\nTarget distribution:")
    print(y_train.value_counts().sort_index())

    print("\nBuilding preprocessing pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                MinMaxScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("\nTraining Logistic Regression model...")

    pipeline.fit(X_train, y_train)

    transformed_features = (
        pipeline.named_steps["preprocessor"]
        .transform(X_train)
        .shape[1]
    )

    # Create only the new Lab13 model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving Lab13 model artifacts...")

    joblib.dump(
        pipeline.named_steps["preprocessor"],
        MODEL_DIR / "preprocessor.joblib",
    )

    joblib.dump(
        pipeline.named_steps["classifier"],
        MODEL_DIR / "classifier.joblib",
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "experiment": "Experiment 13",
        "dataset": "Global Superstore",
        "training_dataset": str(
            TRAINING_DATA.relative_to(PROJECT_ROOT)
        ),
        "training_dataset_sha256": calculate_sha256(
            TRAINING_DATA
        ),
        "training_period": "2011-01-01 to 2013-12-31",
        "training_records": int(len(df)),
        "raw_input_features": len(FEATURE_COLUMNS),
        "encoded_features": int(transformed_features),
        "target_column": TARGET_COLUMN,
        "target_definition": (
            "Profitability = 1 if Profit > 0, otherwise 0"
        ),
        "algorithm": "Logistic Regression",
        "numerical_preprocessing": "MinMaxScaler",
        "categorical_preprocessing": (
            "OneHotEncoder(handle_unknown='ignore')"
        ),
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(
        MODEL_DIR / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 45)
    print("MODEL RETRAINING COMPLETED SUCCESSFULLY")
    print("=" * 45)

    print("Model version:", MODEL_VERSION)
    print("Training records:", len(df))
    print("Encoded features:", transformed_features)
    print("Saved model directory:", MODEL_DIR)

    return metadata


if __name__ == "__main__":
    retrain_model()

