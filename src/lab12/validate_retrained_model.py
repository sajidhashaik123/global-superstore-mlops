import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from lab12.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LAB12_MODEL_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = LAB12_MODEL_DIR / "v2"

EVALUATION_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lab12"
    / "evaluation_2014.csv"
)


def validate_model():

    print("Loading retrained model artifacts...")

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.joblib"
    )

    classifier = joblib.load(
        MODEL_DIR / "classifier.joblib"
    )

    print("Preprocessor loaded successfully.")
    print("Classifier loaded successfully.")

    print("\nLoading evaluation dataset...")

    df = pd.read_csv(EVALUATION_DATA)

    X_test = df[FEATURE_COLUMNS]
    y_test = df[TARGET_COLUMN]

    print(f"Evaluation records: {len(df)}")

    print("\nTransforming input features...")

    X_transformed = preprocessor.transform(X_test)

    print(
        f"Transformed feature shape: {X_transformed.shape}"
    )

    print("\nGenerating predictions...")

    predictions = classifier.predict(X_transformed)

    probabilities = classifier.predict_proba(
        X_transformed
    )[:, 1]

    assert len(predictions) == len(y_test)
    assert len(probabilities) == len(y_test)

    assert set(np.unique(predictions)).issubset({0, 1})

    assert np.all(np.isfinite(probabilities))
    assert np.all(
        (probabilities >= 0) & (probabilities <= 1)
    )

    print("\nVALIDATION RESULTS")
    print("-" * 40)

    print(f"Model version: lab12-v2")
    print(f"Predictions generated: {len(predictions)}")
    print(f"Actual labels: {len(y_test)}")
    print(f"Unique predictions: {np.unique(predictions)}")
    print(
        f"Minimum probability: {probabilities.min():.4f}"
    )
    print(
        f"Maximum probability: {probabilities.max():.4f}"
    )

    print("\nAll validation checks passed successfully.")


if __name__ == "__main__":
    validate_model()