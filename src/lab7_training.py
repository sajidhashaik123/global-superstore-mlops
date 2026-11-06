
import json
import joblib
import pandas as pd

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression

from lab7_config import (
    PROCESSED_DIR,
    MODEL_DIR,
    RANDOM_STATE
)


def train_model():

    # Load validated training data
    X_train = load_npz(
        PROCESSED_DIR / "X_train_transformed.npz"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).iloc[:, 0]

    # Create model
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    # Train model
    model.fit(X_train, y_train)

    # Save trained classifier
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "classifier.joblib"
    joblib.dump(model, model_path)

    # Save training metadata
    metadata = {
        "experiment": "Experiment 7 - ML Pipeline Engineering",
        "dataset": "Global Superstore",
        "dataset_version": "v1",
        "target": "Profitability",
        "algorithm": "Logistic Regression",
        "preprocessing": "MinMaxScaler + OneHotEncoder",
        "training_rows": int(X_train.shape[0]),
        "input_features_before_encoding": 16,
        "features_after_encoding": int(X_train.shape[1]),
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "model_file": "classifier.joblib"
    }

    metadata_path = MODEL_DIR / "training_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("LAB 7 MODEL TRAINING")
    print("------------------------------")
    print("Algorithm: Logistic Regression")
    print("Training rows:", X_train.shape[0])
    print("Transformed features:", X_train.shape[1])
    print("Training completed successfully!")
    print("Model saved to:", model_path)
    print("Metadata saved to:", metadata_path)

    return model


if __name__ == "__main__":
    train_model()
