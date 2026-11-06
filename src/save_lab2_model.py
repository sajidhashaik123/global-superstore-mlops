
from pathlib import Path
import json
import joblib
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lab2_preprocessing import build_preprocessor


def save_model():

    project_root = Path(__file__).resolve().parent.parent

    # Load the selected 70:30 dataset split
    train_file = (
        project_root / "data" / "splits" / "lab2"
        / "70_30" / "train.csv"
    )

    test_file = (
        project_root / "data" / "splits" / "lab2"
        / "70_30" / "test.csv"
    )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=["Profitability"])
    y_train = train_df["Profitability"]

    X_test = test_df.drop(columns=["Profitability"])
    y_test = test_df["Profitability"]

    # Build and train the complete pipeline
    model = Pipeline([
        ("preprocessing", build_preprocessor("minmax")),
        ("model", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 6),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0), 6
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0), 6
        ),
        "f1_score": round(
            f1_score(y_test, predictions, zero_division=0), 6
        ),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 6)
    }

    # Create model version directory
    model_dir = project_root / "models" / "lab2" / "v1"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save trained pipeline
    model_file = model_dir / "profitability_model.joblib"
    joblib.dump(model, model_file)

    # Maintain model version information
    metadata = {
        "model_name": "Global Superstore Profitability Model",
        "model_version": "v1",
        "dataset": "Global Superstore",
        "target": "Profitability",
        "target_definition": "Profit > 0: 1, Profit <= 0: 0",
        "algorithm": "Logistic Regression",
        "preprocessing": "MinMaxScaler and OneHotEncoder",
        "train_test_split": "70:30",
        "random_state": 42,
        "metrics": metrics,
        "scikit_learn_version": sklearn.__version__,
        "model_file": "profitability_model.joblib"
    }

    metadata_file = model_dir / "model_metadata.json"

    with open(metadata_file, "w") as file:
        json.dump(metadata, file, indent=4)

    print("--- MODEL SERIALIZATION COMPLETED ---")
    print("Model saved:", model_file)
    print("Metadata saved:", metadata_file)

    print("\n--- MODEL METRICS ---")
    for metric, value in metrics.items():
        print(metric, ":", value)

    print("\nModel version:", metadata["model_version"])
    print("\nLab 2 Step 7 completed successfully!")


if __name__ == "__main__":
    save_model()
