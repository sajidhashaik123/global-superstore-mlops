
import json
import joblib
import pandas as pd

from datetime import datetime, timezone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.lab13.config import (
    PROJECT_ROOT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    BASELINE_MODEL_DIR,
    RETRAINED_MODEL_DIR,
    LAB13_MODEL_DIR,
    LAB13_REPORT_DIR,
)


EVALUATION_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lab12"
    / "evaluation_2014.csv"
)

MODEL_VERSIONS = {
    "lab7-v1": BASELINE_MODEL_DIR,
    "lab12-v2": RETRAINED_MODEL_DIR,
    "lab13-v3": LAB13_MODEL_DIR / "v3",
}


def evaluate_model(model_dir, X_test, y_test, model_name):
    """Evaluate a model using its matching preprocessing artifact."""

    preprocessor = joblib.load(
        model_dir / "preprocessor.joblib"
    )

    classifier = joblib.load(
        model_dir / "classifier.joblib"
    )

    X_transformed = preprocessor.transform(X_test)

    predictions = classifier.predict(X_transformed)

    probabilities = classifier.predict_proba(
        X_transformed
    )[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True,
        ),
    }

    print(f"\n{model_name} RESULTS")
    print("-" * 45)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def main():
    print("\nEXPERIMENT 13 - MODEL EVALUATION")
    print("=" * 50)

    if not EVALUATION_DATA.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_DATA}"
        )

    df = pd.read_csv(EVALUATION_DATA)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing evaluation columns: {missing_columns}"
        )

    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError(
            "Missing values found in evaluation dataset."
        )

    X_test = df[FEATURE_COLUMNS]
    y_test = df[TARGET_COLUMN]

    print("Dataset: Global Superstore")
    print("Evaluation period: 2014")
    print("Evaluation records:", len(df))

    all_metrics = {}

    for model_version, model_dir in MODEL_VERSIONS.items():
        print(f"\nEvaluating {model_version}...")

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}"
            )

        all_metrics[model_version] = evaluate_model(
            model_dir,
            X_test,
            y_test,
            model_version,
        )

    comparison = {
        "project": "Global Superstore MLOps",
        "experiment": "Experiment 13",
        "evaluation_dataset": "evaluation_2014.csv",
        "evaluation_period": "2014-01-01 to 2014-12-31",
        "evaluation_records": len(df),
        "target_column": TARGET_COLUMN,
        "target_definition": "1 if Profit > 0, otherwise 0",
        "models": all_metrics,
        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    LAB13_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        LAB13_REPORT_DIR / "model_evaluation_comparison.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(comparison, file, indent=4)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON COMPLETED")
    print("Models evaluated:", len(all_metrics))
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()
