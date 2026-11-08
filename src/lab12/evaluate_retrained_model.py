import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from lab12.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LAB12_REPORT_DIR,
    BASELINE_MODEL_DIR,
    LAB12_MODEL_DIR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DATA = (
    PROJECT_ROOT / "data" / "processed" /
    "lab12" / "evaluation_2014.csv"
)

BASELINE_DIR = BASELINE_MODEL_DIR
RETRAINED_DIR = LAB12_MODEL_DIR / "v2"


def evaluate_model(model_dir, X_test, y_test, model_name):

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
    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def main():

    print("Loading 2014 evaluation dataset...")

    df = pd.read_csv(EVALUATION_DATA)

    X_test = df[FEATURE_COLUMNS]
    y_test = df[TARGET_COLUMN]

    print(f"Evaluation records: {len(df)}")

    baseline_metrics = evaluate_model(
        BASELINE_DIR,
        X_test,
        y_test,
        "BASELINE MODEL (lab7-v1)",
    )

    retrained_metrics = evaluate_model(
        RETRAINED_DIR,
        X_test,
        y_test,
        "RETRAINED MODEL (lab12-v2)",
    )

    comparison = {
        "evaluation_dataset": "evaluation_2014.csv",
        "evaluation_period": "2014-01-01 to 2014-12-31",
        "evaluation_records": len(df),
        "target_column": TARGET_COLUMN,
        "baseline_model": "lab7-v1",
        "retrained_model": "lab12-v2",
        "baseline_metrics": baseline_metrics,
        "retrained_metrics": retrained_metrics,
    }

    LAB12_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        LAB12_REPORT_DIR / "retraining_comparison.json"
    )

    with report_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(comparison, file, indent=4)

    print("\nMODEL COMPARISON COMPLETED")
    print("-" * 40)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()