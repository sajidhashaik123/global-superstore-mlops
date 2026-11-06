
import json
import joblib
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from lab7_config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR


def evaluate_model():

    # Load trained model
    model = joblib.load(MODEL_DIR / "classifier.joblib")

    # Load test data
    X_test = load_npz(
        PROCESSED_DIR / "X_test_transformed.npz"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).iloc[:, 0]

    # Generate predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        )
    }

    # Generate confusion matrix
    matrix = confusion_matrix(y_test, predictions)

    # Generate classification report
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True
    )

    # Create report directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(
        REPORT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save confusion matrix
    pd.DataFrame(
        matrix,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    # Display results
    print("LAB 7 MODEL EVALUATION")
    print("------------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nEvaluation reports saved successfully.")
    print("Report directory:", REPORT_DIR)

    return metrics


if __name__ == "__main__":
    evaluate_model()
