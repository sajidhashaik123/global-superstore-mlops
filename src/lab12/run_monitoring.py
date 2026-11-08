import json
import joblib
import pandas as pd

from config import (
    BASELINE_MODEL_DIR,
    PROJECT_ROOT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

from drift_detection import (
    detect_numeric_drift,
    detect_categorical_drift,
    detect_prediction_drift,
    detect_concept_drift,
    save_drift_report,
)


def load_dataset(filename):
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "lab12"
        / filename
    )
    return pd.read_csv(path)


def main():
    print("Loading baseline model...")

    preprocessor = joblib.load(
        BASELINE_MODEL_DIR / "preprocessor.joblib"
    )

    classifier = joblib.load(
        BASELINE_MODEL_DIR / "classifier.joblib"
    )

    print("Baseline model loaded successfully.")

    print("\nLoading reference and monitoring datasets...")

    reference = load_dataset("reference_2011_2012.csv")
    monitoring = load_dataset("monitoring_2013.csv")

    X_reference = reference[FEATURE_COLUMNS]
    y_reference = reference[TARGET_COLUMN]

    X_monitoring = monitoring[FEATURE_COLUMNS]
    y_monitoring = monitoring[TARGET_COLUMN]

    print("Reference records:", len(reference))
    print("Monitoring records:", len(monitoring))

    print("\nGenerating predictions...")

    reference_transformed = preprocessor.transform(X_reference)
    monitoring_transformed = preprocessor.transform(X_monitoring)

    reference_probabilities = classifier.predict_proba(
        reference_transformed
    )[:, 1]

    monitoring_probabilities = classifier.predict_proba(
        monitoring_transformed
    )[:, 1]

    reference_predictions = classifier.predict(
        reference_transformed
    )

    monitoring_predictions = classifier.predict(
        monitoring_transformed
    )

    print("Predictions generated successfully.")

    print("\nChecking numerical feature drift...")

    numerical_results = detect_numeric_drift(
        X_reference,
        X_monitoring,
    )

    print("\nChecking categorical feature drift...")

    categorical_results = detect_categorical_drift(
        X_reference,
        X_monitoring,
    )

    print("\nChecking prediction drift...")

    prediction_results = detect_prediction_drift(
        reference_probabilities,
        monitoring_probabilities,
    )

    print("\nEvaluating reference model performance...")

    reference_performance = detect_concept_drift(
        y_reference,
        reference_predictions,
    )

    print("\nEvaluating monitoring model performance...")

    monitoring_performance = detect_concept_drift(
        y_monitoring,
        monitoring_predictions,
    )

    performance_comparison = {}

    for metric in reference_performance:
        performance_comparison[metric] = {
            "reference": reference_performance[metric],
            "monitoring": monitoring_performance[metric],
            "change": (
                monitoring_performance[metric]
                - reference_performance[metric]
            ),
        }

    results = {
        "experiment": "Experiment 12 - Monitoring and Drift Detection",
        "model_version": "lab7-v1",
        "dataset": "Global Superstore",
        "reference_period": "2011-2012",
        "monitoring_period": "2013",
        "reference_records": len(reference),
        "monitoring_records": len(monitoring),
        "data_drift": {
            "numerical_features": numerical_results,
            "categorical_features": categorical_results,
        },
        "prediction_drift": prediction_results,
        "performance": {
            "reference": reference_performance,
            "monitoring": monitoring_performance,
            "comparison": performance_comparison,
        },
        "limitations": [
            "The baseline model was trained using a random split spanning all four years.",
            "The 2013 monitoring evaluation is not an independent temporal backtest.",
            "Statistical drift signals do not by themselves prove model degradation.",
        ],
    }

    print("\nSaving monitoring report...")

    save_drift_report(results)

    print("\n========== MONITORING SUMMARY ==========")

    numeric_drift_count = sum(
        item["drift_detected"] for item in numerical_results
    )

    categorical_drift_count = sum(
        item["drift_detected"] for item in categorical_results
    )

    print(
        "Numerical features with drift:",
        numeric_drift_count,
        "/",
        len(numerical_results),
    )

    print(
        "Categorical features with drift:",
        categorical_drift_count,
        "/",
        len(categorical_results),
    )

    print(
        "Prediction drift detected:",
        prediction_results["drift_detected"],
    )

    print("\nReference performance:")
    print(json.dumps(reference_performance, indent=4))

    print("\nMonitoring performance:")
    print(json.dumps(monitoring_performance, indent=4))

    print("\nMonitoring completed successfully.")


if __name__ == "__main__":
    main()