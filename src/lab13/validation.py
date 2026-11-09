
import numpy as np

from src.lab13.pipeline import (
    validate_dataset,
    verify_model_artifacts,
    calculate_sha256,
    load_json,
)

from src.lab13.lifecycle import (
    LIFECYCLE_FILE,
    load_registry,
)

from src.lab13.config import (
    DATASET_PATH,
    DATASET_MANIFEST_PATH,
    BASELINE_MODEL_DIR,
    RETRAINED_MODEL_DIR,
    LAB13_MODEL_DIR,
    BASELINE_MODEL_VERSION,
    RETRAINED_MODEL_VERSION,
    FEATURE_COLUMNS,
    LAB12_COMPARISON_REPORT,
    LAB13_REPORT_DIR,
)


LAB13_MODEL_VERSION = "lab13-v3"
LAB13_MODEL_DIR_V3 = LAB13_MODEL_DIR / "v3"

LAB13_EVALUATION_REPORT = (
    LAB13_REPORT_DIR / "model_evaluation_comparison.json"
)


def validate_dataset_checksum():
    """Compare the processed dataset checksum with its manifest."""

    manifest = load_json(DATASET_MANIFEST_PATH)

    actual_hash = calculate_sha256(DATASET_PATH)
    expected_hash = manifest["processed_dataset"]["sha256"]

    if actual_hash.lower() != expected_hash.lower():
        raise ValueError("Dataset checksum does not match manifest.")

    return True


def validate_model_prediction(model_info, dataset):
    """Check whether a model can generate valid predictions."""

    preprocessor = model_info["preprocessor"]
    classifier = model_info["classifier"]

    sample = dataset[FEATURE_COLUMNS].head(5)

    transformed = preprocessor.transform(sample)
    predictions = classifier.predict(transformed)

    if len(predictions) != len(sample):
        raise ValueError("Prediction count does not match input count.")

    if not set(np.asarray(predictions).tolist()).issubset({0, 1}):
        raise ValueError("Model generated invalid prediction classes.")

    return {
        "model_version": model_info["model_version"],
        "sample_count": len(sample),
        "predictions": np.asarray(predictions).tolist(),
        "status": "passed",
    }


def validate_lifecycle_registry():
    """Verify all three model versions and lifecycle states."""

    if not LIFECYCLE_FILE.exists():
        raise FileNotFoundError("Model lifecycle registry is missing.")

    registry = load_registry()

    if registry["current_production_model"] != BASELINE_MODEL_VERSION:
        raise ValueError("Unexpected production model version.")

    expected_statuses = {
        BASELINE_MODEL_VERSION: "production",
        RETRAINED_MODEL_VERSION: "candidate_not_promoted",
        LAB13_MODEL_VERSION: "candidate_not_promoted",
    }

    for version, expected_status in expected_statuses.items():

        if version not in registry["models"]:
            raise ValueError(
                f"Model {version} is missing from lifecycle registry."
            )

        actual_status = registry["models"][version]["status"]

        if actual_status != expected_status:
            raise ValueError(
                f"Incorrect lifecycle status for {version}: "
                f"{actual_status}"
            )

    return True


def validate_evaluation_report():
    """Validate Lab12 and Lab13 evaluation reports."""

    if not LAB12_COMPARISON_REPORT.exists():
        raise FileNotFoundError(
            "Experiment 12 comparison report is missing."
        )

    if not LAB13_EVALUATION_REPORT.exists():
        raise FileNotFoundError(
            "Experiment 13 evaluation report is missing."
        )

    lab12_report = load_json(LAB12_COMPARISON_REPORT)
    lab13_report = load_json(LAB13_EVALUATION_REPORT)

    if not lab12_report:
        raise ValueError("Experiment 12 comparison report is empty.")

    if not lab13_report:
        raise ValueError("Experiment 13 evaluation report is empty.")

    required_models = {
        BASELINE_MODEL_VERSION,
        RETRAINED_MODEL_VERSION,
        LAB13_MODEL_VERSION,
    }

    if set(lab13_report.get("models", {}).keys()) != required_models:
        raise ValueError(
            "Experiment 13 evaluation report does not contain "
            "all three expected models."
        )

    return True


def run_integration_validation():
    """Execute all Experiment 13 integration validation checks."""

    print("\nEXPERIMENT 13 - INTEGRATION VALIDATION")
    print("-" * 50)

    dataset, dataset_result = validate_dataset()

    print("1. Dataset validation: PASSED")
    print("   Dataset rows:", dataset_result["rows"])

    validate_dataset_checksum()

    print("2. Dataset checksum validation: PASSED")

    baseline = verify_model_artifacts(
        BASELINE_MODEL_DIR,
        BASELINE_MODEL_VERSION,
    )

    print("3. Baseline model artifact validation: PASSED")

    retrained = verify_model_artifacts(
        RETRAINED_MODEL_DIR,
        RETRAINED_MODEL_VERSION,
    )

    print("4. Lab12 retrained model artifact validation: PASSED")

    lab13_model = verify_model_artifacts(
        LAB13_MODEL_DIR_V3,
        LAB13_MODEL_VERSION,
    )

    print("5. Lab13 retrained model artifact validation: PASSED")

    baseline_prediction = validate_model_prediction(
        baseline, dataset
    )

    print("6. Baseline model prediction: PASSED")
    print("   Predictions:", baseline_prediction["predictions"])

    retrained_prediction = validate_model_prediction(
        retrained, dataset
    )

    print("7. Lab12 model prediction: PASSED")
    print("   Predictions:", retrained_prediction["predictions"])

    lab13_prediction = validate_model_prediction(
        lab13_model, dataset
    )

    print("8. Lab13 model prediction: PASSED")
    print("   Predictions:", lab13_prediction["predictions"])

    validate_lifecycle_registry()

    print("9. Three-model lifecycle validation: PASSED")

    validate_evaluation_report()

    print("10. Evaluation report validation: PASSED")

    print("-" * 50)
    print("ALL 10 INTEGRATION VALIDATION CHECKS PASSED!")

    return True


if __name__ == "__main__":
    run_integration_validation()
