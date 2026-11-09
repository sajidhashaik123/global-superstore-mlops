
import json

import mlflow

from src.lab13.config import (
    LAB13_REPORT_DIR,
    BASELINE_MODEL_DIR,
    RETRAINED_MODEL_DIR,
    LAB13_MODEL_DIR,
    BASELINE_MODEL_VERSION,
    RETRAINED_MODEL_VERSION,
)


# Isolated MLflow database for Experiment 13
MLFLOW_DB_PATH = LAB13_REPORT_DIR / "lab13_mlflow.db"
MLFLOW_ARTIFACT_DIR = LAB13_REPORT_DIR / "mlflow_artifacts"

EXPERIMENT_NAME = "Global_Superstore_Lab13_E2E"

LAB13_MODEL_VERSION = "lab13-v3"

MODEL_DIRECTORIES = {
    BASELINE_MODEL_VERSION: BASELINE_MODEL_DIR,
    RETRAINED_MODEL_VERSION: RETRAINED_MODEL_DIR,
    LAB13_MODEL_VERSION: LAB13_MODEL_DIR / "v3",
}

EVALUATION_REPORT_PATH = (
    LAB13_REPORT_DIR / "model_evaluation_comparison.json"
)


def setup_mlflow():
    """Configure isolated Experiment 13 MLflow tracking."""

    LAB13_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    print("MLflow tracking configured successfully.")
    print("Experiment:", experiment.name)
    print("Experiment ID:", experiment.experiment_id)
    print("Tracking database:", MLFLOW_DB_PATH)

    return experiment


def load_evaluation_report():
    """Load the Experiment 13 three-model evaluation report."""

    if not EVALUATION_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {EVALUATION_REPORT_PATH}"
        )

    with open(
        EVALUATION_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def log_model_run(
    model_version,
    model_dir,
    metrics,
    evaluation_report,
    lifecycle_status,
):
    """Log model parameters, metrics and artifacts."""

    metadata_path = model_dir / "training_metadata.json"

    required_artifacts = [
        metadata_path,
        model_dir / "preprocessor.joblib",
        model_dir / "classifier.joblib",
    ]

    for artifact in required_artifacts:
        if not artifact.exists():
            raise FileNotFoundError(
                f"Model artifact missing: {artifact}"
            )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    with mlflow.start_run(run_name=model_version):

        # Model parameters
        mlflow.log_param("model_version", model_version)
        mlflow.log_param(
            "algorithm",
            metadata.get("algorithm", "Not specified"),
        )
        mlflow.log_param(
            "target_column",
            evaluation_report["target_column"],
        )
        mlflow.log_param(
            "evaluation_dataset",
            evaluation_report["evaluation_dataset"],
        )
        mlflow.log_param(
            "evaluation_period",
            evaluation_report["evaluation_period"],
        )
        mlflow.log_param(
            "evaluation_records",
            evaluation_report["evaluation_records"],
        )
        mlflow.log_param("lifecycle_status", lifecycle_status)

        # Evaluation metrics
        for metric_name in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]:
            mlflow.log_metric(
                metric_name,
                float(metrics[metric_name]),
            )

        # Model tracking tags
        mlflow.set_tag("dataset", "Global Superstore")
        mlflow.set_tag("experiment", "Experiment 13")
        mlflow.set_tag("model_lifecycle", lifecycle_status)
        mlflow.set_tag("model_version", model_version)

        # Model artifacts
        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="model_metadata",
        )

        mlflow.log_artifact(
            str(EVALUATION_REPORT_PATH),
            artifact_path="evaluation_reports",
        )

        mlflow.log_artifact(
            str(model_dir / "preprocessor.joblib"),
            artifact_path="model_artifacts",
        )

        mlflow.log_artifact(
            str(model_dir / "classifier.joblib"),
            artifact_path="model_artifacts",
        )

        print(f"\nLogged MLflow run: {model_version}")
        print("Lifecycle status:", lifecycle_status)
        print("Accuracy:", metrics["accuracy"])
        print("F1-score:", metrics["f1_score"])
        print("ROC-AUC:", metrics["roc_auc"])


def run_mlflow_tracking():
    """Track all three Global Superstore model versions."""

    setup_mlflow()

    report = load_evaluation_report()

    model_statuses = {
        BASELINE_MODEL_VERSION: "production",
        RETRAINED_MODEL_VERSION: "candidate_not_promoted",
        LAB13_MODEL_VERSION: "candidate_not_promoted",
    }

    for model_version, model_dir in MODEL_DIRECTORIES.items():

        if model_version not in report["models"]:
            raise ValueError(
                f"Evaluation metrics missing for {model_version}"
            )

        log_model_run(
            model_version=model_version,
            model_dir=model_dir,
            metrics=report["models"][model_version],
            evaluation_report=report,
            lifecycle_status=model_statuses[model_version],
        )

    print("\n" + "=" * 50)
    print("EXPERIMENT 13 MLFLOW TRACKING COMPLETED")
    print("Total model versions tracked:", len(MODEL_DIRECTORIES))
    print("Models:", ", ".join(MODEL_DIRECTORIES.keys()))
    print("=" * 50)


if __name__ == "__main__":
    run_mlflow_tracking()
