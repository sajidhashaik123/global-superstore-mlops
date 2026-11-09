
from datetime import datetime, timezone

import mlflow

from src.lab13.validation import run_integration_validation
from src.lab13.mlflow_tracking import setup_mlflow
from src.lab13.lifecycle import load_registry
from src.lab13.config import (
    LAB13_REPORT_DIR,
    BASELINE_MODEL_VERSION,
    RETRAINED_MODEL_VERSION,
)
from src.lab13.pipeline import save_json


LAB13_MODEL_VERSION = "lab13-v3"

REQUIRED_MODEL_VERSIONS = {
    BASELINE_MODEL_VERSION,
    RETRAINED_MODEL_VERSION,
    LAB13_MODEL_VERSION,
}


def verify_mlflow_runs():
    """Verify that all three model versions are recorded in MLflow."""

    experiment = setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    tracked_models = {
        run.data.params.get("model_version")
        for run in runs
        if run.data.params.get("model_version")
    }

    missing_models = REQUIRED_MODEL_VERSIONS - tracked_models

    if missing_models:
        raise ValueError(
            f"Missing MLflow model runs: {sorted(missing_models)}"
        )

    return {
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "total_runs": len(runs),
        "tracked_models": sorted(REQUIRED_MODEL_VERSIONS),
        "status": "passed",
    }


def verify_model_lifecycle():
    """Verify model versions and production selection."""

    registry = load_registry()

    production_model = registry["current_production_model"]

    if production_model != BASELINE_MODEL_VERSION:
        raise ValueError(
            "Unexpected production model: "
            f"{production_model}"
        )

    model_statuses = {
        version: details["status"]
        for version, details in registry["models"].items()
    }

    expected_statuses = {
        BASELINE_MODEL_VERSION: "production",
        RETRAINED_MODEL_VERSION: "candidate_not_promoted",
        LAB13_MODEL_VERSION: "candidate_not_promoted",
    }

    for version, expected_status in expected_statuses.items():
        actual_status = model_statuses.get(version)

        if actual_status != expected_status:
            raise ValueError(
                f"Incorrect lifecycle status for {version}: "
                f"{actual_status}"
            )

    return {
        "production_model": production_model,
        "model_versions": model_statuses,
        "status": "passed",
    }


def verify_deployment_files():
    """Verify that separate Experiment 13 deployment files exist."""

    from src.lab13.config import PROJECT_ROOT

    required_files = [
        PROJECT_ROOT / "Dockerfile.lab13",
        PROJECT_ROOT / "k8s" / "lab13" / "deployment.yaml",
        PROJECT_ROOT / "k8s" / "lab13" / "service.yaml",
        PROJECT_ROOT / "k8s" / "lab13" / "pvc.yaml",
    ]

    missing_files = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Deployment files missing: {missing_files}"
        )

    return {
        "verified_files": [
            str(path.relative_to(PROJECT_ROOT))
            for path in required_files
        ],
        "status": "passed",
    }


def run_end_to_end_pipeline():
    """Execute Experiment 13 integration checks."""

    print("\nGLOBAL SUPERSTORE - END-TO-END MLOPS PIPELINE")
    print("=" * 60)

    results = {
        "project": "Global Superstore MLOps",
        "experiment": "Experiment 13",
        "execution_time": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    # Stage 1: Dataset, models and evaluation validation
    print("\nSTAGE 1: Integration validation")

    run_integration_validation()

    results["stages"]["integration_validation"] = {
        "status": "passed",
    }

    # Stage 2: MLflow tracking verification
    print("\nSTAGE 2: MLflow tracking verification")

    mlflow_result = verify_mlflow_runs()
    results["stages"]["mlflow_tracking"] = mlflow_result

    print("MLflow experiment:", mlflow_result["experiment_name"])
    print("Total MLflow runs:", mlflow_result["total_runs"])
    print("Verified models:", mlflow_result["tracked_models"])
    print("MLflow verification: PASSED")

    # Stage 3: Model lifecycle verification
    print("\nSTAGE 3: Model lifecycle verification")

    lifecycle_result = verify_model_lifecycle()
    results["stages"]["model_lifecycle"] = lifecycle_result

    print("Production model:", lifecycle_result["production_model"])
    print("Model lifecycle verification: PASSED")

    # Stage 4: Deployment configuration verification
    print("\nSTAGE 4: Deployment configuration verification")

    deployment_result = verify_deployment_files()
    results["stages"]["deployment_configuration"] = deployment_result

    print("Verified deployment files:")
    for file_path in deployment_result["verified_files"]:
        print(" -", file_path)

    print("Deployment configuration verification: PASSED")

    # Final execution report
    results["overall_status"] = "passed"

    report_path = LAB13_REPORT_DIR / "e2e_execution_report.json"
    save_json(results, report_path)

    print("\n" + "=" * 60)
    print("END-TO-END INTEGRATION CHECKS SUCCESSFUL")
    print("Execution report:", report_path)
    print("Overall status: PASSED")

    return results


if __name__ == "__main__":
    run_end_to_end_pipeline()
