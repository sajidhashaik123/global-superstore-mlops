
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"

RUN_ID = "e79dd230a7b9431598cc59f33f8b96a0"
MODEL_NAME = "GlobalSuperstore_Profitability_Model"
ARTIFACT_PATH = "profitability_model"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    model_uri = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"

    # Check whether this run is already registered
    existing_versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'"
    )

    for version in existing_versions:
        if version.run_id == RUN_ID:
            print("This model run is already registered.")
            print("Model name:", MODEL_NAME)
            print("Model version:", version.version)
            return

    # Register the existing MLflow model
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("\nModel registered successfully!")
    print("Model name:", registered_model.name)
    print("Model version:", registered_model.version)
    print("Source run ID:", RUN_ID)


if __name__ == "__main__":
    main()
