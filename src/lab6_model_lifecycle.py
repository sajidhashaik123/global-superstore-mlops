
from pathlib import Path
from mlflow.tracking import MlflowClient

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"

MODEL_NAME = "GlobalSuperstore_Profitability_Model"
MODEL_VERSION = "1"


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Retrieve registered model version
    version = client.get_model_version(
        MODEL_NAME,
        MODEL_VERSION
    )

    print("Model Name:", MODEL_NAME)
    print("Model Version:", version.version)

    # Update lifecycle status
    client.set_model_version_tag(
        MODEL_NAME,
        MODEL_VERSION,
        "lifecycle_status",
        "validated"
    )

    # Assign validation alias
    client.set_registered_model_alias(
        MODEL_NAME,
        "validation",
        MODEL_VERSION
    )

    # Verify lifecycle information
    updated_version = client.get_model_version(
        MODEL_NAME,
        MODEL_VERSION
    )

    print("Lifecycle Status: validated")
    print("Model Alias: validation")
    print("Model lifecycle management completed successfully!")


if __name__ == "__main__":
    main()
