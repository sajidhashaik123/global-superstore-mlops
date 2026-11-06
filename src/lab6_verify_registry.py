
from pathlib import Path
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"

MODEL_NAME = "GlobalSuperstore_Profitability_Model"


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    version = client.get_model_version(MODEL_NAME, "1")

    print("MODEL REGISTRY VERIFICATION")
    print("---------------------------")
    print("Model:", MODEL_NAME)
    print("Version:", version.version)

    print("\nModel Tags:")
    for key, value in version.tags.items():
        print(f"{key}: {value}")

    alias_version = client.get_model_version_by_alias(
        MODEL_NAME, "validation"
    )

    print("\nValidation Alias Version:", alias_version.version)

    print("\nRegistry verification completed!")


if __name__ == "__main__":
    main()
