
from pathlib import Path
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"

MODEL_NAME = "GlobalSuperstore_Profitability_Model"
MODEL_VERSION = "1"

def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    metadata = {
        "algorithm": "Logistic Regression",
        "dataset": "Global Superstore",
        "dataset_version": "v1",
        "prediction_target": "Order Profitability",
        "preprocessing": "MinMaxScaler",
        "data_split": "80:20",
        "feature_selection": "Selected 100 features",
        "accuracy": "0.9190",
        "f1_score": "0.9468",
        "roc_auc": "0.9531",
        "experiment": "Experiment 4 - MLflow Tracking"
    }

    for key, value in metadata.items():
        client.set_model_version_tag(
            MODEL_NAME,
            MODEL_VERSION,
            key,
            value
        )

    version = client.get_model_version(
        MODEL_NAME, MODEL_VERSION
    )

    print("Model metadata added successfully!")
    print("Model:", version.name)
    print("Version:", version.version)
    print("\n--- MODEL METADATA ---")

    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
