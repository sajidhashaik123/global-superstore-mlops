from pathlib import Path
import hashlib
import json
import mlflow


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def track_dataset_lineage():
    project_root = Path(__file__).resolve().parent.parent

    dataset_path = (
        project_root / "data" / "processed" / "v1"
        / "global_superstore_clean.csv"
    )

    manifest_path = (
        project_root / "data" / "processed" / "v1"
        / "manifest.json"
    )

    if not dataset_path.exists():
        raise FileNotFoundError("Processed dataset not found!")

    dataset_hash = calculate_hash(dataset_path)

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    mlflow.set_tracking_uri(
        f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
    )

    mlflow.set_experiment("Global_Superstore_Profitability")

    with mlflow.start_run(run_name="Dataset_Lineage_Tracking"):

        mlflow.log_param("dataset_name", "Global Superstore")
        mlflow.log_param("dataset_version", "v1")
        mlflow.log_param("target", "Profit > 0")
        mlflow.log_param("random_state", 42)

        mlflow.set_tag("dataset_source", "data/processed/v1")
        mlflow.set_tag("dataset_file", dataset_path.name)
        mlflow.set_tag("dataset_sha256", dataset_hash)

        mlflow.log_artifact(str(manifest_path))

        print("Dataset lineage tracked successfully!")
        print("Dataset version: v1")
        print("Dataset SHA-256:", dataset_hash)


if __name__ == "__main__":
    track_dataset_lineage()