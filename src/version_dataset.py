
from pathlib import Path
import pandas as pd
import hashlib
import json
from datetime import datetime, timezone


def calculate_checksum(file_path):
    """Calculate SHA-256 checksum of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_file_details(file_path):
    """Collect file information."""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "file_size_bytes": file_path.stat().st_size,
        "sha256": calculate_checksum(file_path)
    }


def create_dataset_manifest():

    # 1. Get project root
    project_root = Path(__file__).resolve().parent.parent

    # 2. Define dataset paths
    raw_file = (
        project_root / "data" / "raw" / "global_superstore.csv"
    )

    processed_file = (
        project_root / "data" / "processed" / "v1"
        / "global_superstore_clean.csv"
    )

    train_file = (
        project_root / "data" / "splits" / "v1" / "train.csv"
    )

    test_file = (
        project_root / "data" / "splits" / "v1" / "test.csv"
    )

    # 3. Collect file details
    raw_details = get_file_details(raw_file)
    processed_details = get_file_details(processed_file)
    train_details = get_file_details(train_file)
    test_details = get_file_details(test_file)

    # 4. Verify dataset consistency
    if train_details["rows"] + test_details["rows"] != processed_details["rows"]:
        raise ValueError("Training and testing row counts do not match!")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    if set(train_df["Row ID"]).intersection(set(test_df["Row ID"])):
        raise ValueError("Training and testing datasets contain overlapping rows!")

    # 5. Create version information
    manifest = {
        "project_name": "Global Superstore MLOps Project",
        "dataset_name": "Global Superstore",
        "dataset_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "raw_dataset": raw_details,

        "processed_dataset": {
            **processed_details,
            "operations": [
                "Removed Postal Code due to excessive missing values",
                "Converted Order Date to datetime",
                "Converted Ship Date to datetime",
                "Removed leading and trailing spaces from text columns"
            ]
        },

        "dataset_split": {
            "training_dataset": train_details,
            "testing_dataset": test_details,
            "test_size": 0.20,
            "train_size": 0.80,
            "random_state": 42,
            "shuffle": True,
            "overlapping_row_ids": 0
        }
    }

    # 6. Save manifest
    output_file = (
        project_root / "data" / "processed" / "v1"
        / "manifest.json"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print("\n--- DATASET VERSIONING SUMMARY ---")
    print("Dataset name:", manifest["dataset_name"])
    print("Dataset version:", manifest["dataset_version"])

    print("\nRaw dataset rows:", raw_details["rows"])
    print("Processed dataset rows:", processed_details["rows"])
    print("Training dataset rows:", train_details["rows"])
    print("Testing dataset rows:", test_details["rows"])

    print("\nRandom state:", manifest["dataset_split"]["random_state"])
    print("Overlapping Row IDs: 0")

    print("\nSHA-256 checksums generated successfully.")
    print("Manifest saved at:", output_file)

    print("\nDataset versioning completed successfully!")


if __name__ == "__main__":
    create_dataset_manifest()
