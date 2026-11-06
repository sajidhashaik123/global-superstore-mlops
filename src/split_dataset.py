
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset():

    # 1. Get project root directory
    project_root = Path(__file__).resolve().parent.parent

    # 2. Load processed dataset
    input_file = (
        project_root
        / "data"
        / "processed"
        / "v1"
        / "global_superstore_clean.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            "Processed dataset not found! Run preprocess_data.py first."
        )

    df = pd.read_csv(input_file)

    print("Loaded dataset shape:", df.shape)

    # 3. Split dataset into training and testing sets
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # 4. Create versioned split directory
    output_dir = project_root / "data" / "splits" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Save training dataset
    train_file = output_dir / "train.csv"
    train_df.to_csv(train_file, index=False)

    # 6. Save testing dataset
    test_file = output_dir / "test.csv"
    test_df.to_csv(test_file, index=False)

    # 7. Verify split sizes
    print("\n--- DATASET SPLIT SUMMARY ---")
    print("Original rows:", len(df))
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))

    # 8. Verify no overlapping Row IDs
    train_ids = set(train_df["Row ID"])
    test_ids = set(test_df["Row ID"])

    overlap = train_ids.intersection(test_ids)

    print("Overlapping Row IDs:", len(overlap))

    # 9. Verify all rows are accounted for
    print("Total split rows:", len(train_df) + len(test_df))

    print("\nTraining dataset saved:", train_file)
    print("Testing dataset saved:", test_file)

    print("\nDataset splitting completed successfully!")


if __name__ == "__main__":
    split_dataset()
