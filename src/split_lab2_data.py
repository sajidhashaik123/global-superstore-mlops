
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_lab2_dataset():

    project_root = Path(__file__).resolve().parent.parent

    input_file = (
        project_root / "data" / "processed" / "v1"
        / "lab2_selected_features.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            "Selected features not found! Complete Step 2 first."
        )

    df = pd.read_csv(input_file)

    X = df.drop(columns=["Profitability"])
    y = df["Profitability"]

    print("--- LAB 2 DATASET ---")
    print("Total records:", len(df))
    print("Input features:", X.shape[1])
    print("\nOriginal target distribution:")
    print(y.value_counts(normalize=True).round(4))

    split_configs = {
        "80_20": 0.20,
        "70_30": 0.30
    }

    for split_name, test_size in split_configs.items():

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
            shuffle=True
        )

        output_dir = (
            project_root / "data" / "splits" / "lab2" / split_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df = X_train.copy()
        train_df["Profitability"] = y_train

        test_df = X_test.copy()
        test_df["Profitability"] = y_test

        train_file = output_dir / "train.csv"
        test_file = output_dir / "test.csv"

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f"\n--- {split_name} SPLIT ---")
        print("Training records:", len(train_df))
        print("Testing records:", len(test_df))
        print("Training class distribution:")
        print(y_train.value_counts().sort_index())
        print("Testing class distribution:")
        print(y_test.value_counts().sort_index())
        print("Training file:", train_file)
        print("Testing file:", test_file)

    print("\nLab 2 Step 3 completed successfully!")


if __name__ == "__main__":
    split_lab2_dataset()
