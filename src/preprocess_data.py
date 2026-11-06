
from pathlib import Path
import pandas as pd


def preprocess_dataset():

    # 1. Get project root directory
    project_root = Path(__file__).resolve().parent.parent

    # 2. Load original Global Superstore dataset
    raw_file = (
        project_root / "data" / "raw" / "global_superstore.csv"
    )

    if not raw_file.exists():
        raise FileNotFoundError("Global Superstore dataset not found!")

    df = pd.read_csv(raw_file)

    print("Original dataset shape:", df.shape)

    # 3. Remove Postal Code due to excessive missing values
    df = df.drop(columns=["Postal Code"])

    # 4. Convert date columns into datetime format
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d-%m-%Y",
        errors="raise"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d-%m-%Y",
        errors="raise"
    )

    # 5. Remove leading and trailing spaces from text columns
    text_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # 6. Check duplicate rows
    print("Duplicate rows:", df.duplicated().sum())

    # 7. Check missing values
    print("\nMissing values after preprocessing:")
    print(df.isnull().sum().sum())

    # 8. Create processed dataset directory
    output_dir = (
        project_root / "data" / "processed" / "v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 9. Save processed dataset
    output_file = output_dir / "global_superstore_clean.csv"

    df.to_csv(output_file, index=False)

    # 10. Display final results
    print("\n--- PREPROCESSING SUMMARY ---")
    print("Original dataset shape: (51290, 24)")
    print("Processed dataset shape:", df.shape)
    print("Remaining missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("\nPreprocessing completed successfully!")
    print("Saved file:", output_file)


if __name__ == "__main__":
    preprocess_dataset()
