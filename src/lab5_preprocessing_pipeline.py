
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "global_superstore.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lab5"
    / "global_superstore_processed.csv"
)


def preprocess_dataset(df):
    # Work on a copy to preserve the original data
    processed_df = df.copy()

    # Remove Postal Code because it contains many missing values
    processed_df = processed_df.drop(columns=["Postal Code"])

    # Convert date columns
    for column in ["Order Date", "Ship Date"]:
        processed_df[column] = pd.to_datetime(
            processed_df[column],
            format="%d-%m-%Y",
            errors="raise"
        )

    # Remove extra whitespace from text columns
    text_columns = processed_df.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in text_columns:
        processed_df[column] = processed_df[column].str.strip()

    return processed_df


def main():
    df = pd.read_csv(RAW_PATH)

    processed_df = preprocess_dataset(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing completed successfully!")
    print("Original shape:", df.shape)
    print("Processed shape:", processed_df.shape)
    print("Processed missing values:", processed_df.isnull().sum().sum())
    print("Output saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
