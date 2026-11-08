import pandas as pd

from config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PROFIT_COLUMN,
    PROFITABILITY_THRESHOLD,
    REFERENCE_START,
    REFERENCE_END,
    MONITOR_START,
    MONITOR_END,
    RETRAIN_START,
    RETRAIN_END,
    EVALUATION_START,
    EVALUATION_END,
    PROJECT_ROOT,
)


def prepare_dataset():
    print("Loading Global Superstore dataset...")

    df = pd.read_csv(DATASET_PATH)

    # Convert order date
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Create the actual target using the original target definition
    df[TARGET_COLUMN] = (
        df[PROFIT_COLUMN] > PROFITABILITY_THRESHOLD
    ).astype(int)

    # Extract date-based features
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month

    # Select model features, target and date
    required_columns = (
        ["Order Date"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    df = df[required_columns].copy()

    # Create chronological datasets
    reference = df[
        (df["Order Date"] >= REFERENCE_START)
        & (df["Order Date"] <= REFERENCE_END)
    ].copy()

    monitoring = df[
        (df["Order Date"] >= MONITOR_START)
        & (df["Order Date"] <= MONITOR_END)
    ].copy()

    retraining = df[
        (df["Order Date"] >= RETRAIN_START)
        & (df["Order Date"] <= RETRAIN_END)
    ].copy()

    evaluation = df[
        (df["Order Date"] >= EVALUATION_START)
        & (df["Order Date"] <= EVALUATION_END)
    ].copy()

    # Save prepared datasets
    output_dir = PROJECT_ROOT / "data" / "processed" / "lab12"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "reference_2011_2012.csv": reference,
        "monitoring_2013.csv": monitoring,
        "retraining_2011_2013.csv": retraining,
        "evaluation_2014.csv": evaluation,
    }

    print("\nSaving chronological datasets...")

    for filename, dataset in datasets.items():
        output_path = output_dir / filename
        dataset.to_csv(output_path, index=False)

        print(
            f"{filename}: {len(dataset)} records saved"
        )

    print("\nDataset preparation completed.")
    print("Original dataset records:", len(df))
    print("Selected features:", len(FEATURE_COLUMNS))
    print("Target:", TARGET_COLUMN)
    print("Target rule: Profit > 0")

    print("\nProfitability distribution:")

    for name, dataset in datasets.items():
        print(f"\n{name}")
        print(
            dataset[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .to_string()
        )


if __name__ == "__main__":
    prepare_dataset()