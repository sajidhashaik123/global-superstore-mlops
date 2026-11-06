
from pathlib import Path
import pandas as pd

def prepare_lab2_dataset():

    project_root = Path(__file__).resolve().parent.parent

    input_file = (
        project_root / "data" / "processed" / "v1"
        / "global_superstore_clean.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            "Processed Global Superstore dataset not found!"
        )

    # Load processed dataset
    df = pd.read_csv(input_file)

    print("--- ORIGINAL DATASET ---")
    print("Dataset shape:", df.shape)

    # Create target variable
    df["Profitability"] = (df["Profit"] > 0).astype(int)

    # Remove original Profit column
    df = df.drop(columns=["Profit"])

    print("\n--- TARGET DISTRIBUTION ---")
    print(df["Profitability"].value_counts().sort_index())

    print("\n--- TARGET PERCENTAGES ---")
    print(
        (df["Profitability"].value_counts(normalize=True) * 100)
        .sort_index().round(2)
    )

    # Save prepared dataset
    output_dir = project_root / "data" / "processed" / "v1"
    output_file = output_dir / "lab2_profitability_dataset.csv"

    df.to_csv(output_file, index=False)

    print("\nPrepared dataset shape:", df.shape)
    print("Saved file:", output_file)
    print("\nLab 2 Step 1 completed successfully!")


if __name__ == "__main__":
    prepare_lab2_dataset()
