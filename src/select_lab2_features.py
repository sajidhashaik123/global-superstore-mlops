
from pathlib import Path
import pandas as pd

def select_lab2_features():

    project_root = Path(__file__).resolve().parent.parent

    input_file = (
        project_root / "data" / "processed" / "v1"
        / "lab2_profitability_dataset.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            "Lab 2 prepared dataset not found! Run Step 1 first."
        )

    df = pd.read_csv(input_file)

    print("--- LOADED DATASET ---")
    print("Shape:", df.shape)

    # Convert Order Date into datetime
    df["Order Date"] = pd.to_datetime(
        df["Order Date"], errors="raise"
    )

    # Create date-based features
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month

    # Select useful input features
    feature_columns = [
        "Sales",
        "Quantity",
        "Discount",
        "Shipping Cost",
        "Ship Mode",
        "Segment",
        "City",
        "State",
        "Country",
        "Market",
        "Region",
        "Category",
        "Sub-Category",
        "Order Priority",
        "Order Year",
        "Order Month"
    ]

    # Add target variable
    selected_columns = feature_columns + ["Profitability"]
    df = df[selected_columns]

    print("\n--- SELECTED FEATURES ---")
    print(feature_columns)

    print("\nNumber of input features:", len(feature_columns))
    print("Target variable: Profitability")

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().sum())

    # Save selected features
    output_dir = (
        project_root / "data" / "processed" / "v1"
    )
    output_file = output_dir / "lab2_selected_features.csv"

    df.to_csv(output_file, index=False)

    print("\nSelected dataset shape:", df.shape)
    print("Saved file:", output_file)
    print("\nLab 2 Step 2 completed successfully!")


if __name__ == "__main__":
    select_lab2_features()
