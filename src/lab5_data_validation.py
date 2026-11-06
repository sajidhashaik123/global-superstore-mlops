
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "global_superstore.csv"

REQUIRED_COLUMNS = [
    "Row ID", "Order ID", "Order Date", "Ship Date",
    "Ship Mode", "Customer ID", "Customer Name", "Segment",
    "City", "State", "Country", "Postal Code", "Market",
    "Region", "Product ID", "Category", "Sub-Category",
    "Product Name", "Sales", "Quantity", "Discount",
    "Profit", "Shipping Cost", "Order Priority"
]

ALLOWED_MISSING_COLUMNS = ["Postal Code"]

VALID_CATEGORIES = {
    "Ship Mode": [
        "Standard Class", "Second Class", "First Class", "Same Day"
    ],
    "Segment": ["Consumer", "Corporate", "Home Office"],
    "Category": ["Furniture", "Office Supplies", "Technology"],
    "Order Priority": ["Critical", "High", "Medium", "Low"]
}


def validate_dataset(df):
    errors = []

    print("\n--- LAB 5: DATA VALIDATION ---")
    print("Dataset shape:", df.shape)

    # 1. Schema validation
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(REQUIRED_COLUMNS)

    print("\nMissing columns:", missing_columns)
    print("Unexpected columns:", extra_columns)

    if missing_columns:
        errors.append("Required columns are missing.")

    # 2. Missing value validation
    print("\n--- MISSING VALUES ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    for column in df.columns:
        if column not in ALLOWED_MISSING_COLUMNS:
            if df[column].isnull().any():
                errors.append(f"Missing values found in {column}.")

    # 3. Data type validation
    print("\n--- DATA TYPE VALIDATION ---")
    print(df.dtypes)

    numeric_columns = [
        "Row ID", "Quantity", "Sales", "Discount",
        "Profit", "Shipping Cost"
    ]

    for column in numeric_columns:
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
            errors.append(f"Invalid data type in {column}.")

    # 4. Category validation
    print("\n--- CATEGORY VALIDATION ---")

    for column, allowed_values in VALID_CATEGORIES.items():
        if column in df.columns:
            invalid_values = set(df[column].dropna().unique()) - set(allowed_values)
            print(f"{column} invalid values:", invalid_values)

            if invalid_values:
                errors.append(f"Invalid categories found in {column}.")

    # 5. Date validation
    print("\n--- DATE VALIDATION ---")

    for column in ["Order Date", "Ship Date"]:
        if column in df.columns:
            dates = pd.to_datetime(
                df[column], format="%d-%m-%Y", errors="coerce"
            )
            invalid_dates = dates.isna().sum()
            print(f"Invalid {column} values:", invalid_dates)

            if invalid_dates:
                errors.append(f"Invalid dates found in {column}.")

    # 6. Numeric range validation
    print("\n--- NUMERIC RANGE VALIDATION ---")

    checks = {
        "Sales": df["Sales"] < 0,
        "Quantity": df["Quantity"] <= 0,
        "Discount": (df["Discount"] < 0) | (df["Discount"] > 1),
        "Shipping Cost": df["Shipping Cost"] < 0
    }

    for column, invalid_rows in checks.items():
        print(f"Invalid {column} values:", invalid_rows.sum())

        if invalid_rows.any():
            errors.append(f"Invalid values found in {column}.")

    # 7. Duplicate validation
    duplicate_count = df.duplicated().sum()
    print("\nDuplicate rows:", duplicate_count)

    if duplicate_count:
        errors.append("Duplicate rows found.")

    # Final result
    print("\n--- VALIDATION RESULT ---")

    if errors:
        print("Validation FAILED")
        for error in errors:
            print("-", error)
    else:
        print("Validation PASSED")
        print("All validation checks completed successfully.")

    return errors


if __name__ == "__main__":
    dataset = pd.read_csv(DATA_PATH)
    validate_dataset(dataset)
