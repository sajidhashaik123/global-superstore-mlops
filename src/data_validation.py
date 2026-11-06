
import pandas as pd


def validate_dataset(df):

    print("\n--- DATA VALIDATION ---")

    # 1. Duplicate row validation
    print("\n--- DUPLICATE ROWS ---")
    print("Duplicate rows:", df.duplicated().sum())

    # 2. Missing value validation
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    # 3. Blank string validation
    print("\n--- BLANK STRING VALUES ---")
    object_columns = df.select_dtypes(include="object").columns

    blank_values = (
        df[object_columns]
        .apply(lambda column: column.str.strip().eq("").sum())
    )

    print(blank_values)

    # 4. Date validation
    print("\n--- DATE VALIDATION ---")

    order_dates = pd.to_datetime(
        df["Order Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    ship_dates = pd.to_datetime(
        df["Ship Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    print("Invalid order dates:", order_dates.isna().sum())
    print("Invalid ship dates:", ship_dates.isna().sum())

    # 5. Shipping date consistency
    print("\n--- SHIP DATE CONSISTENCY ---")

    invalid_ship_dates = (
        ship_dates.notna()
        & order_dates.notna()
        & (ship_dates < order_dates)
    )

    print(
        "Ship dates before order dates:",
        invalid_ship_dates.sum()
    )

    # 6. Negative value validation
    print("\n--- NEGATIVE VALUE CHECK ---")

    for column in ["Sales", "Quantity", "Shipping Cost"]:
        print(
            column,
            ":",
            (df[column] < 0).sum()
        )

    print("\n--- DATA VALIDATION COMPLETED ---")
