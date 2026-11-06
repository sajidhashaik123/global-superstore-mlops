
import pandas as pd

from lab7_config import (
    TRAIN_PATH,
    TEST_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def load_datasets():

    # Check whether dataset files exist
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Testing file not found: {TEST_PATH}")

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    # Validate required columns
    for name, df in [("Training", train_df), ("Testing", test_df)]:
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{name} dataset is missing columns: {missing_columns}"
            )

        if df[required_columns].isnull().any().any():
            raise ValueError(
                f"Missing values found in {name} dataset."
            )

        if not df[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"Invalid target values found in {name} dataset."
            )

    # Separate features and target
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # Display dataset information
    print("LAB 7 DATA LOADER")
    print("------------------------------")
    print("Training features:", X_train.shape)
    print("Training target:", y_train.shape)
    print("Testing features:", X_test.shape)
    print("Testing target:", y_test.shape)
    print("Number of input features:", len(FEATURE_COLUMNS))
    print("Target column:", TARGET_COLUMN)
    print("Target values:", sorted(y_train.unique()))

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load_datasets()
