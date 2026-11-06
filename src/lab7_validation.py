
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from lab7_config import PROCESSED_DIR


def validate_preprocessed_data():

    train_path = PROCESSED_DIR / "X_train_transformed.npz"
    test_path = PROCESSED_DIR / "X_test_transformed.npz"

    y_train_path = PROCESSED_DIR / "y_train.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"

    required_files = [
        train_path,
        test_path,
        y_train_path,
        y_test_path
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required pipeline output missing: {file_path}"
            )

    # Load transformed features and targets
    X_train = load_npz(train_path)
    X_test = load_npz(test_path)

    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    # Check row consistency
    if X_train.shape[0] != len(y_train):
        raise ValueError("Training feature and target row counts differ.")

    if X_test.shape[0] != len(y_test):
        raise ValueError("Testing feature and target row counts differ.")

    # Check feature consistency
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and testing feature dimensions differ.")

    # Check numerical validity
    if not np.isfinite(X_train.data).all():
        raise ValueError("Invalid values found in training features.")

    if not np.isfinite(X_test.data).all():
        raise ValueError("Invalid values found in testing features.")

    # Check target values
    if not y_train.isin([0, 1]).all():
        raise ValueError("Invalid training target values.")

    if not y_test.isin([0, 1]).all():
        raise ValueError("Invalid testing target values.")

    print("LAB 7 PIPELINE VALIDATION")
    print("------------------------------")
    print("Training feature shape:", X_train.shape)
    print("Testing feature shape:", X_test.shape)
    print("Training target rows:", len(y_train))
    print("Testing target rows:", len(y_test))
    print("Feature dimensions match: YES")
    print("Row counts match: YES")
    print("All feature values are finite: YES")
    print("Target values are valid: YES")
    print("\nPIPELINE VALIDATION PASSED!")

    return True


if __name__ == "__main__":
    validate_preprocessed_data()
