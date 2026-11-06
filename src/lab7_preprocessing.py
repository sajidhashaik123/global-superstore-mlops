
import json
import joblib
import pandas as pd

from scipy.sparse import save_npz, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from lab7_config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PROCESSED_DIR,
    MODEL_DIR
)

from lab7_data_loader import load_datasets


def build_preprocessor():

    numerical_transformer = MinMaxScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


def preprocess_datasets():

    # Load training and testing data
    X_train, y_train, X_test, y_test = load_datasets()

    # Create preprocessing component
    preprocessor = build_preprocessor()

    # Fit only on training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Apply the same fitted transformations to testing data
    X_test_transformed = preprocessor.transform(X_test)

    # Convert outputs to sparse CSR format
    X_train_transformed = csr_matrix(X_train_transformed)
    X_test_transformed = csr_matrix(X_test_transformed)

    # Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save preprocessed features
    save_npz(
        PROCESSED_DIR / "X_train_transformed.npz",
        X_train_transformed
    )

    save_npz(
        PROCESSED_DIR / "X_test_transformed.npz",
        X_test_transformed
    )

    # Save target values
    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    # Save fitted preprocessor
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.joblib"
    )

    # Save transformed feature names
    feature_names = preprocessor.get_feature_names_out().tolist()

    with open(
        PROCESSED_DIR / "transformed_feature_names.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(feature_names, file, indent=4)

    # Display results
    print("\nLAB 7 PREPROCESSING RESULTS")
    print("------------------------------")
    print("Original training shape:", X_train.shape)
    print("Original testing shape:", X_test.shape)
    print("Transformed training shape:", X_train_transformed.shape)
    print("Transformed testing shape:", X_test_transformed.shape)
    print("Number of transformed features:", len(feature_names))
    print("Preprocessing strategy: MinMaxScaler + OneHotEncoder")
    print("Preprocessor saved successfully.")
    print("Intermediate outputs saved successfully.")


if __name__ == "__main__":
    preprocess_datasets()
