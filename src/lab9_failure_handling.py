
import logging
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lab7_config import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    REPORT_DIR,
    TEST_PATH
)


# --------------------------------------------------
# 1. CONFIGURATION AND LOGGING
# --------------------------------------------------

LAB9_REPORT_DIR = REPORT_DIR.parent / "lab9"
LAB9_REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LAB9_REPORT_DIR / "failure_handling.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

REQUIRED_ARTIFACTS = [
    "preprocessor.joblib",
    "classifier.joblib"
]


# --------------------------------------------------
# 2. INPUT VALIDATION
# --------------------------------------------------

def validate_input_data(data):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input dataset is empty.")

    missing_columns = set(FEATURE_COLUMNS) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Schema mismatch. Missing columns: {sorted(missing_columns)}"
        )

    missing_values = data[FEATURE_COLUMNS].isnull().sum()

    columns_with_missing = missing_values[missing_values > 0]

    if not columns_with_missing.empty:
        raise ValueError(
            f"Missing data detected: {columns_with_missing.to_dict()}"
        )

    for column in NUMERICAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(
                f"Invalid datatype in numerical column: {column}"
            )

        if not np.isfinite(data[column]).all():
            raise ValueError(
                f"Non-finite values found in column: {column}"
            )

    return True


# --------------------------------------------------
# 3. DATA RECOVERY
# --------------------------------------------------

def recover_input_data(data):

    recovered = data.copy()

    # Recover missing numerical values
    for column in NUMERICAL_FEATURES:

        recovered[column] = pd.to_numeric(
            recovered[column],
            errors="coerce"
        )

        recovered[column] = recovered[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

        median_value = recovered[column].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Cannot recover numerical column: {column}"
            )

        recovered[column] = recovered[column].fillna(
            median_value
        )

    # Recover missing categorical values
    for column in CATEGORICAL_FEATURES:

        recovered[column] = recovered[column].astype("string")

        recovered[column] = recovered[column].replace(
            ["", "nan", "None", "<NA>"],
            pd.NA
        )

        mode_values = recovered[column].mode(dropna=True)

        if mode_values.empty:
            raise ValueError(
                f"Cannot recover categorical column: {column}"
            )

        recovered[column] = recovered[column].fillna(
            mode_values.iloc[0]
        )

        recovered[column] = recovered[column].astype(str)

    logging.info("Input data recovered using median and mode imputation.")

    return recovered


# --------------------------------------------------
# 4. SAFE MODEL ARTIFACT LOADING
# --------------------------------------------------

def load_model_artifacts(
    artifact_dir,
    fallback_dir=MODEL_DIR
):

    loaded_artifacts = {}

    for artifact_name in REQUIRED_ARTIFACTS:

        primary_path = Path(artifact_dir) / artifact_name

        if primary_path.exists():

            selected_path = primary_path

        else:

            logging.warning(
                "Missing model artifact: %s",
                primary_path
            )

            fallback_path = Path(fallback_dir) / artifact_name

            if not fallback_path.exists():

                raise FileNotFoundError(
                    f"Artifact unavailable in primary and fallback "
                    f"locations: {artifact_name}"
                )

            logging.info(
                "Recovering artifact from fallback: %s",
                fallback_path
            )

            selected_path = fallback_path

        loaded_artifacts[artifact_name] = joblib.load(
            selected_path
        )

    return loaded_artifacts


# --------------------------------------------------
# 5. PREPROCESSING WITH RECOVERY
# --------------------------------------------------

def safe_preprocess(data, preprocessor):

    try:

        validate_input_data(data)

        transformed = preprocessor.transform(
            data[FEATURE_COLUMNS]
        )

        logging.info("Preprocessing completed successfully.")

        return transformed

    except Exception as error:

        logging.error(
            "Preprocessing failure: %s",
            error
        )

        print("Preprocessing failed:", error)
        print("Attempting data recovery...")

        recovered_data = recover_input_data(data)

        validate_input_data(recovered_data)

        transformed = preprocessor.transform(
            recovered_data[FEATURE_COLUMNS]
        )

        logging.info("Preprocessing recovered successfully.")

        print("Recovery successful!")

        return transformed


# --------------------------------------------------
# 6. FAILURE SCENARIO TESTS
# --------------------------------------------------

def run_failure_tests():

    print("\n========================================")
    print(" EXPERIMENT 9: PIPELINE FAILURE HANDLING")
    print("========================================")

    results = []

    def record_result(name, function):

        try:

            function()

            results.append({
                "Scenario": name,
                "Status": "PASSED",
                "Details": "Failure detected and recovery verified."
            })

            print(f"\n[PASS] {name}")

        except Exception as error:

            results.append({
                "Scenario": name,
                "Status": "FAILED",
                "Details": str(error)
            })

            logging.exception(
                "Test failed: %s",
                name
            )

            print(f"\n[FAIL] {name}: {error}")

    # Load a real sample from the existing test dataset
    test_df = pd.read_csv(TEST_PATH)

    sample_data = test_df[FEATURE_COLUMNS].head(20).copy()

    # Load the existing fitted preprocessor
    artifacts = load_model_artifacts(MODEL_DIR)

    preprocessor = artifacts["preprocessor.joblib"]

    # SCENARIO 1: MISSING DATA
    def test_missing_data():

        faulty_data = sample_data.copy()

        faulty_data.loc[
            faulty_data.index[0], "Sales"
        ] = np.nan

        try:
            validate_input_data(faulty_data)
            raise AssertionError("Missing data was not detected.")

        except ValueError as error:

            if "Missing data" not in str(error):
                raise

        recovered = recover_input_data(faulty_data)

        validate_input_data(recovered)

    record_result(
        "Missing Data Detection and Recovery",
        test_missing_data
    )

    # SCENARIO 2: INVALID DATATYPES
    def test_invalid_datatypes():

        faulty_data = sample_data.copy()

        faulty_data["Sales"] = faulty_data[
            "Sales"
        ].astype(object)

        faulty_data.loc[
            faulty_data.index[0], "Sales"
        ] = "invalid_sales"

        try:
            validate_input_data(faulty_data)
            raise AssertionError("Invalid datatype was not detected.")

        except TypeError as error:

            if "Invalid datatype" not in str(error):
                raise

        recovered = recover_input_data(faulty_data)

        validate_input_data(recovered)

    record_result(
        "Invalid Datatype Detection and Recovery",
        test_invalid_datatypes
    )

    # SCENARIO 3: SCHEMA MISMATCH
    def test_schema_mismatch():

        faulty_data = sample_data.drop(
            columns=["Sales"]
        )

        try:
            validate_input_data(faulty_data)
            raise AssertionError("Schema mismatch was not detected.")

        except ValueError as error:

            if "Schema mismatch" not in str(error):
                raise

        # Recovery: restore the required column from original input
        recovered = faulty_data.copy()

        recovered["Sales"] = sample_data["Sales"]

        validate_input_data(recovered)

    record_result(
        "Schema Mismatch Detection and Recovery",
        test_schema_mismatch
    )

    # SCENARIO 4: PREPROCESSING ERROR
    def test_preprocessing_error():

        faulty_data = sample_data.copy()

        faulty_data.loc[
            faulty_data.index[0], "Quantity"
        ] = np.nan

        transformed = safe_preprocess(
            faulty_data,
            preprocessor
        )

        if transformed.shape[0] != len(sample_data):
            raise AssertionError(
                "Recovered preprocessing row count mismatch."
            )

    record_result(
        "Preprocessing Error Handling and Recovery",
        test_preprocessing_error
    )

    # SCENARIO 5: MISSING MODEL ARTIFACT
    def test_missing_model_artifact():

        # Use a temporary directory to simulate missing artifacts.
        # Original Lab 7 model files are never deleted.
        with tempfile.TemporaryDirectory() as temp_dir:

            missing_artifact_dir = Path(temp_dir)

            recovered_artifacts = load_model_artifacts(
                missing_artifact_dir,
                fallback_dir=MODEL_DIR
            )

            for artifact_name in REQUIRED_ARTIFACTS:

                if artifact_name not in recovered_artifacts:
                    raise AssertionError(
                        f"Recovery failed: {artifact_name}"
                    )

    record_result(
        "Missing Model Artifact Detection and Recovery",
        test_missing_model_artifact
    )

    # --------------------------------------------------
    # 7. SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = (
        LAB9_REPORT_DIR / "failure_handling_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n========================================")
    print(" EXPERIMENT 9 TEST SUMMARY")
    print("========================================")

    print(results_df.to_string(index=False))

    passed = sum(
        result["Status"] == "PASSED"
        for result in results
    )

    print("\nTotal scenarios:", len(results))
    print("Passed:", passed)
    print("Failed:", len(results) - passed)

    print("\nResults saved to:", results_path)
    print("Failure logs saved to:", LOG_FILE)

    if passed != 5:
        raise RuntimeError(
            "Some Experiment 9 failure-handling tests failed."
        )

    print("\nAll five failure-handling scenarios passed!")


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    run_failure_tests()
