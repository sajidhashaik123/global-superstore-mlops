# Experiment 9: Pipeline Failure Handling and Recovery

## 1. AIM

Implement and validate failure-handling and recovery mechanisms for ML pipelines by simulating missing data, invalid datatypes, schema mismatches, preprocessing errors, and missing model artifacts. Identify root causes, implement validation checks, exception handling, fallback mechanisms, and recovery workflows, and evaluate pipeline robustness and fault tolerance.

## 2. OBJECTIVES

- Detect missing values in input datasets.
- Identify invalid datatypes in numerical features.
- Detect missing columns and schema mismatches.
- Handle preprocessing errors using exception handling.
- Recover missing or corrupted input values.
- Implement fallback mechanisms for missing model artifacts.
- Maintain execution logs and test results.
- Evaluate the robustness and fault tolerance of the ML pipeline.

## 3. DATASET DESCRIPTION

**Dataset:** Global Superstore

- Total records: 51,290
- Total columns: 24
- Dataset file: `data/raw/global_superstore.csv`
- Training and testing datasets: Existing 80:20 split from Experiment 2.
- Target variable: `Profitability`

The existing Experiment 7 preprocessing and trained model artifacts were reused for testing failure-handling mechanisms.

## 4. IMPLEMENTATION

A separate Python module named `lab9_failure_handling.py` was developed.

The implementation contains the following components:

1. Configuration and logging.
2. Input data validation.
3. Data recovery using median and mode imputation.
4. Safe model artifact loading with fallback support.
5. Preprocessing with exception handling.
6. Failure scenario simulation and testing.
7. Result generation and logging.

### 4.1 Input Validation

The input validation function checks:

- Whether the input is a Pandas DataFrame.
- Whether the dataset is empty.
- Whether all required feature columns are available.
- Whether missing values are present.
- Whether numerical columns contain valid numeric values.
- Whether numerical values are finite.

### 4.2 Data Recovery

The recovery function handles missing and invalid input values.

- Numerical columns are converted into numeric format.
- Invalid numerical values are replaced with missing values.
- Missing numerical values are filled using the median.
- Missing categorical values are filled using the mode.

### 4.3 Model Artifact Recovery

The pipeline checks for the availability of:

- `preprocessor.joblib`
- `classifier.joblib`

If an artifact is missing from the primary artifact directory, the system attempts to load it from the existing Lab 7 model directory.

A temporary directory is used to simulate missing model artifacts without deleting the original model files.

### 4.4 Preprocessing Error Handling

The preprocessing function first validates the input data.

If an error occurs:

1. The error is recorded in the log.
2. The recovery process is initiated.
3. Missing and invalid values are corrected.
4. The recovered input is validated again.
5. Preprocessing is retried.

### 4.5 Logging

Python's logging module is used to record:

- Detected failures.
- Error messages.
- Recovery operations.
- Successful preprocessing.
- Failed test executions.

## 5. FAILURE SCENARIOS AND TESTING

Five failure scenarios were simulated.

### Scenario 1: Missing Data

**Failure:** A missing value was introduced into the Sales column.

**Detection:** The validation function identified the missing value.

**Recovery:** The missing numerical value was replaced using median imputation.

**Result:** PASSED.

### Scenario 2: Invalid Datatypes

**Failure:** An invalid text value was inserted into the numerical Sales column.

**Detection:** The validation function identified the incorrect datatype.

**Recovery:** The invalid value was converted into a missing value and replaced using median imputation.

**Result:** PASSED.

### Scenario 3: Schema Mismatch

**Failure:** The Sales column was removed from the input dataset.

**Detection:** The validation function detected the missing required column.

**Recovery:** The missing column was restored from the original sample dataset.

**Result:** PASSED.

### Scenario 4: Preprocessing Error

**Failure:** A missing value was introduced into the Quantity column before preprocessing.

**Detection:** Input validation detected the missing value.

**Recovery:** The recovery function filled the missing value, validated the recovered input, and retried preprocessing.

**Result:** PASSED.

### Scenario 5: Missing Model Artifacts

**Failure:** A temporary directory without model artifacts was used to simulate missing model files.

**Detection:** The artifact-loading function identified the missing files.

**Recovery:** The original Lab 7 model directory was used as the fallback location.

**Result:** PASSED.

## 6. EXECUTION RESULTS

All five failure-handling scenarios were executed successfully.

| Failure Scenario | Status |
|---|---|
| Missing Data Detection and Recovery | PASSED |
| Invalid Datatype Detection and Recovery | PASSED |
| Schema Mismatch Detection and Recovery | PASSED |
| Preprocessing Error Handling and Recovery | PASSED |
| Missing Model Artifact Detection and Recovery | PASSED |

**Execution Summary:**

- Total scenarios tested: 5
- Passed: 5
- Failed: 0
- Overall result: SUCCESS

## 7. OUTPUT FILES

The experiment generated the following output files:

| File | Description |
|---|---|
| `reports/lab9/failure_handling_results.csv` | Stores the results of all five failure scenarios. |
| `reports/lab9/failure_handling.log` | Stores failure messages and recovery activity. |

**Implementation file:**

`src/lab9_failure_handling.py`

## 8. CONCLUSION

Pipeline failure handling and recovery mechanisms were successfully implemented and validated using the Global Superstore dataset.

Five failure scenarios involving missing data, invalid datatypes, schema mismatches, preprocessing errors, and missing model artifacts were simulated and tested.

Validation checks, exception handling, median and mode imputation, fallback artifact loading, and recovery workflows were implemented.

All five test scenarios passed successfully, demonstrating that the implemented pipeline can detect the tested failures and recover from them using the defined recovery mechanisms.

Thus, the experiment successfully demonstrated pipeline robustness and fault tolerance.
