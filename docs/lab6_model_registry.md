
# Experiment 6: Model Registry and Version Management

## 1. Aim
To implement model registration, metadata management, version tracking, and lifecycle management using MLflow Model Registry.

## 2. Dataset
Global Superstore dataset.

## 3. Model Details
- Model Name: GlobalSuperstore_Profitability_Model
- Model Version: 1
- Algorithm: Logistic Regression
- Prediction Target: Order Profitability
- Dataset Version: v1
- Preprocessing: MinMaxScaler
- Data Split: 70:30

## 4. Implementation

### Task 1: Model Registration
Registered the existing trained profitability prediction model from the MLflow experiment.

File: `src/lab6_register_model.py`

### Task 2: Model Metadata
Added model tags containing algorithm, dataset version, prediction target, preprocessing method, data split, accuracy, F1-score, and ROC-AUC.

File: `src/lab6_add_model_metadata.py`

### Task 3: Model Lifecycle Management
Updated the model lifecycle status to validated and assigned the validation alias to Version 1.

File: `src/lab6_model_lifecycle.py`

### Task 4: Registry Verification
Retrieved the registered model version, checked stored metadata, and verified the validation alias.

File: `src/lab6_verify_registry.py`

## 5. Results
- Model registration completed successfully.
- Model metadata stored successfully.
- Model Version 1 created.
- Lifecycle status updated to validated.
- Validation alias assigned successfully.
- Registry verification passed.

## 6. Conclusion
Successfully implemented model registration, metadata tracking, version management, lifecycle management, and verification using MLflow Model Registry with the Global Superstore profitability prediction model.
