# Lab 6: Model Registry and Version Management

## Aim

To implement model registration and version management using MLflow Model Registry by registering trained machine learning models, maintaining metadata, managing model lifecycle stages, comparing model versions, and demonstrating model promotion and rollback.

## Dataset

**Global Superstore Dataset**

- Dataset: Global Superstore
- Dataset version: v1
- Prediction target: Order Profitability
- Target definition: Profit > 0

## Tools and Technologies

- Python
- Pandas
- Scikit-learn
- MLflow
- MLflow Model Registry
- VS Code

## Implementation

### 1. Model Registration

Registered trained machine learning models in the MLflow Model Registry.

Registered model name: `GlobalSuperstore_Profitability_Model`

Two model versions were registered:

- Version 1: Logistic Regression
- Version 2: Random Forest

Both versions were successfully registered and their status was verified as READY.

### 2. Model Metadata Management

Added metadata to both registered model versions.

Metadata included:

- Algorithm
- Dataset name and version, where recorded
- Preprocessing method
- Data split information
- Accuracy
- F1-score
- ROC-AUC
- Prediction target
- Lifecycle status

Version 1 metadata recorded an 80:20 data split. The data split for Version 2 was not recorded in its source run.

### 3. Model Version Comparison

Compared the performance of the two registered model versions.

| Metric | Version 1 | Version 2 |
|---|---:|---:|
| Algorithm | Logistic Regression | Random Forest |
| Preprocessing | MinMaxScaler | MinMaxScaler |
| Data Split | 80:20 | Not recorded |
| Accuracy | 0.9190 | 0.915968 |
| F1-score | 0.9468 | 0.944630 |
| ROC-AUC | 0.9531 | 0.951362 |

**Observation:** Version 1 achieved better accuracy, F1-score, and ROC-AUC than Version 2. Therefore, Version 1 was selected as the production candidate based on the available evaluation metrics.

### 4. Model Lifecycle Management

Used MLflow aliases to manage the lifecycle of registered model versions.

The following aliases were assigned:

- Validation: Version 1
- Staging: Version 2
- Production: Version 1

Lifecycle status tags were also updated for both model versions.

### 5. Model Promotion and Rollback

Performed a model version switching demonstration using MLflow aliases.

The production alias was initially assigned to Version 1. It was temporarily switched to Version 2 and then restored to Version 1.

**Observed output:**

- Before rollback: Version 1
- Temporary production version: Version 2
- After rollback: Version 1

The rollback demonstration was successful.

### 6. Model Registry Verification

Verified the registered model versions using the MLflow tracking client.

The verification confirmed:

- Both model versions were registered.
- Model metadata was available.
- Version 1 had the validation alias.
- Version 2 had the staging alias.
- Version 1 was restored as the production version after rollback.

## Files Created

- `src/lab6_register_model.py`
- `src/lab6_add_model_metadata.py`
- `src/lab6_model_lifecycle.py`
- `src/lab6_verify_registry.py`
- `docs/lab6_model_registry.md`

## Conclusion

Model registration and version management were successfully implemented using MLflow Model Registry with the Global Superstore dataset. Two model versions, Logistic Regression and Random Forest, were registered with their corresponding metadata and evaluation metrics.

Model versions were compared, lifecycle aliases were assigned for validation, staging, and production, and a rollback demonstration was performed successfully. Version 1 was selected as the production candidate because it achieved better performance across the reported evaluation metrics.

Thus, the experiment demonstrated model registration, metadata management,