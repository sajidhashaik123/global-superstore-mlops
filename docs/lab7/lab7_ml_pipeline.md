# Experiment 7: ML Pipeline Engineering

## 1. Aim

To design and build modular ML pipelines integrating preprocessing, training, validation, evaluation, and inference stages. Develop reusable pipeline components, track intermediate outputs and artifacts, validate workflow consistency and reproducibility, and assess pipeline reliability, scalability, modularity, and maintainability.

## 2. Dataset

**Dataset:** Global Superstore

**Dataset file:** `data/raw/global_superstore.csv`

- Total records: 51,290
- Total original columns: 24
- Input features: 16
- Target variable: `Profitability`

The target variable is derived from Profit:

- 1 – Profitable order (Profit > 0)
- 0 – Non-profitable order (Profit <= 0)

The original Profit column is excluded from model input features to prevent target leakage.

## 3. Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Joblib
- VS Code
- PowerShell

## 4. Pipeline Architecture

The ML pipeline consists of the following stages:

1. Data Loading
2. Data Preprocessing
3. Data Validation
4. Model Training
5. Model Evaluation
6. Model Inference

The stages are implemented as separate reusable Python modules and connected through a main pipeline execution file.

## 5. Implementation Details

### 5.1 Data Loading

- Loaded the existing Global Superstore training and testing datasets.
- Verified the required feature columns and target column.
- Checked for missing values and valid target values.
- Training dataset: 41,032 records.
- Testing dataset: 10,258 records.

### 5.2 Data Preprocessing

Numerical features were processed using MinMaxScaler.

Categorical features were encoded using OneHotEncoder with `handle_unknown="ignore"`.

The preprocessing transformation was fitted only on the training data and applied to the testing data.

- Original input features: 16
- Transformed features: 4,808
- Sparse matrices were used to store transformed data efficiently.

### 5.3 Data Validation

The validation stage checked:

- Training and testing feature dimensions.
- Feature-target row counts.
- Consistency of transformed feature dimensions.
- Finite feature values.
- Valid target values.

All validation checks passed.

### 5.4 Model Training

Logistic Regression was used as the classification algorithm.

Configuration:

- Algorithm: Logistic Regression
- Maximum iterations: 1000
- Random state: 42
- Training records: 41,032

The trained model and preprocessing object were saved separately using Joblib.

### 5.5 Model Evaluation

The model was evaluated using the unseen test dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 91.79% |
| Precision | 92.45% |
| Recall | 96.85% |
| F1-score | 94.60% |
| ROC-AUC | 95.36% |

Confusion Matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2040 | 602 |
| Actual 1 | 240 | 7376 |

Evaluation metrics, confusion matrix, and classification report were saved in the Lab 7 reports directory.

### 5.6 Model Inference

- Loaded the saved preprocessing object and trained classifier.
- Accepted order information without requiring the actual target value.
- Applied the same preprocessing transformation used during training.
- Generated profitability predictions and probabilities.
- Tested inference using five sample orders.

The sample inference results were saved as `sample_inference_predictions.csv`.

## 6. Main Pipeline Integration

All stages were connected using `lab7_pipeline.py`.

The complete pipeline executes:

Preprocessing → Validation → Training → Evaluation → Inference

The complete pipeline executed successfully, and the evaluation metrics matched the individual evaluation stage.

## 7. Intermediate Outputs and Artifacts

The following artifacts were generated:

- `X_train_transformed.npz`
- `X_test_transformed.npz`
- `y_train.csv`
- `y_test.csv`
- `transformed_feature_names.json`
- `preprocessor.joblib`
- `classifier.joblib`
- `training_metadata.json`
- `evaluation_metrics.json`
- `confusion_matrix.csv`
- `classification_report.csv`
- `sample_inference_predictions.csv`
- `reproducibility_results.json`

## 8. Reproducibility Testing

Two independent preprocessing and training runs were performed using the same training data and configuration.

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Accuracy | 0.9179 | 0.9179 |
| F1-score | 0.9460 | 0.9460 |

The predictions from both runs matched exactly.

**Result: Reproducibility test passed.**

## 9. Pipeline Assessment

### Reliability
- Input data and target values are validated.
- Transformed features are checked for finite values.
- Model artifacts are saved for reuse.
- Separate evaluation and inference stages are implemented.

### Scalability
- Sparse matrices reduce storage requirements for high-dimensional encoded data.
- Modular components support independent development and modification.
- The pipeline was tested on 51,290 records; larger-scale performance has not yet been benchmarked.

### Modularity
- Data loading, preprocessing, validation, training, evaluation, and inference are separated into reusable modules.
- Each stage can be executed independently or through the main pipeline.

### Maintainability
- Configuration values are centralized.
- Reusable functions reduce code duplication.
- Model metadata and intermediate artifacts support traceability.

## 10. Result

A modular ML pipeline for Global Superstore profitability prediction was successfully designed and implemented. The pipeline integrates preprocessing, validation, training, evaluation, and inference. Reproducibility was verified through two independent runs with matching predictions and evaluation metrics.