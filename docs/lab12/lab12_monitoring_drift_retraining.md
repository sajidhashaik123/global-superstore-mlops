# Experiment 12: Monitoring, Drift Detection, and Retraining Workflows

## 1. Aim

To implement a monitoring and drift detection workflow for a Machine Learning model, monitor API performance, identify changes in incoming data and predictions, retrain the model using updated historical data, and evaluate the retrained model.

## 2. Objectives

- Prepare reference, monitoring, retraining, and evaluation datasets.
- Monitor numerical and categorical feature distributions.
- Detect prediction distribution changes.
- Evaluate model performance over different periods.
- Implement API request monitoring and performance analysis.
- Retrain the model using updated historical data.
- Compare the original and retrained models.
- Validate the saved model artifacts independently.

## 3. Dataset Description

**Dataset:** Global Superstore

**Original dataset:**
- Records: 51,290
- Columns: 24
- Processed columns: 23
- Date range: 2011-01-01 to 2014-12-31

**Target variable:** `Profitability`

Target definition:

- `Profitability = 1` if Profit > 0.
- `Profitability = 0` otherwise.

The Profit column is used to derive the target and is not included as a model input feature.

The model uses 16 input features.

## 4. Chronological Data Preparation

The dataset was divided into chronological periods.

| Dataset | Period | Records |
|---|---|---:|
| Reference | 2011–2012 | 19,960 |
| Monitoring | 2013 | 13,799 |
| Retraining | 2011–2013 | 33,759 |
| Evaluation | 2014 | 17,531 |

The chronological datasets were generated using:

`src/lab12/prepare_monitoring_data.py`

### Target distribution

| Dataset | Class 0 | Class 1 |
|---|---:|---:|
| Reference | 5,162 | 14,798 |
| Monitoring | 3,515 | 10,284 |
| Retraining | 8,677 | 25,082 |
| Evaluation | 4,535 | 12,996 |

## 5. Drift Detection

The drift detection implementation was developed in:

`src/lab12/drift_detection.py`

The following methods were implemented:

### 5.1 Numerical Feature Drift

- Kolmogorov-Smirnov (KS) test.
- Population Stability Index (PSI).

The implementation flags numerical drift when the KS-test p-value is below 0.05 or PSI is at least 0.2.

### 5.2 Categorical Feature Drift

The Chi-square test was used to compare categorical feature distributions between reference and monitoring datasets.

A p-value below 0.05 indicates a statistically significant distribution difference.

### 5.3 Prediction Drift

The predicted probability distributions were compared using the KS test. Positive prediction rates were also compared.

### 5.4 Performance Monitoring

Accuracy, precision, recall, and F1-score were calculated to monitor model performance.

## 6. Monitoring Results

The monitoring workflow compared the 2011–2012 reference period with the 2013 monitoring period.

**Input drift results:**

- Numerical features with drift: 2 out of 6.
- Categorical features with drift: 6 out of 10.
- Prediction drift detected: False.

### Reference and Monitoring Performance

| Metric | Reference | Monitoring |
|---|---:|---:|
| Accuracy | 92.09% | 92.11% |
| Precision | 92.57% | 92.77% |
| Recall | 97.13% | 96.97% |
| F1-score | 94.80% | 94.82% |

The model showed nearly unchanged performance between the reference and monitoring datasets.

Although several input features showed distribution changes, the prediction drift test did not detect a statistically significant change.

**Important methodological note:** The original `lab7-v1` model was trained using a random split spanning all four years. Therefore, the 2013 monitoring comparison is not an independent temporal evaluation. These results are monitoring observations, not proof of future generalization.

## 7. API Monitoring

API monitoring was implemented in:

`src/lab12/monitoring_api.py`

The monitoring API used the original model API and added request-monitoring middleware.

The monitored API was executed on port 8001.

### API execution command

```powershell
python -m uvicorn lab12.monitoring_api:app --app-dir src --host 127.0.0.1 --port 8001
```

### Monitored endpoints

- `/health`
- `/predict`
- `/monitoring/metrics`

The middleware records:

- Timestamp
- HTTP method
- Endpoint
- Status code
- Request success or failure
- Request latency

Request logs were stored in:

`reports/lab12/logs/api_requests.jsonl`

## 8. API Performance Analysis

The script:

`src/lab12/analyze_api_logs.py`

was implemented to calculate API performance statistics.

### Observed Results

| Metric | Result |
|---|---:|
| Total requests | 3 |
| Successful requests | 3 |
| Failed requests | 0 |
| Failure rate | 0% |
| Average latency | 27.797 ms |
| Minimum latency | 2.988 ms |
| Maximum latency | 76.854 ms |
| P50 latency | 3.550 ms |
| P95 latency | 69.524 ms |
| Time span | 100.147 seconds |
| Throughput | 0.030 requests/second |

HTTP status distribution: 200 – 3 requests.

Endpoint distribution:

- `/health`: 1 request
- `/predict`: 1 request
- `/monitoring/metrics`: 1 request

The API performance report was saved in:

`reports/lab12/api_performance_report.json`

**Observation:** The monitoring system successfully recorded API requests and calculated latency and failure statistics. These measurements are based on only three requests and should not be interpreted as production performance benchmarks.

## 9. Model Retraining

The retraining implementation was developed in:

`src/lab12/retrain_model.py`

The model was retrained using the 2011–2013 dataset.

### Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Model version | lab12-v2 |
| Training records | 33,759 |
| Input features | 16 |
| Encoded features | 4,417 |
| Random state | 42 |
| Maximum iterations | 1,000 |
| Numerical preprocessing | MinMaxScaler |
| Categorical preprocessing | OneHotEncoder |

The retrained model was saved separately in:

`models/lab12/v2/`

Generated artifacts:

- `classifier.joblib`
- `preprocessor.joblib`
- `training_metadata.json`

The original `models/lab7/v1/` artifacts were preserved.

## 10. Model Evaluation

The original baseline and retrained model were evaluated on the same 2014 evaluation dataset containing 17,531 records.

The evaluation implementation is:

`src/lab12/evaluate_retrained_model.py`

### Model Comparison

| Metric | Baseline lab7-v1 | Retrained lab12-v2 |
|---|---:|---:|
| Accuracy | 92.25% | 91.80% |
| Precision | 92.90% | 92.66% |
| Recall | 96.96% | 96.59% |
| F1-score | 94.89% | 94.59% |
| ROC-AUC | 96.29% | 95.04% |

### Confusion Matrices

**Baseline lab7-v1**

```text
[[3572, 963],
 [395, 12601]]
```

**Retrained lab12-v2**

```text
[[3541, 994],
 [443, 12553]]
```

### Observation

The baseline model achieved slightly higher values across all five evaluation metrics.

The retrained model did not improve performance on the 2014 evaluation dataset.

However, the comparison has an important limitation: the original baseline model was trained using a random split spanning 2011–2014. It therefore had exposure to records from the 2014 evaluation period.

The retrained model was trained only on 2011–2013 and evaluated on 2014. Its evaluation is the proper chronological assessment of future-period generalization.

The comparison report was saved in:

`reports/lab12/retraining_comparison.json`

## 11. Model Validation

The validation implementation was developed in:

`src/lab12/validate_retrained_model.py`

The validation script independently loaded the saved model artifacts and generated predictions on the 2014 evaluation dataset.

### Validation Results

| Validation | Result |
|---|---|
| Preprocessor loading | Successful |
| Classifier loading | Successful |
| Evaluation records | 17,531 |
| Transformed feature shape | 17,531 × 4,417 |
| Predictions generated | 17,531 |
| Unique prediction classes | 0 and 1 |
| Minimum probability | 0.0000 |
| Maximum probability | 0.9991 |
| Validation checks | Passed |

All validation assertions passed successfully.

## 12. Generated Files and Reports

### Source Files

- `src/lab12/config.py`
- `src/lab12/prepare_monitoring_data.py`
- `src/lab12/drift_detection.py`
- `src/lab12/run_monitoring.py`
- `src/lab12/monitoring_api.py`
- `src/lab12/analyze_api_logs.py`
- `src/lab12/retrain_model.py`
- `src/lab12/evaluate_retrained_model.py`
- `src/lab12/validate_retrained_model.py`

### Data Files

- `data/processed/lab12/reference_2011_2012.csv`
- `data/processed/lab12/monitoring_2013.csv`
- `data/processed/lab12/retraining_2011_2013.csv`
- `data/processed/lab12/evaluation_2014.csv`

### Model Artifacts

- `models/lab12/v2/classifier.joblib`
- `models/lab12/v2/preprocessor.joblib`
- `models/lab12/v2/training_metadata.json`

### Reports

- `reports/lab12/drift_report.json`
- `reports/lab12/api_performance_report.json`
- `reports/lab12/retraining_comparison.json`
- `reports/lab12/logs/api_requests.jsonl`

## 13. Conclusion

Experiment 12 successfully implemented a monitoring, drift detection, API performance analysis, and model retraining workflow using the Global Superstore dataset.

Numerical and categorical feature distributions were monitored, prediction drift was evaluated, and API requests were recorded with latency and status information.

A new Logistic Regression model, `lab12-v2`, was trained using 2011–2013 records and evaluated using 2014 records. The saved model artifacts were independently loaded and validated successfully.

Although the retrained model did not outperform the original baseline on the reported metrics, the chronological evaluation provides a more appropriate assessment of future-period generalization.

Thus, the experiment demonstrates an end-to-end MLOps monitoring and retraining workflow while preserving the original model artifacts and previous experiments.
