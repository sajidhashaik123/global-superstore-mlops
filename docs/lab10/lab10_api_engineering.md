# Experiment 10 – Model Deployment and API Engineering

## 1. Aim

Develop and deploy a trained machine learning model as an inference API using FastAPI with health-check, prediction, metadata, and batch inference endpoints. Implement input-output schemas, request and datatype validation, standardized responses, error handling, and evaluate API latency, throughput, response consistency, and inference reliability.

## 2. Objectives

- Deploy the trained machine learning model using FastAPI.
- Implement health-check and model metadata endpoints.
- Develop single-record and batch prediction endpoints.
- Define input and output schemas using Pydantic.
- Validate incoming data types and required fields.
- Handle invalid requests and inference errors.
- Evaluate API performance using latency, throughput, consistency, and reliability.

## 3. Tools and Technologies

- Python
- FastAPI
- Uvicorn
- Pydantic
- Scikit-learn
- Joblib
- Pandas
- Requests
- VS Code
- Swagger UI

## 4. Model and Artifact Details

The trained model from Experiment 7 was reused for deployment.

**Model details:**
- Model type: Logistic Regression
- Model version: lab7-v1
- Prediction target: Profitability
- Preprocessing: MinMaxScaler and OneHotEncoder
- Number of input features: 16
- Number of transformed features: 4,808

**Model artifacts:**
- `models/lab7/v1/classifier.joblib`
- `models/lab7/v1/preprocessor.joblib`
- `models/lab7/v1/training_metadata.json`

## 5. Implementation

The API implementation was created in:

`src/lab10/api.py`

The trained classifier and preprocessor were loaded using Joblib. FastAPI was used to expose the model through HTTP endpoints.

### 5.1 Health-check endpoint

- Method: GET
- Route: `/health`
- Purpose: Checks whether the API is running and whether the model is loaded.
- Result: HTTP 200 – Healthy.

### 5.2 Metadata endpoint

- Method: GET
- Route: `/metadata`
- Purpose: Provides model information, version, target, preprocessing details, and training metadata.
- Result: HTTP 200 – Metadata returned successfully.

### 5.3 Single prediction endpoint

- Method: POST
- Route: `/predict`
- Purpose: Accepts one order record and returns its profitability prediction and probability.
- Result: HTTP 200 – Prediction returned successfully.

### 5.4 Batch prediction endpoint

- Method: POST
- Route: `/predict/batch`
- Purpose: Accepts multiple order records and returns predictions for the submitted records.
- Batch input is restricted to 1–100 records.
- Result: HTTP 200 – Batch predictions returned successfully.

### 5.5 Root endpoint

- Method: GET
- Route: `/`
- Purpose: Provides basic API information and links to the documentation and health-check endpoint.
- Result: HTTP 200.

## 6. Input Validation and Error Handling

Pydantic schemas were used to validate incoming prediction requests.

The API checks:
- Required input fields.
- Numeric and categorical data types.
- Valid request structure.
- Batch request size.

An invalid input test was performed by passing a text value for the numeric `Sales` field.

**Observed result:** HTTP 422 – Unprocessable Entity, with a validation error indicating that a valid number was expected.

This confirms that invalid request data is rejected before model inference.

## 7. API Testing

The API was started using:

```bash
uvicorn lab10.api:app --app-dir src --reload
```

Swagger UI was accessed at:

`http://127.0.0.1:8000/docs`

The following tests were performed:

| Test | Endpoint | Observed result |
|---|---|---|
| API root | GET `/` | HTTP 200 |
| Health check | GET `/health` | HTTP 200 |
| Model metadata | GET `/metadata` | HTTP 200 |
| Single prediction | POST `/predict` | HTTP 200 |
| Batch prediction | POST `/predict/batch` | HTTP 200 |
| Invalid numeric input | POST `/predict` | HTTP 422 |

The single prediction test returned a `Profitable` prediction with a probability of approximately 0.9687 for the sample order.

## 8. Performance Evaluation

Performance testing was implemented in:

`src/lab10/performance_test.py`

A total of 100 sequential prediction requests were sent to the running API. The script measured response latency and calculated throughput, reliability, and response consistency.

### Performance results

| Metric | Observed value |
|---|---:|
| Total requests | 100 |
| Successful requests | 100 |
| Failed requests | 0 |
| Average latency | 36.399 ms |
| Median latency | 37.170 ms |
| Minimum latency | 20.701 ms |
| Maximum latency | 56.317 ms |
| Throughput | 27.473 requests/second |
| Reliability | 100% |
| Response consistency | True |

### Performance interpretation

- **Latency:** The average response time was approximately 36.399 milliseconds.
- **Throughput:** The API processed approximately 27.473 requests per second during the test.
- **Reliability:** All 100 requests completed successfully.
- **Response consistency:** Repeated requests using the same input produced consistent predictions.

These results demonstrate successful inference and stable responses during the tested sequential workload.

## 9. Generated Reports

The performance testing script generated the following reports:

- `reports/lab10/performance_results.csv`
- `reports/lab10/performance_results.json`

These files contain the measured performance results.

## 10. Project Structure

```text
global_superstore_mlop_project/
│
├── src/
│   └── lab10/
│       ├── api.py
│       └── performance_test.py
│
├── models/
│   └── lab7/
│       └── v1/
│           ├── classifier.joblib
│           ├── preprocessor.joblib
│           └── training_metadata.json
│
├── reports/
│   └── lab10/
│       ├── performance_results.csv
│       └── performance_results.json
│
└── docs/
    └── lab10/
        └── lab10_api_engineering.md
```

## 11. Result

The trained Logistic Regression model from Experiment 7 was successfully deployed as a FastAPI inference service. Health-check, metadata, single prediction, and batch prediction endpoints were implemented and tested. Input validation and invalid-request handling were verified. Performance testing of 100 requests achieved 100% reliability, an average latency of 36.399 ms, and a throughput of 27.473 requests per second.

Therefore, Experiment 10 was successfully completed.