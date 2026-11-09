# Experiment 13: End-to-End MLOps System Mini Project

## 1. Aim

Develop and integrate an end-to-end MLOps system incorporating data preprocessing, versioning, experiment tracking, validation, pipeline automation, deployment, monitoring, retraining, and lifecycle management. Build reproducible and traceable workflows using Docker and guided Kubernetes exposure, and validate the complete system through end-to-end execution, monitoring, and documentation.

## 2. Dataset Overview

The project uses the Global Superstore dataset.

- Original records: 51,290
- Original columns: 24
- Processed records: 51,290
- Processed columns: 23
- Date range: 2011-01-01 to 2014-12-31
- Unique customers: 1,590
- Unique orders: 25,035
- Postal Code was removed because it contained 41,296 missing values.
- The processed dataset contains no remaining missing values.

Processed dataset:
`data/processed/v1/global_superstore_clean.csv`

Dataset manifest:
`data/processed/v1/manifest.json`

The manifest stores dataset metadata and checksum information to support data traceability and integrity verification.

## 3. Prediction Objective and Features

The project predicts whether a transaction is profitable.

Target:
- `Profitability = 1` when Profit is greater than 0.
- `Profitability = 0` when Profit is less than or equal to 0.

The model uses 16 input features.

Numeric features:
- Sales
- Quantity
- Discount
- Shipping Cost
- Order Year
- Order Month

Categorical features:
- Ship Mode
- Segment
- City
- State
- Country
- Market
- Region
- Category
- Sub-Category
- Order Priority

The target column Profit is not used as an input feature.

## 4. Existing and New Model Versions

Experiment 13 integrates the models developed during earlier experiments and creates a new model version.

| Model version | Description | Lifecycle status |
|---|---|---|
| lab7-v1 | Existing baseline model | Production |
| lab12-v2 | Experiment 12 retrained model | Candidate - Not Promoted |
| lab13-v3 | Experiment 13 retrained model | Candidate - Not Promoted |

The models use Logistic Regression with MinMaxScaler and OneHotEncoder. The categorical encoder uses `handle_unknown="ignore"`.

Each model is stored with its corresponding preprocessor and classifier. Model artifacts must always be loaded as matching pairs.

New Experiment 13 model artifacts:
- `models/lab13/v3/classifier.joblib`
- `models/lab13/v3/preprocessor.joblib`
- `models/lab13/v3/training_metadata.json`

## 5. Model Retraining

The Experiment 13 model was trained using the existing Lab12 training dataset:

`data/processed/lab12/retraining_2011_2013.csv`

Training records: 33,759

The retraining script is:

`src/lab13/retrain_model.py`

The script generates the classifier, preprocessor, and training metadata for model version lab13-v3.

## 6. Model Evaluation

All three model versions were evaluated using:

`data/processed/lab12/evaluation_2014.csv`

The evaluation report is:

`reports/lab13/model_evaluation_comparison.json`

| Metric | lab7-v1 | lab12-v2 | lab13-v3 |
|---|---:|---:|---:|
| Accuracy | 0.9225 | 0.9180 | 0.9180 |
| Precision | 0.9290 | 0.9266 | 0.9266 |
| Recall | 0.9696 | 0.9659 | 0.9659 |
| F1-score | 0.9489 | 0.9459 | 0.9459 |
| ROC-AUC | 0.9629 | 0.9504 | 0.9504 |

The baseline model achieved the highest values among the three evaluated versions. Therefore, lab7-v1 remains the production model. The lab12-v2 and lab13-v3 models remain candidates and are not promoted.

**Evaluation note:** The baseline model was trained using a random split spanning all four years, whereas lab12-v2 and lab13-v3 were trained on 2011–2013 data and evaluated on 2014 data. Consequently, the evaluation comparison is not fully apples-to-apples because the training periods differ.

## 7. Model Lifecycle Management

The lifecycle registry is maintained in:

`reports/lab13/model_lifecycle.json`

It records the model versions, artifact locations, lifecycle statuses, and current production model.

Current lifecycle:
- lab7-v1: production
- lab12-v2: candidate_not_promoted
- lab13-v3: candidate_not_promoted

The lifecycle implementation is:

`src/lab13/lifecycle.py`

The production model is not automatically replaced when a new candidate is trained. Promotion requires an evaluation-based decision.

## 8. MLflow Experiment Tracking

Experiment 13 uses MLflow to track all three model versions.

Experiment name:

`Global_Superstore_Lab13_E2E`

Tracking implementation:

`src/lab13/mlflow_tracking.py`

The tracking workflow records:
- Model version and lifecycle status
- Training parameters
- Evaluation metrics
- Model metadata
- Preprocessor and classifier artifacts
- Evaluation report information

Experiment 13 uses an isolated MLflow SQLite database:

`reports/lab13/lab13_mlflow.db`

The database and runtime logs are excluded from Git version control. MLflow tracking can be reproduced by executing:

`python -m src.lab13.mlflow_tracking`

## 9. Integration Validation

The integration validation module is:

`src/lab13/validation.py`

It performs ten checks:

1. Dataset validation.
2. Dataset checksum validation.
3. Baseline model artifact validation.
4. Lab12 retrained model artifact validation.
5. Lab13 retrained model artifact validation.
6. Baseline model prediction validation.
7. Lab12 model prediction validation.
8. Lab13 model prediction validation.
9. Three-model lifecycle validation.
10. Evaluation report validation.

The local integration validation completed successfully, with all ten checks passing.

## 10. End-to-End Orchestration

The orchestration module is:

`src/lab13/orchestrator.py`

It verifies:
- Integration validation results.
- MLflow tracking for all three model versions.
- Model lifecycle and production selection.
- Required Docker and Kubernetes deployment configuration files.

Execution report:

`reports/lab13/e2e_execution_report.json`

The local end-to-end orchestration completed with an overall status of PASSED.

The orchestrator verifies the integration and configuration stages. It does not independently rebuild Docker images or redeploy Kubernetes resources.

## 11. API Deployment and Monitoring

The Experiment 13 API is implemented in:

`src/lab13/api.py`

It provides the following endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | API information |
| `/health` | Health check |
| `/metadata` | Model metadata |
| `/predict` | Single prediction |
| `/predict/batch` | Batch predictions |
| `/monitoring/metrics` | API monitoring metrics |

The API was tested locally. Health, metadata, single prediction, batch prediction, and monitoring requests completed successfully.

The API supports model selection through the `LAB13_MODEL_VERSION` environment variable. The deployed Experiment 13 instance uses lab7-v1 as its production model.

## 12. Docker Deployment

A separate Dockerfile was created for Experiment 13:

`Dockerfile.lab13`

Docker image:

`global-superstore-mlops-e2e:v1`

Container:

`global-superstore-lab13`

The container exposes the API on host port 8001.

The Docker image was built and the container was started successfully. API health, prediction, and monitoring endpoints were tested through the running container.

The existing Lab10 Dockerfile, image, and container were preserved.

## 13. Kubernetes Deployment

Experiment 13 Kubernetes configuration files are stored separately in:

`k8s/lab13/`

Files:
- `deployment.yaml`
- `service.yaml`
- `pvc.yaml`

Resources:
- Deployment: `global-superstore-lab13`
- Service: `global-superstore-lab13-service`
- Persistent Volume Claim: `global-superstore-lab13-logs-pvc`
- Service type: NodePort
- Configured NodePort: 30081
- Replicas: 1

The Kubernetes deployment reached the Running state and the PVC reached the Bound state.

The API was tested successfully through Kubernetes port forwarding. The monitoring endpoint returned metrics, and request logs were verified inside the mounted persistent volume.

**Environment note:** Direct access to NodePort 30081 from the host was unsuccessful in the Docker Desktop environment. The Kubernetes service endpoint was available, and port-forwarding was used successfully for API testing.

The existing Lab10 Kubernetes deployment and its configuration were preserved.

## 14. Monitoring and Logging

The API records request information in JSON Lines format.

Log location:

`reports/lab13/logs/api_requests.jsonl`

The Kubernetes deployment mounts a persistent volume at:

`/app/reports/lab13/logs`

This allows API request logs to persist through the configured volume.

The current API monitoring counters are maintained in application memory. They reset when the application process restarts. The JSONL request logs are persisted separately through the Kubernetes volume.

## 15. GitHub Actions Automation

The Experiment 13 workflow is:

`.github/workflows/lab13_e2e.yml`

Workflow name:

`Experiment 13 - End-to-End MLOps`

Triggers:
- Push to the master branch when relevant project files change.
- Pull requests targeting master when relevant project files change.
- Manual execution through workflow_dispatch.

The workflow:
1. Checks out the repository.
2. Sets up Python.
3. Installs dependencies.
4. Retrains the Lab13 model.
5. Evaluates all three models.
6. Initializes the lifecycle registry.
7. Updates model lifecycle information.
8. Tracks models with MLflow.
9. Runs integration validation.
10. Runs the end-to-end orchestrator.
11. Uploads Experiment 13 reports and model artifacts.

The workflow validates the MLOps pipeline and deployment configuration. Docker and Kubernetes deployment were tested separately in the local environment; the workflow does not automatically deploy to Kubernetes.

## 16. Reproducibility Commands

Run the following commands from the project root with the virtual environment activated.

Retrain the Experiment 13 model:

`python -m src.lab13.retrain_model`

Evaluate all model versions:

`python -m src.lab13.evaluate_models`

Initialize the lifecycle registry when setting up a fresh environment:

`python -c "from src.lab13.lifecycle import create_initial_registry; create_initial_registry()"`

Update the lifecycle registry:

`python -m src.lab13.lifecycle`

Run MLflow tracking:

`python -m src.lab13.mlflow_tracking`

Run integration validation:

`python -m src.lab13.validation`

Run the end-to-end orchestrator:

`python -m src.lab13.orchestrator`

## 17. Final Outcome

Experiment 13 integrates the existing Global Superstore MLOps work with a new retrained model, model evaluation, lifecycle management, MLflow experiment tracking, integration validation, end-to-end orchestration, API monitoring, Docker deployment, Kubernetes deployment, and GitHub Actions automation.

The baseline model lab7-v1 remains the production model because it achieved the strongest evaluation results in the current comparison. The newer model versions remain candidates and are not promoted.

The local integration validation and end-to-end orchestration completed successfully. Docker and Kubernetes API deployment tests were also completed successfully using the documented environment-specific access method.

## 18. Limitations and Future Improvements

- The model evaluation comparison uses different training periods for the baseline and retrained models.
- API monitoring counters are in memory and reset after process restarts.
- Kubernetes NodePort access from the host was unsuccessful in the current Docker Desktop environment; port-forwarding was used for testing.
- GitHub Actions validates the MLOps workflow and deployment configuration but does not automatically build and deploy the Docker image or update the Kubernetes cluster.
- Future improvements can include persistent centralized metrics, automated deployment, and a more controlled evaluation strategy using consistent training and testing periods.
