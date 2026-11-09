# Global Superstore MLOps Project

## 1. Project Overview

The Global Superstore MLOps Project demonstrates the implementation of Machine Learning Operations (MLOps) concepts using the Global Superstore dataset.

The project covers the complete machine learning lifecycle, starting from dataset preparation and preprocessing to model development, experiment tracking, model deployment, monitoring, drift detection and retraining.

The main objective is to develop a reproducible, automated and traceable MLOps workflow by integrating different tools and technologies.

**Technologies Used:**
- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- Git and GitHub
- FastAPI
- Docker
- Kubernetes
- GitHub Actions

## 2. Dataset Information

**Dataset:** Global Superstore

| Attribute | Details |
|---|---|
| Original records | 51,290 |
| Original columns | 24 |
| Processed records | 51,290 |
| Processed columns | 23 |
| Date range | 2011–2014 |
| Dataset version | v1 |

### Data Preprocessing

The following preprocessing operations were performed:

- Removed the Postal Code column because it contained 41,296 missing values.
- Converted Order Date and Ship Date into datetime format.
- Removed unnecessary leading and trailing spaces from text columns.
- Checked missing values and duplicate records.
- Preserved valid negative Profit values.
- Generated dataset version information using SHA-256 checksums.

**Processed dataset:** `data/processed/v1/global_superstore_clean.csv`

### Prediction Objective

The project implements a binary classification task to predict transaction profitability.

- Profit greater than zero: Profitability = 1
- Profit less than or equal to zero: Profitability = 0

The model uses numerical and categorical features from the Global Superstore dataset.

## 3. Project Structure

The project is organized into separate folders for datasets, source code, models, reports, documentation and deployment configurations.

- `data/` – Raw and processed datasets, dataset versions and splits.
- `src/` – Source code developed throughout the experiments.
- `models/` – Trained model artifacts and version-specific preprocessors.
- `reports/` – Experiment results, evaluation metrics and monitoring reports.
- `docs/` – Experiment documentation.
- `k8s/` – Kubernetes deployment, service and storage configurations.
- `.github/workflows/` – GitHub Actions automation workflows.
- `Dockerfile` – Existing application Docker configuration.
- `Dockerfile.lab13` – Experiment 13 Docker configuration.
- `requirements.txt` – Python dependencies.
- `.gitignore` – Files and folders excluded from Git tracking.

## 4. Environment Setup

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Experiment Implementation

All 13 experiments have been completed using the Global Superstore dataset.

| Experiment | Description | Status |
|---|---|---|
| 1 | ML Project Initialization and Dataset Preparation | Completed |
| 2 | Baseline Model Development and Evaluation | Completed |
| 3 | Git-Based Version Control and Collaborative ML Workflows | Completed |
| 4 | Experiment Tracking and Reproducibility with MLflow | Completed |
| 5 | Data Validation and Reproducible ML Pipelines | Completed |
| 6 | Model Registry and Version Management | Completed |
| 7 | ML Pipeline Engineering | Completed |
| 8 | Pipeline Automation and Continuous Integration | Completed |
| 9 | Pipeline Failure Handling and Recovery | Completed |
| 10 | Model Serving using FastAPI | Completed |
| 11 | Containerization and Deployment using Docker and Kubernetes | Completed |
| 12 | Monitoring, Drift Detection and Model Retraining | Completed |
| 13 | End-to-End MLOps System Mini Project | Completed |

## 6. Model Development and Lifecycle Management

The project maintains multiple model versions to demonstrate model development, evaluation and lifecycle management.

| Model Version | Description | Status |
|---|---|---|
| lab7-v1 | Baseline model | Production |
| lab12-v2 | Retrained model | Candidate - Not Promoted |
| lab13-v3 | Experiment 13 retrained model | Candidate - Not Promoted |

The models use Logistic Regression with MinMaxScaler and OneHotEncoder preprocessing.

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The baseline model remains the production model because it achieved stronger results in the recorded evaluation comparison.

Note: The baseline and retrained models used different training periods, so their evaluation results are not a fully like-for-like comparison.

## 7. Experiment Tracking Using MLflow

MLflow is used to maintain experiment tracking and reproducibility.

The tracking process records:

- Model versions
- Training parameters
- Evaluation metrics
- Model artifacts
- Model lifecycle information

Experiment 13 uses the MLflow experiment:

`Global_Superstore_Lab13_E2E`

## 8. End-to-End MLOps Pipeline

Experiment 13 integrates the components developed in the earlier experiments into an end-to-end MLOps workflow.

The integrated workflow includes:

1. Dataset validation.
2. Dataset checksum verification.
3. Model artifact verification.
4. Model prediction validation.
5. Model lifecycle verification.
6. Model evaluation report verification.
7. MLflow tracking verification.
8. Deployment configuration verification.
9. End-to-end execution report generation.

### Integration Validation

Run:

```bash
python -m src.lab13.validation
```

### End-to-End Orchestration

Run:

```bash
python -m src.lab13.orchestrator
```

The generated execution report is stored in:

`reports/lab13/e2e_execution_report.json`

## 9. Model Serving Using FastAPI

FastAPI is used to expose the trained model through REST API endpoints.

The Lab13 API provides:

| Endpoint | Purpose |
|---|---|
| `/` | API information |
| `/health` | Health check |
| `/metadata` | Model metadata |
| `/predict` | Single prediction |
| `/predict/batch` | Batch predictions |
| `/monitoring/metrics` | Monitoring metrics |

The API also records request information and prediction activity.

## 10. Docker Deployment

Docker is used to package the Experiment 13 application and its dependencies into a container.

The separate Docker configuration is:

`Dockerfile.lab13`

### Build the Docker image

```bash
docker build -f Dockerfile.lab13 -t global-superstore-mlops-e2e:v1 .
```

### Run the container

```bash
docker run -d --name global-superstore-lab13 -p 8001:8000 global-superstore-mlops-e2e:v1
```

The API can be accessed at:

`http://localhost:8001`

API documentation:

`http://localhost:8001/docs`

## 11. Kubernetes Deployment

Kubernetes is used to deploy and manage the Experiment 13 application.

The Lab13 Kubernetes configurations are maintained separately under `k8s/lab13/`.

The configurations include:

- Deployment configuration
- NodePort service
- Persistent Volume Claim
- Container health probes
- Persistent API log storage

The Lab13 service uses NodePort 30081.

The application can also be accessed using Kubernetes port forwarding:

```bash
kubectl port-forward deployment/global-superstore-lab13 8002:8000
```

Access the API through:

`http://localhost:8002`

## 12. Monitoring and Retraining

Experiment 12 and Experiment 13 demonstrate monitoring and retraining workflows.

The monitoring and retraining process includes:

- Monitoring model performance.
- Detecting data drift.
- Preparing retraining datasets.
- Training new model versions.
- Evaluating retrained models.
- Comparing model performance.
- Managing model promotion decisions.

The existing production model is retained when a candidate model does not demonstrate better evaluation performance.

## 13. GitHub Actions Automation

The Experiment 13 workflow is maintained at:

`.github/workflows/lab13_e2e.yml`

The workflow supports:

- Push events to the master branch.
- Pull requests targeting the master branch.
- Manual execution using workflow_dispatch.

The workflow automates:

- Dependency installation.
- Model retraining.
- Model evaluation.
- Lifecycle registry updates.
- MLflow experiment tracking.
- Integration validation.
- End-to-end orchestration.
- Artifact uploading.

Docker building and Kubernetes deployment remain separate deployment operations.

## 14. Reproducibility

The project follows reproducibility practices through:

- Dataset versioning.
- SHA-256 checksum verification.
- Version-specific model artifacts.
- Fixed random states where applicable.
- Dependency version pinning.
- MLflow experiment tracking.
- Dataset and model validation.
- Git-based source code version control.
- Documented execution workflows.

## 15. Final Outcome

The Global Superstore MLOps Project demonstrates the implementation of an integrated machine learning lifecycle using practical MLOps tools and technologies.

All 13 experiments have been completed.

The final Experiment 13 implementation integrates dataset validation, model versioning, experiment tracking, lifecycle management, pipeline orchestration, API serving, Docker containerization, Kubernetes deployment, monitoring and GitHub Actions automation.

The project provides a reproducible and traceable workflow for managing machine learning models from development to deployment and monitoring.

---

**Project:** Global Superstore MLOps Project  
**Dataset:** Global Superstore  
**Domain:** Machine Learning Operations (MLOps)  
**Experiments Completed:** 13  
**Primary Objective:** Develop an end-to-end, reproducible and traceable MLOps system.