# Global Superstore MLOps Project

## 1. Project Overview

This project implements MLOps concepts using the **Global Superstore dataset**.

The project focuses on preparing data, developing machine learning models, tracking experiments, validating datasets, managing model versions, building ML pipelines, automating workflows, serving models through APIs, and deploying applications using Docker and Kubernetes.

The same Global Superstore dataset is used throughout the project.

## 2. Dataset Information

- **Dataset:** Global Superstore
- **Original file:** `global_superstore.csv`
- **Original records:** 51,290
- **Original columns:** 24
- **Processed columns:** 23
- **Date range:** January 2011 – December 2014
- **Dataset version:** v1

### Dataset Preprocessing

The following preprocessing operations were performed:

- Removed the Postal Code column because it contained excessive missing values.
- Converted Order Date and Ship Date into datetime format.
- Removed leading and trailing spaces from text columns.
- Checked duplicate records and missing values.
- Preserved valid negative Profit values.

## 3. Project Structure

```text
global_superstore_mlop_project/
│
├── data/
│   ├── raw/
│   │   └── global_superstore.csv
│   │
│   ├── processed/
│   │   └── v1/
│   │       ├── global_superstore_clean.csv
│   │       └── manifest.json
│   │
│   └── splits/
│       └── v1/
│           ├── train.csv
│           └── test.csv
│
├── docs/
│   └── lab1_preprocessing.md
│
├── reports/
│   ├── lab1_dataset_profile.json
│   └── lab1_dataset_profile.txt
│
├── src/
│   ├── data_loader.py
│   ├── data_analysis.py
│   ├── data_validation.py
│   ├── test_validation.py
│   ├── preprocess_data.py
│   ├── split_dataset.py
│   └── version_dataset.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 4. Environment Setup

### Step 1: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 2: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Experiment 1 – ML Project Initialization and Dataset Preparation

**Status: Completed**

Activities performed:

- Loaded and analyzed the Global Superstore dataset.
- Examined data types, missing values, duplicate records, and categorical distributions.
- Validated date columns and numerical values.
- Preprocessed the dataset.
- Created reproducible training and testing datasets.
- Generated dataset version information using SHA-256 checksums.
- Documented the preprocessing and dataset details.

### Dataset Split

| Dataset | Records |
|---|---:|
| Training | 41,032 |
| Testing | 10,258 |
| Total | 51,290 |

- **Training percentage:** 80%
- **Testing percentage:** 20%
- **Random state:** 42

## 6. Planned MLOps Experiments

The project covers the following 11 experiments:

1. ML Project Initialization and Dataset Preparation
2. Baseline Model Development and Evaluation
3. Git-Based Version Control and Collaborative ML Workflows
4. Experiment Tracking and Reproducibility with MLflow
5. Data Validation and Reproducible ML Pipelines
6. Model Registry and Version Management
7. ML Pipeline Engineering
8. Pipeline Automation and Continuous Integration
9. Pipeline Failure Handling and Recovery
10. Model Serving using FastAPI
11. Containerization and Deployment using Docker and Kubernetes

All experiments will use the Global Superstore dataset, with each experiment adapted to an appropriate task supported by the dataset.

## 7. Reproducibility

To maintain reproducibility:

- The original raw dataset is preserved.
- Processed datasets are stored separately.
- Dataset splits use a fixed random state of 42.
- Dataset versions are maintained.
- SHA-256 checksums are generated.
- Dependencies are listed in `requirements.txt`.

## 8. Current Project Progress

| Experiment | Status |
|---|---|
| Experiment 1 – Dataset Preparation | Completed |
| Experiment 2 – Baseline Model Development | Pending |
| Experiment 3 – Git Workflows | Pending |
| Experiment 4 – MLflow Tracking | Pending |
| Experiment 5 – Data Validation Pipelines | Pending |
| Experiment 6 – Model Registry | Pending |
| Experiment 7 – ML Pipeline Engineering | Pending |
| Experiment 8 – Pipeline Automation | Pending |
| Experiment 9 – Failure Handling | Pending |
| Experiment 10 – FastAPI Model Serving | Pending |
| Experiment 11 – Docker and Kubernetes | Pending |

---

**Project:** Global Superstore MLOps Project  
**Dataset:** Global Superstore  
**Purpose:** Practical implementation of MLOps concepts using a consistent dataset.