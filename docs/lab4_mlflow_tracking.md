
# Experiment 4: Experiment Tracking and Reproducibility with MLflow

## Objective
To track machine learning experiments using MLflow by recording parameters, metrics, models, artifacts, dataset versions and experiment configurations.

## Dataset
Global Superstore Dataset

Target:
- Profit > 0: Profitable (1)
- Profit <= 0: Not Profitable (0)

## Tools and Technologies
- Python
- Pandas
- Scikit-learn
- MLflow
- SQLite

## Implementation

1. Installed and configured MLflow.
2. Created an MLflow experiment named Global_Superstore_Profitability.
3. Tracked Logistic Regression, Decision Tree and Random Forest models.
4. Recorded accuracy, precision, recall, F1-score and ROC-AUC.
5. Compared StandardScaler and MinMaxScaler.
6. Compared 80:20 and 70:30 data splits.
7. Tracked 12 model comparison runs.
8. Compared Logistic Regression hyperparameters.
9. Applied SelectKBest for feature selection.
10. Tracked dataset version and SHA-256 hash.
11. Logged trained models and evaluation artifacts.
12. Exported experiment comparison reports.

## Hyperparameter and Feature Selection Results

| Configuration | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|
| All features, C=1 | 91.79% | 94.60% | 95.36% |
| Selected 100, C=1 | 91.90% | 94.68% | 95.31% |
| All features, C=0.1 | 91.87% | 94.66% | 95.57% |
| All features, C=10 | 91.29% | 94.26% | 94.75% |

## Conclusion

MLflow was successfully used to track machine learning experiments, compare models and configurations, record evaluation metrics, save model artifacts and maintain dataset lineage. This improves experiment reproducibility and traceability.
