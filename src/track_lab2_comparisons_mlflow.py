
from pathlib import Path
import hashlib

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from lab2_preprocessing import build_preprocessor


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def track_comparisons():

    project_root = Path(__file__).resolve().parent.parent

    database_path = project_root / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{database_path.as_posix()}"
    )

    mlflow.set_experiment("Global_Superstore_Profitability")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    splits = ["80_20", "70_30"]
    strategies = ["standard", "minmax"]

    all_results = []

    for split in splits:

        train_file = (
            project_root / "data" / "splits" / "lab2"
            / split / "train.csv"
        )

        test_file = (
            project_root / "data" / "splits" / "lab2"
            / split / "test.csv"
        )

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        X_train = train_df.drop(columns=["Profitability"])
        y_train = train_df["Profitability"]

        X_test = test_df.drop(columns=["Profitability"])
        y_test = test_df["Profitability"]

        for strategy in strategies:

            for model_name, base_model in models.items():

                print("\nTracking:", model_name)
                print("Preprocessing:", strategy)
                print("Data split:", split)

                pipeline = Pipeline([
                    ("preprocessing", build_preprocessor(strategy)),
                    ("model", clone(base_model))
                ])

                with mlflow.start_run(
                    run_name=f"{model_name}_{strategy}_{split}"
                ):

                    pipeline.fit(X_train, y_train)

                    predictions = pipeline.predict(X_test)
                    probabilities = pipeline.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(
                        y_test, predictions, zero_division=0
                    )
                    recall = recall_score(
                        y_test, predictions, zero_division=0
                    )
                    f1 = f1_score(
                        y_test, predictions, zero_division=0
                    )
                    roc_auc = roc_auc_score(y_test, probabilities)

                    # Log parameters
                    mlflow.log_params({
                        "algorithm": model_name,
                        "preprocessing": strategy,
                        "data_split": split,
                        "random_state": 42,
                        "target": "Profit > 0",
                        "dataset_version": "v1"
                    })

                    # Log metrics
                    mlflow.log_metrics({
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "roc_auc": roc_auc
                    })

                    # Log dataset lineage
                    mlflow.set_tags({
                        "dataset_name": "Global Superstore",
                        "dataset_version": "v1",
                        "train_data_sha256": calculate_hash(train_file),
                        "test_data_sha256": calculate_hash(test_file)
                    })

                    # Log evaluation artifacts
                    mlflow.log_dict(
                        {
                            "confusion_matrix": confusion_matrix(
                                y_test, predictions
                            ).tolist()
                        },
                        "evaluation/confusion_matrix.json"
                    )

                    mlflow.log_dict(
                        classification_report(
                            y_test,
                            predictions,
                            zero_division=0,
                            output_dict=True
                        ),
                        "evaluation/classification_report.json"
                    )

                    # Log trained model
                    model_options = {}

                    if model_name in ["Decision Tree", "Random Forest"]:
                        model_options["skops_trusted_types"] = [
                            "sklearn.tree._tree.Tree"
                        ]

                    mlflow.sklearn.log_model(
                        pipeline,
                        name="profitability_model",
                        **model_options
                    )

                    all_results.append({
                        "Split": split,
                        "Preprocessing": strategy,
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "ROC-AUC": roc_auc
                    })

                    print("Accuracy:", round(accuracy, 4))
                    print("F1 Score:", round(f1, 4))
                    print("ROC-AUC:", round(roc_auc, 4))
                    print("MLflow Run ID:", mlflow.active_run().info.run_id)

    # Save comparison report
    results_df = pd.DataFrame(all_results)

    output_file = (
        project_root / "reports"
        / "lab4_mlflow_12_run_comparison.csv"
    )

    results_df.to_csv(output_file, index=False)

    # Log final comparison report as an artifact
    mlflow.log_artifact(str(output_file))

    print("\n" + "=" * 60)
    print("ALL 12 COMPARISON RUNS TRACKED SUCCESSFULLY!")
    print("=" * 60)

    print("Total runs:", len(results_df))
    print("Comparison report:", output_file)


if __name__ == "__main__":
    track_comparisons()
