
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lab2_preprocessing import build_preprocessor


def track_feature_experiments():

    root = Path(__file__).resolve().parent.parent

    # Configure MLflow
    database_path = root / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{database_path.as_posix()}"
    )

    mlflow.set_experiment("Global_Superstore_Profitability")

    # Load Global Superstore split
    train_file = root / "data" / "splits" / "lab2" / "80_20" / "train.csv"
    test_file = root / "data" / "splits" / "lab2" / "80_20" / "test.csv"

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    X_train = train.drop(columns=["Profitability"])
    y_train = train["Profitability"]

    X_test = test.drop(columns=["Profitability"])
    y_test = test["Profitability"]

    # Four experiment configurations
    configurations = [
        {"name": "All_Features_C1", "k": "all", "C": 1.0},
        {"name": "Selected_100_C1", "k": 100, "C": 1.0},
        {"name": "All_Features_C01", "k": "all", "C": 0.1},
        {"name": "All_Features_C10", "k": "all", "C": 10.0}
    ]

    for config in configurations:

        print("\n" + "=" * 50)
        print("Experiment:", config["name"])
        print("=" * 50)

        steps = [
            ("preprocessing", build_preprocessor("minmax"))
        ]

        # Apply feature selection when required
        if config["k"] != "all":
            steps.append(
                (
                    "feature_selection",
                    SelectKBest(
                        score_func=f_classif,
                        k=config["k"]
                    )
                )
            )

        # Add Logistic Regression
        steps.append(
            (
                "model",
                LogisticRegression(
                    C=config["C"],
                    max_iter=1000
                )
            )
        )

        pipeline = Pipeline(steps)

        with mlflow.start_run(run_name=config["name"]):

            # Train model
            pipeline.fit(X_train, y_train)

            # Predictions
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            # Evaluation metrics
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(
                    y_test, probabilities
                )
            }

            # Track parameters
            mlflow.log_params({
                "algorithm": "Logistic Regression",
                "preprocessing": "MinMaxScaler",
                "data_split": "80:20",
                "feature_selection": str(config["k"]),
                "C": config["C"],
                "random_state": 42,
                "dataset_version": "v1"
            })

            # Track metrics
            mlflow.log_metrics(metrics)

            # Track dataset information
            mlflow.set_tags({
                "dataset_name": "Global Superstore",
                "dataset_version": "v1",
                "target": "Profit > 0"
            })

            # Save trained model in MLflow
            mlflow.sklearn.log_model(
                pipeline,
                name="profitability_model",
                skops_trusted_types=[
                    "sklearn.feature_selection._univariate_selection.f_classif"
                ]
            )

            print("Accuracy:", round(metrics["accuracy"], 4))
            print("Precision:", round(metrics["precision"], 4))
            print("Recall:", round(metrics["recall"], 4))
            print("F1 Score:", round(metrics["f1_score"], 4))
            print("ROC-AUC:", round(metrics["roc_auc"], 4))

            print(
                "Run ID:",
                mlflow.active_run().info.run_id
            )

    print("\n" + "=" * 50)
    print("ALL FOUR EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    track_feature_experiments()
