
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lab2_preprocessing import build_preprocessor


def track_first_run():

    project_root = Path(__file__).resolve().parent.parent

    # Configure MLflow
    database_path = project_root / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{database_path.as_posix()}"
    )
    mlflow.set_experiment("Global_Superstore_Profitability")

    # Load Global Superstore dataset
    data_path = (
        project_root / "data" / "processed" / "v1"
        / "lab2_selected_features.csv"
    )
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Profitability"])
    y = df["Profitability"]

    # Reproduce the 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create model pipeline
    model = Pipeline([
        ("preprocessing", build_preprocessor("minmax")),
        ("model", LogisticRegression(max_iter=1000))
    ])

    with mlflow.start_run(run_name="Logistic_Regression_MinMax_80_20"):

        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities)
        }

        # Track parameters
        mlflow.log_param("algorithm", "Logistic Regression")
        mlflow.log_param("preprocessing", "MinMaxScaler")
        mlflow.log_param("train_test_split", "80:20")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)

        # Track metrics
        mlflow.log_metrics(metrics)

        # Log trained model
        mlflow.sklearn.log_model(
            model,
            name="profitability_model"
        )

        print("MLflow run completed successfully!")
        print("\nEvaluation metrics:")

        for name, value in metrics.items():
            print(name, ":", round(value, 4))

        print("\nRun ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    track_first_run()
