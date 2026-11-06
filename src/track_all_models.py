
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lab2_preprocessing import build_preprocessor


def track_all_models():

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

    # Reproduce 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Models not yet successfully tracked
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    for model_name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessing", build_preprocessor("minmax")),
            ("model", classifier)
        ])

        with mlflow.start_run(run_name=model_name):

            # Train model
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            # Calculate evaluation metrics
            metrics = {
                "accuracy": accuracy_score(
                    y_test, predictions
                ),
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
            mlflow.log_param("algorithm", model_name)
            mlflow.log_param("preprocessing", "MinMaxScaler")
            mlflow.log_param("train_test_split", "80:20")
            mlflow.log_param("random_state", 42)

            # Track metrics
            mlflow.log_metrics(metrics)

            # Log trained model
            mlflow.sklearn.log_model(
                pipeline,
                name="profitability_model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            print("\nModel:", model_name)

            for metric, value in metrics.items():
                print(metric, ":", round(value, 4))

            print(
                "Run ID:",
                mlflow.active_run().info.run_id
            )

    print("\nAll remaining models tracked successfully!")


if __name__ == "__main__":
    track_all_models()
