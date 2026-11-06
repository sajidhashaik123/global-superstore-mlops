
from pathlib import Path
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


def export_mlflow_results():

    project_root = Path(__file__).resolve().parent.parent

    # Configure MLflow
    database_path = project_root / "mlflow.db"
    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    # Get experiment
    experiment = client.get_experiment_by_name(
        "Global_Superstore_Profitability"
    )

    if experiment is None:
        raise ValueError("MLflow experiment not found!")

    # Retrieve all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time ASC"]
    )

    results = []

    for run in runs:

        results.append({
            "Run ID": run.info.run_id,
            "Run Name": run.data.tags.get("mlflow.runName", ""),
            "Algorithm": run.data.params.get("algorithm", ""),
            "Preprocessing": run.data.params.get("preprocessing", ""),
            "Data Split": run.data.params.get("train_test_split", ""),
            "Random State": run.data.params.get("random_state", ""),
            "Accuracy": run.data.metrics.get("accuracy"),
            "Precision": run.data.metrics.get("precision"),
            "Recall": run.data.metrics.get("recall"),
            "F1 Score": run.data.metrics.get("f1_score"),
            "ROC AUC": run.data.metrics.get("roc_auc"),
            "Run Status": run.info.status
        })

    # Create comparison report
    report = pd.DataFrame(results)

    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "lab4_mlflow_comparison.csv"

    report.to_csv(output_file, index=False)

    print("MLflow comparison report created successfully!")
    print("\nTotal runs:", len(report))
    print("\nComparison results:")
    print(report.to_string(index=False))
    print("\nSaved file:", output_file)


if __name__ == "__main__":
    export_mlflow_results()
