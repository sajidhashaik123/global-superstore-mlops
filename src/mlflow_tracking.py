
import mlflow
from pathlib import Path

def setup_mlflow():
    project_root = Path(__file__).resolve().parent.parent

    # Create SQLite database for MLflow tracking
    database_path = project_root / "mlflow.db"
    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    # Configure MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Create or select experiment
    experiment = mlflow.set_experiment(
        "Global_Superstore_Profitability"
    )

    print("MLflow tracking configured successfully!")
    print("Experiment name:", experiment.name)
    print("Experiment ID:", experiment.experiment_id)
    print("Tracking URI:", tracking_uri)

if __name__ == "__main__":
    setup_mlflow()
