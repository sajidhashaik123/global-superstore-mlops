
from pathlib import Path
import joblib
import pandas as pd
import json

def verify_model():

    project_root = Path(__file__).resolve().parent.parent

    model_file = (
        project_root / "models" / "lab2" / "v1"
        / "profitability_model.joblib"
    )

    metadata_file = (
        project_root / "models" / "lab2" / "v1"
        / "model_metadata.json"
    )

    # Load saved model without training again
    model = joblib.load(model_file)

    # Load test dataset
    test_file = (
        project_root / "data" / "splits" / "lab2"
        / "70_30" / "test.csv"
    )

    test_df = pd.read_csv(test_file)

    X_test = test_df.drop(columns=["Profitability"])

    # Predict using the saved model
    predictions = model.predict(X_test.head(5))

    print("--- SAVED MODEL VERIFICATION ---")
    print("Model loaded successfully!")

    print("\nPredictions for first 5 records:")
    print(predictions)

    # Read metadata
    with open(metadata_file, "r") as file:
        metadata = json.load(file)

    print("\nModel name:", metadata["model_name"])
    print("Model version:", metadata["model_version"])
    print("Algorithm:", metadata["algorithm"])

    print("\nSaved model works successfully!")

if __name__ == "__main__":
    verify_model()
