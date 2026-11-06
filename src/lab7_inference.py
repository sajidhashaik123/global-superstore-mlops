import joblib
import pandas as pd

from lab7_config import MODEL_DIR, REPORT_DIR, FEATURE_COLUMNS
from lab7_data_loader import load_datasets


def predict_new_data(input_data):

    # Check required input columns
    missing_columns = set(FEATURE_COLUMNS) - set(input_data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if input_data[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Input data contains missing values.")

    # Load saved preprocessing and classification models
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    model = joblib.load(MODEL_DIR / "classifier.joblib")

    # Apply the same preprocessing used during training
    transformed_data = preprocessor.transform(input_data[FEATURE_COLUMNS])

    # Generate predictions
    predictions = model.predict(transformed_data)
    probabilities = model.predict_proba(transformed_data)[:, 1]

    # Prepare results
    results = pd.DataFrame({
        "Predicted_Profitability": predictions,
        "Profitability_Probability": probabilities
    })

    results["Prediction_Label"] = results[
        "Predicted_Profitability"
    ].map({
        0: "Non-Profitable",
        1: "Profitable"
    })

    return results


def run_sample_inference():

    # Load original test features without the target column
    _, _, X_test, _ = load_datasets()

    # Select five orders for demonstration
    sample_data = X_test.head(5).copy()

    results = predict_new_data(sample_data)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        REPORT_DIR / "sample_inference_predictions.csv",
        index=False
    )

    print("\nLAB 7 INFERENCE")
    print("------------------------------")
    print("Number of new orders:", len(sample_data))
    print("\nPrediction Results:")
    print(results.to_string(index=False))
    print("\nInference completed successfully!")
    print("Results saved to:", REPORT_DIR / "sample_inference_predictions.csv")


if __name__ == "__main__":
    run_sample_inference()