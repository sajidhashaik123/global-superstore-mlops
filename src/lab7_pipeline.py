from lab7_preprocessing import preprocess_datasets
from lab7_validation import validate_preprocessed_data
from lab7_training import train_model
from lab7_evaluation import evaluate_model
from lab7_inference import run_sample_inference


def run_pipeline():

    print("\n===================================")
    print(" GLOBAL SUPERSTORE - LAB 7 PIPELINE")
    print("===================================")

    print("\nSTAGE 1: PREPROCESSING")
    preprocess_datasets()

    print("\nSTAGE 2: DATA VALIDATION")
    validation_result = validate_preprocessed_data()

    if not validation_result:
        raise RuntimeError("Pipeline stopped: validation failed.")

    print("\nSTAGE 3: MODEL TRAINING")
    train_model()

    print("\nSTAGE 4: MODEL EVALUATION")
    metrics = evaluate_model()

    print("\nSTAGE 5: MODEL INFERENCE")
    run_sample_inference()

    print("\n===================================")
    print(" LAB 7 PIPELINE COMPLETED")
    print("===================================")

    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_pipeline()