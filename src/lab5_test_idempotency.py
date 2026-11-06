
from pathlib import Path
import pandas as pd

from lab5_preprocessing_pipeline import preprocess_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "global_superstore.csv"


def main():
    original_df = pd.read_csv(RAW_PATH)

    # Run preprocessing twice on the same original dataset
    first_result = preprocess_dataset(original_df)
    second_result = preprocess_dataset(original_df)

    # Compare both results
    are_equal = first_result.equals(second_result)

    print("--- IDEMPOTENCY TEST ---")
    print("First result shape:", first_result.shape)
    print("Second result shape:", second_result.shape)
    print("Both results are identical:", are_equal)

    if are_equal:
        print("IDEMPOTENCY TEST PASSED")
    else:
        print("IDEMPOTENCY TEST FAILED")


if __name__ == "__main__":
    main()
