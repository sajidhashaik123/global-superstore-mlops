
from pathlib import Path
import pandas as pd

from lab5_data_validation import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "global_superstore.csv"


def main():
    # Load a temporary sample; never modify the original CSV
    df = pd.read_csv(RAW_PATH).head(5).copy()

    # Introduce three intentional errors
    df.loc[df.index[0], "Ship Mode"] = "Invalid Mode"
    df.loc[df.index[1], "Order Date"] = "Invalid Date"
    df.loc[df.index[2], "Sales"] = -100

    print("--- TESTING INVALID DATA ---")

    errors = validate_dataset(df)

    print("\n--- INVALID DATA TEST RESULT ---")

    if (
        any("Invalid categories" in error for error in errors)
        and any("Invalid dates" in error for error in errors)
        and any("Invalid values" in error for error in errors)
    ):
        print("INVALID DATA DETECTION TEST PASSED")
        print("All three intentional errors were detected.")
    else:
        print("INVALID DATA DETECTION TEST FAILED")
        print("Detected errors:", errors)


if __name__ == "__main__":
    main()
