
from pathlib import Path
import pandas as pd

def clean_comparison():

    project_root = Path(__file__).resolve().parent.parent

    input_file = (
        project_root / "reports"
        / "lab4_mlflow_comparison.csv"
    )

    df = pd.read_csv(input_file)

    # Keep only successful runs
    df = df[df["Run Status"] == "FINISHED"].copy()

    # Keep the latest successful run for each algorithm
    df = df.drop_duplicates(
        subset=["Algorithm"],
        keep="last"
    )

    output_file = (
        project_root / "reports"
        / "lab4_mlflow_final_comparison.csv"
    )

    df.to_csv(output_file, index=False)

    print("Final comparison report created!")
    print("\nNumber of models:", len(df))
    print(df[
        ["Algorithm", "Accuracy", "Precision",
         "Recall", "F1 Score", "ROC AUC"]
    ].to_string(index=False))

    print("\nSaved:", output_file)


if __name__ == "__main__":
    clean_comparison()
