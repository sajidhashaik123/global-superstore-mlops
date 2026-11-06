
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from lab2_preprocessing import build_preprocessor


def analyze_errors():

    project_root = Path(__file__).resolve().parent.parent

    train_file = (
        project_root / "data" / "splits" / "lab2"
        / "70_30" / "train.csv"
    )

    test_file = (
        project_root / "data" / "splits" / "lab2"
        / "70_30" / "test.csv"
    )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=["Profitability"])
    y_train = train_df["Profitability"]

    X_test = test_df.drop(columns=["Profitability"])
    y_test = test_df["Profitability"]

    pipeline = Pipeline([
        ("preprocessing", build_preprocessor("minmax")),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    errors = X_test.copy()
    errors["Actual"] = y_test.values
    errors["Predicted"] = predictions
    errors["Profitability_Probability"] = probabilities

    errors = errors[errors["Actual"] != errors["Predicted"]].copy()

    errors["Error_Type"] = errors.apply(
        lambda row: (
            "False Positive"
            if row["Actual"] == 0
            else "False Negative"
        ),
        axis=1
    )

    print("--- ERROR ANALYSIS ---")
    print("Total test records:", len(y_test))
    print("Incorrect predictions:", len(errors))

    print("\n--- ERROR TYPES ---")
    print(errors["Error_Type"].value_counts())

    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, predictions))

    print("\n--- FIRST 10 INCORRECT PREDICTIONS ---")
    print(errors.head(10).to_string(index=False))

    output_dir = project_root / "reports"
    output_file = output_dir / "lab2_incorrect_predictions.csv"

    errors.to_csv(output_file, index=False)

    print("\nIncorrect predictions saved to:", output_file)
    print("\nLab 2 Step 6 completed successfully!")


if __name__ == "__main__":
    analyze_errors()
