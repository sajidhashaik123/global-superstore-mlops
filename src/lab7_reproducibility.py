import json
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from lab7_config import RANDOM_STATE, REPORT_DIR
from lab7_data_loader import load_datasets
from lab7_preprocessing import build_preprocessor


def run_reproducibility_test():

    print("\nLAB 7 REPRODUCIBILITY TEST")
    print("------------------------------")

    X_train, y_train, X_test, y_test = load_datasets()

    # First independent pipeline run
    preprocessor_1 = build_preprocessor()
    X_train_1 = preprocessor_1.fit_transform(X_train)
    X_test_1 = preprocessor_1.transform(X_test)

    model_1 = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model_1.fit(X_train_1, y_train)
    predictions_1 = model_1.predict(X_test_1)

    # Second independent pipeline run
    preprocessor_2 = build_preprocessor()
    X_train_2 = preprocessor_2.fit_transform(X_train)
    X_test_2 = preprocessor_2.transform(X_test)

    model_2 = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    model_2.fit(X_train_2, y_train)
    predictions_2 = model_2.predict(X_test_2)

    # Compare results
    predictions_match = (
        predictions_1 == predictions_2
    ).all()

    accuracy_1 = accuracy_score(y_test, predictions_1)
    accuracy_2 = accuracy_score(y_test, predictions_2)

    f1_1 = f1_score(y_test, predictions_1)
    f1_2 = f1_score(y_test, predictions_2)

    print("Run 1 Accuracy:", round(accuracy_1, 4))
    print("Run 2 Accuracy:", round(accuracy_2, 4))

    print("Run 1 F1-score:", round(f1_1, 4))
    print("Run 2 F1-score:", round(f1_2, 4))

    print("Predictions match:", predictions_match)

    reproducibility_passed = (
        predictions_match
        and accuracy_1 == accuracy_2
        and f1_1 == f1_2
    )

    results = {
        "experiment": "Lab 7 - Reproducibility Test",
        "dataset": "Global Superstore",
        "random_state": RANDOM_STATE,
        "run_1_accuracy": float(accuracy_1),
        "run_2_accuracy": float(accuracy_2),
        "run_1_f1_score": float(f1_1),
        "run_2_f1_score": float(f1_2),
        "predictions_match": bool(predictions_match),
        "reproducibility_passed": bool(reproducibility_passed)
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        REPORT_DIR / "reproducibility_results.json",
        "w"
    ) as file:
        json.dump(results, file, indent=4)

    if reproducibility_passed:
        print("\nREPRODUCIBILITY TEST PASSED!")
    else:
        print("\nREPRODUCIBILITY TEST FAILED!")

    print("Results saved to:", REPORT_DIR / "reproducibility_results.json")


if __name__ == "__main__":
    run_reproducibility_test()