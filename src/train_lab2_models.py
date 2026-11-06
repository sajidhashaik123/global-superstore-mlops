
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from lab2_preprocessing import build_preprocessor


def train_and_evaluate():

    project_root = Path(__file__).resolve().parent.parent

    results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    splits = ["80_20", "70_30"]
    strategies = ["standard", "minmax"]

    for split in splits:

        train_file = (
            project_root / "data" / "splits" / "lab2"
            / split / "train.csv"
        )

        test_file = (
            project_root / "data" / "splits" / "lab2"
            / split / "test.csv"
        )

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        X_train = train_df.drop(columns=["Profitability"])
        y_train = train_df["Profitability"]

        X_test = test_df.drop(columns=["Profitability"])
        y_test = test_df["Profitability"]

        for strategy in strategies:

            for model_name, model in models.items():

                print("\n" + "=" * 60)
                print("SPLIT:", split)
                print("PREPROCESSING:", strategy)
                print("MODEL:", model_name)
                print("=" * 60)

                pipeline = Pipeline([
                    ("preprocessing", build_preprocessor(strategy)),
                    ("model", model)
                ])

                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_test)
                probabilities = pipeline.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(
                    y_test, predictions, zero_division=0
                )
                recall = recall_score(
                    y_test, predictions, zero_division=0
                )
                f1 = f1_score(
                    y_test, predictions, zero_division=0
                )
                roc_auc = roc_auc_score(y_test, probabilities)

                print("Accuracy:", round(accuracy, 4))
                print("Precision:", round(precision, 4))
                print("Recall:", round(recall, 4))
                print("F1 Score:", round(f1, 4))
                print("ROC-AUC:", round(roc_auc, 4))

                print("\nConfusion Matrix:")
                print(confusion_matrix(y_test, predictions))

                print("\nClassification Report:")
                print(classification_report(
                    y_test, predictions, zero_division=0
                ))

                results.append({
                    "Split": split,
                    "Preprocessing": strategy,
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "ROC-AUC": roc_auc
                })

    results_df = pd.DataFrame(results)

    output_dir = project_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "lab2_model_comparison.csv"
    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print("\nResults saved to:", output_file)
    print("\nLab 2 Step 5 completed successfully!")


if __name__ == "__main__":
    train_and_evaluate()
