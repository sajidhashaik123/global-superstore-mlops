
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from lab2_preprocessing import build_preprocessor


def test_preprocessing():

    project_root = Path(__file__).resolve().parent.parent

    input_file = (
        project_root / "data" / "processed" / "v1"
        / "lab2_selected_features.csv"
    )

    df = pd.read_csv(input_file)

    X = df.drop(columns=["Profitability"])
    y = df["Profitability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    strategies = ["standard", "minmax"]

    for strategy in strategies:

        print(f"\n--- {strategy.upper()} SCALING ---")

        preprocessor = build_preprocessor(strategy)

        # Fit only on training data
        X_train_transformed = preprocessor.fit_transform(X_train)

        # Transform test data without fitting again
        X_test_transformed = preprocessor.transform(X_test)

        print("Training transformed shape:", X_train_transformed.shape)
        print("Testing transformed shape:", X_test_transformed.shape)

        print("Training records:", len(y_train))
        print("Testing records:", len(y_test))

        print("Preprocessing completed successfully!")


if __name__ == "__main__":
    test_preprocessing()
