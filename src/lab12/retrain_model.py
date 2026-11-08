import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


from lab12.config import (
    LAB12_MODEL_DIR,
    RETRAIN_START,
    RETRAIN_END,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    MAX_ITER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DATA = (
    PROJECT_ROOT / "data" / "processed" /
    "lab12" / "retraining_2011_2013.csv"
)

MODEL_VERSION = "lab12-v2"
MODEL_DIR = LAB12_MODEL_DIR / "v2"


def retrain_model():

    print("Loading retraining dataset...")

    if not TRAINING_DATA.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_DATA}"
        )

    df = pd.read_csv(TRAINING_DATA)

    X_train = df[FEATURE_COLUMNS]
    y_train = df[TARGET_COLUMN]

    print(f"Training records: {len(df)}")
    print(f"Input features: {len(FEATURE_COLUMNS)}")
    print("\nTarget distribution:")
    print(y_train.value_counts().sort_index())

    print("\nBuilding preprocessing pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                MinMaxScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
        ]
    )

    classifier = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\nTraining Logistic Regression model...")

    pipeline.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        pipeline.named_steps["preprocessor"],
        MODEL_DIR / "preprocessor.joblib"
    )

    joblib.dump(
        pipeline.named_steps["classifier"],
        MODEL_DIR / "classifier.joblib"
    )

    transformed_features = (
        pipeline.named_steps["preprocessor"]
        .transform(X_train)
        .shape[1]
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "Logistic Regression",
        "dataset": str(TRAINING_DATA.relative_to(PROJECT_ROOT)),
        "training_start": RETRAIN_START,
        "training_end": RETRAIN_END,
        "training_records": int(len(df)),
        "raw_input_features": len(FEATURE_COLUMNS),
        "encoded_features": int(transformed_features),
        "target_column": TARGET_COLUMN,
        "target_definition": "Profitability = 1 if Profit > 0, otherwise 0",
        "numerical_preprocessing": "MinMaxScaler",
        "categorical_preprocessing": "OneHotEncoder(handle_unknown='ignore')",
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    with (MODEL_DIR / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nMODEL RETRAINING COMPLETED")
    print("-" * 40)
    print(f"Model version: {MODEL_VERSION}")
    print(f"Training records: {len(df)}")
    print(f"Encoded features: {transformed_features}")
    print(f"Saved model directory: {MODEL_DIR}")


if __name__ == "__main__":
    retrain_model()