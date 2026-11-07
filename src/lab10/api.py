
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import logging
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict


# ==================================================
# 1. PROJECT PATHS AND LOGGING
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "lab7" / "v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lab10_api")


# ==================================================
# 2. LOAD TRAINED MODEL AND PREPROCESSOR
# ==================================================

try:
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    classifier = joblib.load(MODEL_DIR / "classifier.joblib")

    with open(MODEL_DIR / "training_metadata.json", "r") as file:
        metadata = json.load(file)

    logger.info("Model and preprocessor loaded successfully.")

except Exception as error:
    logger.exception("Model loading failed.")
    raise RuntimeError(
        f"Unable to load model artifacts: {error}"
    ) from error


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Global Superstore Profitability API",
    description="ML inference API for predicting order profitability.",
    version="1.0.0"
)


# ==================================================
# 4. INPUT SCHEMA
# ==================================================

class OrderInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    Sales: float = Field(gt=0)
    Quantity: int = Field(gt=0)
    Discount: float = Field(ge=0, le=1)

    Shipping_Cost: float = Field(
        alias="Shipping Cost", ge=0
    )

    Order_Year: int = Field(
        alias="Order Year", ge=1900, le=2100
    )

    Order_Month: int = Field(
        alias="Order Month", ge=1, le=12
    )

    Ship_Mode: str = Field(
        alias="Ship Mode", min_length=1
    )

    Segment: str = Field(min_length=1)
    City: str = Field(min_length=1)
    State: str = Field(min_length=1)
    Country: str = Field(min_length=1)
    Market: str = Field(min_length=1)
    Region: str = Field(min_length=1)
    Category: str = Field(min_length=1)

    Sub_Category: str = Field(
        alias="Sub-Category", min_length=1
    )

    Order_Priority: str = Field(
        alias="Order Priority", min_length=1
    )


# ==================================================
# 5. OUTPUT SCHEMAS
# ==================================================

class PredictionResponse(BaseModel):

    prediction: str
    profitability: int
    probability: float
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):

    total_records: int
    predictions: List[PredictionResponse]
    model_version: str


# ==================================================
# 6. EXPECTED FEATURE COLUMNS
# ==================================================

FEATURE_COLUMNS = [
    "Sales",
    "Quantity",
    "Discount",
    "Shipping Cost",
    "Order Year",
    "Order Month",
    "Ship Mode",
    "Segment",
    "City",
    "State",
    "Country",
    "Market",
    "Region",
    "Category",
    "Sub-Category",
    "Order Priority"
]


# ==================================================
# 7. PREDICTION HELPER FUNCTION
# ==================================================

def predict_order(order: OrderInput):

    data = order.model_dump(by_alias=True)

    input_df = pd.DataFrame([data])

    input_df = input_df[FEATURE_COLUMNS]

    transformed_data = preprocessor.transform(input_df)

    prediction = int(classifier.predict(transformed_data)[0])

    probabilities = classifier.predict_proba(transformed_data)[0]

    positive_index = list(classifier.classes_).index(1)

    probability = float(probabilities[positive_index])

    return PredictionResponse(
        prediction=(
            "Profitable" if prediction == 1
            else "Not Profitable"
        ),
        profitability=prediction,
        probability=round(probability, 4),
        model_version="lab7-v1",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ==================================================
# 8. HEALTH CHECK ENDPOINT
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==================================================
# 9. MODEL METADATA ENDPOINT
# ==================================================

@app.get("/metadata")
def get_metadata():

    return {
        "model_version": "lab7-v1",
        "model_type": "Logistic Regression",
        "target": "Profitability",
        "metadata": metadata
    }


# ==================================================
# 10. SINGLE PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(input_data: OrderInput):

    try:
        return predict_order(input_data)

    except Exception as error:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal model error."
        ) from error


# ==================================================
# 11. BATCH PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def batch_predict(
    input_data: List[OrderInput] = Body(
        ...,
        min_length=1,
        max_length=100
    )
):

    try:

        results = [
            predict_order(order)
            for order in input_data
        ]

        return BatchPredictionResponse(
            total_records=len(results),
            predictions=results,
            model_version="lab7-v1"
        )

    except Exception as error:

        logger.exception("Batch prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed due to an internal model error."
        ) from error


# ==================================================
# 12. ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():

    return {
        "message": "Global Superstore Profitability API is running.",
        "docs": "/docs",
        "health": "/health"
    }
