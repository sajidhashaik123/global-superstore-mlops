
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel, Field, ConfigDict

from src.lab13.config import (
    PROJECT_ROOT,
    BASELINE_MODEL_DIR,
    RETRAINED_MODEL_DIR,
    BASELINE_MODEL_VERSION,
    RETRAINED_MODEL_VERSION,
    INITIAL_PRODUCTION_MODEL,
    LAB13_LOG_DIR,
    FEATURE_COLUMNS,
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lab13_api")

MODEL_DIRECTORIES = {
    BASELINE_MODEL_VERSION: BASELINE_MODEL_DIR,
    RETRAINED_MODEL_VERSION: RETRAINED_MODEL_DIR,
}

# Select the production model by default.
MODEL_VERSION = os.getenv(
    "LAB13_MODEL_VERSION",
    INITIAL_PRODUCTION_MODEL,
)

if MODEL_VERSION not in MODEL_DIRECTORIES:
    raise ValueError(
        f"Unsupported model version: {MODEL_VERSION}"
    )

MODEL_DIR = MODEL_DIRECTORIES[MODEL_VERSION]

LAB13_LOG_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_LOG_FILE = LAB13_LOG_DIR / "api_requests.jsonl"


# ==================================================
# 2. LOAD SELECTED MODEL
# ==================================================

try:
    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.joblib"
    )

    classifier = joblib.load(
        MODEL_DIR / "classifier.joblib"
    )

    with open(
        MODEL_DIR / "training_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    logger.info(
        "Loaded model version: %s",
        MODEL_VERSION,
    )

except Exception as error:
    logger.exception("Model loading failed.")

    raise RuntimeError(
        f"Unable to load model artifacts: {error}"
    ) from error


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Global Superstore Lab13 MLOps API",
    description=(
        "End-to-end profitability prediction API "
        "with model versioning and monitoring."
    ),
    version="1.0.0",
)


# ==================================================
# 4. MONITORING COUNTERS
# ==================================================

request_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0,
    "status_codes": {},
}


# ==================================================
# 5. REQUEST MONITORING MIDDLEWARE
# ==================================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()

    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    finally:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        request_metrics["total_requests"] += 1
        request_metrics["total_latency_ms"] += latency_ms

        if status_code < 400:
            request_metrics["successful_requests"] += 1
        else:
            request_metrics["failed_requests"] += 1

        code = str(status_code)

        request_metrics["status_codes"][code] = (
            request_metrics["status_codes"].get(code, 0) + 1
        )

        log_entry = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "method": request.method,
            "endpoint": request.url.path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            "model_version": MODEL_VERSION,
        }

        try:
            with open(
                REQUEST_LOG_FILE,
                "a",
                encoding="utf-8",
            ) as file:
                file.write(json.dumps(log_entry) + "\n")

        except OSError:
            logger.exception("Unable to write request log.")


# ==================================================
# 6. INPUT SCHEMA
# ==================================================

class OrderInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    Sales: float = Field(gt=0)
    Quantity: int = Field(gt=0)
    Discount: float = Field(ge=0, le=1)

    Shipping_Cost: float = Field(
        alias="Shipping Cost",
        ge=0,
    )

    Order_Year: int = Field(
        alias="Order Year",
        ge=1900,
        le=2100,
    )

    Order_Month: int = Field(
        alias="Order Month",
        ge=1,
        le=12,
    )

    Ship_Mode: str = Field(
        alias="Ship Mode",
        min_length=1,
    )

    Segment: str = Field(min_length=1)
    City: str = Field(min_length=1)
    State: str = Field(min_length=1)
    Country: str = Field(min_length=1)
    Market: str = Field(min_length=1)
    Region: str = Field(min_length=1)
    Category: str = Field(min_length=1)

    Sub_Category: str = Field(
        alias="Sub-Category",
        min_length=1,
    )

    Order_Priority: str = Field(
        alias="Order Priority",
        min_length=1,
    )


# ==================================================
# 7. OUTPUT SCHEMAS
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
# 8. PREDICTION FUNCTION
# ==================================================

def predict_order(order: OrderInput):

    data = order.model_dump(by_alias=True)

    input_df = pd.DataFrame([data])

    input_df = input_df[FEATURE_COLUMNS]

    transformed_data = preprocessor.transform(
        input_df
    )

    prediction = int(
        classifier.predict(transformed_data)[0]
    )

    probabilities = classifier.predict_proba(
        transformed_data
    )[0]

    positive_index = list(
        classifier.classes_
    ).index(1)

    probability = float(
        probabilities[positive_index]
    )

    return PredictionResponse(
        prediction=(
            "Profitable"
            if prediction == 1
            else "Not Profitable"
        ),
        profitability=prediction,
        probability=round(probability, 4),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(
            timezone.utc
        ).isoformat(),
    )


# ==================================================
# 9. HEALTH ENDPOINT
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ==================================================
# 10. MODEL METADATA ENDPOINT
# ==================================================

@app.get("/metadata")
def get_metadata():

    return {
        "model_version": MODEL_VERSION,
        "model_type": metadata.get(
            "algorithm",
            "Logistic Regression",
        ),
        "target": "Profitability",
        "metadata": metadata,
    }


# ==================================================
# 11. SINGLE PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(input_data: OrderInput):

    try:
        return predict_order(input_data)

    except Exception as error:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal model error.",
        ) from error


# ==================================================
# 12. BATCH PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def batch_predict(
    input_data: List[OrderInput] = Body(
        ...,
        min_length=1,
        max_length=100,
    ),
):

    try:

        results = [
            predict_order(order)
            for order in input_data
        ]

        return BatchPredictionResponse(
            total_records=len(results),
            predictions=results,
            model_version=MODEL_VERSION,
        )

    except Exception as error:

        logger.exception("Batch prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed due to an internal model error.",
        ) from error


# ==================================================
# 13. MONITORING METRICS ENDPOINT
# ==================================================

@app.get("/monitoring/metrics")
def get_monitoring_metrics():

    total = request_metrics["total_requests"]

    average_latency = (
        request_metrics["total_latency_ms"] / total
        if total > 0
        else 0.0
    )

    return {
        "model_version": MODEL_VERSION,
        "total_requests": total,
        "successful_requests": request_metrics[
            "successful_requests"
        ],
        "failed_requests": request_metrics[
            "failed_requests"
        ],
        "average_latency_ms": round(
            average_latency,
            3,
        ),
        "status_codes": request_metrics[
            "status_codes"
        ],
        "request_log_file": str(REQUEST_LOG_FILE),
    }


# ==================================================
# 14. ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():

    return {
        "message": "Global Superstore Lab13 API is running.",
        "model_version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "monitoring": "/monitoring/metrics",
    }
