import logging
import math
import os
import time
import uuid

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

APP_NAME = os.getenv("APP_NAME", "ML Prediction API")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure application logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("ml_api")

# Load the trained model
artifact = joblib.load("best_model.joblib")
model = artifact["model"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

logger.info("Model loaded: %s", artifact["model_name"])

# Create FastAPI application
app = FastAPI(title=APP_NAME)


# Request-tracing middleware
@app.middleware("http")
async def request_tracing(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "Request failed: id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Attach request ID to the response
    response.headers["X-Request-ID"] = request_id

    # Log request details and total elapsed time
    logger.info(
        "Request completed: id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    return response


# Request validation schema
class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "ML Prediction API is running",
        "model": artifact["model_name"],
        "expected_features": len(feature_names),
    }


# Health-check endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": artifact["model_name"],
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Measure prediction processing time
    start_time = time.perf_counter()

    values = request.features

    # Reject invalid numeric inputs
    if any(not math.isfinite(x) or x < 0 for x in values):
        logger.warning("Rejected prediction request: invalid feature values")

        raise HTTPException(
            status_code=422,
            detail="Features must be finite, non-negative numbers.",
        )

    sample = pd.DataFrame([values], columns=feature_names)

    try:
        prediction = int(model.predict(sample)[0])

        result = {
            "predicted_class": target_names[prediction],
            "model": artifact["model_name"],
            "educational_use_only": True,
            "disclaimer": "Not a medical diagnostic tool.",
        }

        # Include class probabilities when supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]

            result["probabilities"] = {
                target_names[int(class_id)]: round(float(probability), 4)
                for class_id, probability in zip(
                    model.classes_,
                    probabilities,
                )
            }

        # Calculate elapsed prediction time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction completed successfully in %.2f ms",
            elapsed_ms,
        )

        return result

    except Exception as exc:
        logger.exception("Prediction failed due to an unexpected error")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        ) from exc
