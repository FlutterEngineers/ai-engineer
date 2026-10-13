
import math
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

artifact = joblib.load("best_model.joblib")
model = artifact["model"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

app = FastAPI(title="ML Prediction API")


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


@app.get("/")
def home():
    return {
        "message": "ML Prediction API is running",
        "model": artifact["model_name"],
        "expected_features": len(feature_names),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    values = request.features

    if any(not math.isfinite(x) or x < 0 for x in values):
        raise HTTPException(
            status_code=422,
            detail="Features must be finite, non-negative numbers.",
        )

    sample = pd.DataFrame([values], columns=feature_names)
    prediction = int(model.predict(sample)[0])

    result = {
        "predicted_class": target_names[prediction],
        "model": artifact["model_name"],
        "educational_use_only": True,
        "disclaimer": "Not a medical diagnostic tool.",
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        result["probabilities"] = {
            target_names[int(class_id)]: round(float(probability), 4)
            for class_id, probability in zip(model.classes_, probabilities)
        }

    return result