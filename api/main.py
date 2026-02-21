import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ==============================
# Initialize FastAPI App
# ==============================

app = FastAPI(
    title="Commodity Price Forecasting API",
    description="Production-ready ML Model Serving Layer",
    version="1.0"
)

# ==============================
# Load Trained Model
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found in artifacts folder.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ==============================
# Request Schema
# ==============================

class PredictionRequest(BaseModel):
    Open: float
    High: float
    Low: float
    Vol: float
    Prev_Close: float


# ==============================
# Health Check Endpoint
# ==============================

@app.get("/")
def health_check():
    return {"status": "API is running successfully"}


# ==============================
# Prediction Endpoint
# ==============================

@app.post("/predict")
def predict(data: PredictionRequest):

    input_array = np.array([[ 
        data.Open,
        data.High,
        data.Low,
        data.Vol,
        data.Prev_Close
    ]])

    prediction = model.predict(input_array)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }
# ==============================
# Metrics Endpoint
# ==============================

import json

METRICS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "metrics.json")

@app.get("/metrics")
def get_metrics():

    if not os.path.exists(METRICS_PATH):
        return {"error": "Metrics file not found."}

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    return metrics
# ==============================
# Model Info Endpoint
# ==============================

METADATA_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model_metadata.json")

@app.get("/model-info")
def get_model_info():

    if not os.path.exists(METADATA_PATH):
        return {"error": "Model metadata not found."}

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return metadata