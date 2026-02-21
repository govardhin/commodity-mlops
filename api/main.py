from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import os
import numpy as np

app = FastAPI(
    title="Commodity Price Forecasting API",
    description="Production-ready MLOps serving layer",
    version="1.0.0"
)

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

# ===============================
# LOAD MODEL ON STARTUP
# ===============================
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# REQUEST SCHEMA
# ===============================
class PredictionRequest(BaseModel):
    Open: float
    High: float
    Low: float
    Vol: float
    Prev_Close: float

# ===============================
# ROOT
# ===============================
@app.get("/")
def root():
    return {
        "message": "Commodity Price Forecasting API is running",
        "status": "active"
    }

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ===============================
# MODEL INFO
# ===============================
@app.get("/model-info")
def model_info():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        return metadata
    return {"message": "Metadata not found"}

# ===============================
# METRICS
# ===============================
@app.get("/metrics")
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return metrics
    return {"message": "Metrics not found"}

# ===============================
# PREDICT
# ===============================
@app.post("/predict")
def predict(data: PredictionRequest):
    input_data = np.array([[ 
        data.Open,
        data.High,
        data.Low,
        data.Vol,
        data.Prev_Close
    ]])

    prediction = model.predict(input_data)

    return {
        "predicted_price": float(prediction[0])
    }