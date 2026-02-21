import os
import json
import datetime

from src.data_ingestion import load_data
from src.data_validation import validate_columns
from src.feature_engineering import clean_numeric_columns, create_features
from src.model_trainer import train_model, save_model
from src.model_evaluation import evaluate_model


# ===============================
# PATH CONFIGURATION
# ===============================

DATA_PATH = "data/gold_data.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

required_columns = ["Open", "High", "Low", "Vol.", "Price"]


# ===============================
# MAIN PIPELINE FUNCTION
# ===============================

def run_pipeline():

    # Ensure artifacts folder exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1️⃣ Load Data
    df = load_data(DATA_PATH)

    # 2️⃣ Validate Required Columns
    validate_columns(df, required_columns)

    # 3️⃣ Clean and Feature Engineering
    df = clean_numeric_columns(df)
    df = create_features(df)

    # 4️⃣ Prepare Training Data
    X = df[["Open", "High", "Low", "Vol.", "Prev_Close"]]
    y = df["Price"]

    # 5️⃣ Train Model
    model = train_model(X, y)

    # 6️⃣ Save Model
    save_model(model, MODEL_PATH)

    # 7️⃣ Evaluate Model
    metrics = evaluate_model(model, X, y, METRICS_PATH)

    # 8️⃣ Save Model Metadata
    metadata = {
        "model_version": "v1",
        "trained_at": str(datetime.datetime.now()),
        "dataset_size": len(df),
        "model_type": type(model).__name__
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Pipeline Completed Successfully")
    print(metrics)


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    run_pipeline()