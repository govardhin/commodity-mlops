import os
import yaml
from datetime import datetime

from src.data_ingestion import load_data
from src.data_validation import validate_columns
from src.feature_engineering import clean_numeric_columns, create_features
from src.model_trainer import train_and_select_best_model
from src.model_evaluation import save_metrics

# ===============================
# PATHS
# ===============================
DATA_PATH = "data/gold_data.csv"
CONFIG_PATH = "config/config.yaml"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")


# ===============================
# LOAD CONFIG
# ===============================
def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config


# ===============================
# MAIN PIPELINE
# ===============================
def run_pipeline():
    print("🔹 Loading configuration...")
    config = load_config()

    target_column = config["data"]["target_column"]
    feature_columns = config["data"]["feature_columns"]
    model_list = config["models"]
    selection_metric = config["selection_metric"]

    print("🔹 Loading dataset...")
    df = load_data(DATA_PATH)

    print("🔹 Cleaning numeric columns...")
    df = clean_numeric_columns(df)

    print("🔹 Creating features...")
    df = create_features(df)

    print("🔹 Validating required columns...")
    required_columns = feature_columns + [target_column]
    validate_columns(df, required_columns)

    X = df[feature_columns]
    y = df[target_column]

    print("🔹 Training multiple models...")
    best_model, all_metrics, best_model_name = train_and_select_best_model(
        X,
        y,
        model_list,
        selection_metric
    )

    print("🔹 Saving best model...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print("🔹 Saving metrics...")
    save_metrics(all_metrics, METRICS_PATH)

    print("🔹 Saving metadata...")
    metadata = {
        "best_model": best_model_name,
        "selection_metric": selection_metric,
        "dataset": os.path.basename(DATA_PATH),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    import json
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Pipeline Completed Successfully")
    print("Best Model:", best_model_name)
    print("Metrics:", all_metrics)


if __name__ == "__main__":
    run_pipeline()