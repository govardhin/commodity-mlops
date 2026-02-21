import json
import os

def save_metrics(all_metrics, metrics_path):

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Metrics saved successfully.")