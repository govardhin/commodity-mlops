from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


def train_and_select_best_model(X, y, model_list, selection_metric):

    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    for model_name in model_list:
        if model_name == "LinearRegression":
            models["LinearRegression"] = LinearRegression()
        elif model_name == "RandomForest":
            models["RandomForest"] = RandomForestRegressor(random_state=42)
        elif model_name == "GradientBoosting":
            models["GradientBoosting"] = GradientBoostingRegressor(random_state=42)

    all_metrics = {}
    best_model = None
    best_score = float("inf")
    best_model_name = None

    for name, model in models.items():

        # Train only on training data
        model.fit(X_train, y_train)

        # Evaluate on test data
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        all_metrics[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2)
        }

        if selection_metric == "RMSE":
            score = rmse
        elif selection_metric == "MAE":
            score = mae
        else:
            score = rmse

        if score < best_score:
            best_score = score
            best_model = model
            best_model_name = name

    return best_model, all_metrics, best_model_name