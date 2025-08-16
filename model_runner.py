# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/16/2025
File: model_runner.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

import pandas as pd
import numpy as np
import yaml
import traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import json
from datetime import datetime

def log_failed_config(config, error_msg, log_path="logs/failed_configs.jsonl"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "error": error_msg
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_model(name, config, df):
    logs = [f"Running model: {name}"]
    try:
        target = config["target"]
        features = config.get("features", df.columns.drop(target).tolist())
        test_size = config.get("test_size", 0.2)
        random_state = config.get("random_state", 42)

        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if config["type"] == "linear":
            model = LinearRegression()
        elif config["type"] == "random_forest":
            model = RandomForestRegressor(**config.get("params", {}))
        elif config["type"] == "xgboost":
            model = XGBRegressor(**config.get("params", {}))
        else:
            raise ValueError(f"Unknown model type: {config['type']}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "timestamp": pd.Timestamp.now().isoformat(),
            "predictions": y_pred.tolist(),
            "actuals": y_test.tolist()
        }

        # Feature importance
        if hasattr(model, "feature_importances_"):
            metrics["feature_importance"] = dict(zip(features, model.feature_importances_))
        elif hasattr(model, "coef_"):
            metrics["feature_importance"] = dict(zip(features, model.coef_))

        # PCA variance
        pca = PCA()
        pca.fit(X)
        metrics["pca_variance"] = pca.explained_variance_ratio_.tolist()

        logs.append("Model trained and evaluated successfully.")

        return {
            "name": name,
            "config": config,
            "metrics": metrics,
            "logs": logs
        }

    except Exception as e:
        logs.append("Error during model run:")
        logs.append(traceback.format_exc())

        return {
            "name": name,
            "config": config,
            "metrics": {},
            "logs": logs
        }

def run_all_models(config_path="config.yaml", data_path="data/processed/train.csv"):
    df = pd.read_csv(data_path)
    config = load_config(config_path)
    results = []
    try:
        for name, cfg in config["models"].items():
            result = run_model(name, cfg, df)
            results.append(result)
    except Exception as e:
        result["metrics"] = {}
        result["logs"] = [str(e)]
        log_failed_config(config=result["config"], error_msg=str(e))

    return results
