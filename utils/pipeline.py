# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/16/2025
File: pipeline.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

import os
import json
import logging
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.model_builder import build_model
from utils.file_io import load_real_data
from utils.validation import validate_model_config
import sys
sys.path.append("..")

# 📋 Logging setup
logging.basicConfig(
    filename="dashboard.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def save_model_json(name, config, metrics, logs, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    model_data = {
        "name": name,
        "config": config,
        "metrics": {
            "MSE": metrics["MSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "timestamp": metrics["timestamp"],
            "pca_variance": metrics["pca_variance"],
            "feature_importance": metrics["feature_importance"],
            "predictions": metrics["predictions"],
            "actuals": metrics["actuals"]
        },
        "logs": logs
    }
    file_path = os.path.join(output_dir, f"{name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)
    logging.info(f"📁 Saved model JSON to {file_path}")

def validate_output_files(model_configs, output_dir="models"):
    missing = []
    for cfg in model_configs:
        path = os.path.join(output_dir, f"{cfg['name']}.json")
        if not os.path.exists(path):
            missing.append(cfg['name'])
    if missing:
        logging.warning(f"⚠️ Missing model files: {missing}")
    else:
        logging.info("✅ All model files present.")


# 📁 Load model configs from JSON
def load_model_configs(path="config/model_configs.json"):
    with open(path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    return [cfg for cfg in configs if validate_model_config(cfg)]

# 🧪 Train model and compute metrics
from sklearn.decomposition import PCA

def train_model_and_get_metrics(model, params, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # PCA variance
    pca = PCA()
    pca.fit(X_train)
    pca_variance = pca.explained_variance_ratio_.tolist()

    # Feature importance (if supported)
    try:
        importance = model.feature_importances_.tolist()
        feature_names = X_train.columns.tolist()
        feature_importance = dict(zip(feature_names, importance))
    except AttributeError:
        feature_importance = {}

    return {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "timestamp": datetime.utcnow().isoformat(),
        "params": params,
        "predictions": y_pred.tolist(),
        "actuals": y_test.tolist(),
        "pca_variance": pca_variance,
        "feature_importance": feature_importance
    }

# 🚀 Main pipeline
def run_model_pipeline_and_save_metrics(
    output_dir="outputs",
    data_path=r"data/processed/train.csv",
    config_path="config/model_configs.json"
):
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_real_data(data_path)
    model_configs = load_model_configs(config_path)

    for cfg in model_configs:
        try:
            model_name = cfg["name"]
            logging.info(f"🚀 Training model: {model_name}")
            logs = [f"Started training: {model_name}"]

            model = build_model(cfg["model"], cfg["params"])
            logs.append(f"Built model: {cfg['model']} with params: {cfg['params']}")

            metrics = train_model_and_get_metrics(model, cfg["params"], X_train, X_test, y_train, y_test)
            logs.append(f"Computed metrics: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

            # Save metrics to outputs/ (optional legacy)
            metrics_path = os.path.join(output_dir, f"metrics_{model_name}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logging.info(f"📊 Saved metrics to {metrics_path}")

            # Save full model JSON to models/
            save_model_json(model_name, cfg, metrics, logs, output_dir="models")

        except Exception as e:
            logging.error(f"❌ Failed to train {cfg['name']}: {e}")

    validate_output_files(model_configs, output_dir="models")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run model training pipeline and save metrics.")
    parser.add_argument("--data", type=str, default="data/processed/train.csv", help="Path to training data")
    parser.add_argument("--config", type=str, default="config/model_configs.json", help="Path to model config file")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save metrics")
    parser.add_argument("--models", type=str, default="models", help="Directory to save model JSONs")

    args = parser.parse_args()

    run_model_pipeline_and_save_metrics(
        output_dir=args.output,
        data_path=args.data,
        config_path=args.config
    )

