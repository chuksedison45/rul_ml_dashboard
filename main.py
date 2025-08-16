# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/15/2025
File: main.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""
import sys

from src.utils.config_validator import validate_config

sys.path.append('C:/Users/chuks/OneDrive/Brand_Projects')
sys.path.append('C:/Users/chuks/OneDrive/Brand_Projects/rul_dashboard')
sys.path.append("../..")
import argparse
import yaml
from src.features.transform import preprocess_data
from src.features.select import apply_feature_selection
from src.models.train_ml import train_ml_model
from src.models.train_dl import train_dl_model
from src.utils.io import load_data, save_artifacts
from src.dashboard.layout import launch_dashboard

def run_pipeline(config_path):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    errors = validate_config(config)
    if errors:
        for err in errors:
            print(f"❌ Config Error: {err}")
            logger.error(err)
        raise ValueError("Invalid configuration. See logs for details.")
    df = load_data(config["data"]["path"])
    X, y = preprocess_data(df, config["data"]["target"], config["features"])
    X_selected = apply_feature_selection(X, y, config["features"])

    ml_model, ml_metrics = train_ml_model(X_selected, y, config["models"]["ml"])
    dl_model, dl_metrics = train_dl_model(X_selected, y, config["models"]["dl"])

    save_artifacts(ml_model, dl_model, ml_metrics, dl_metrics)

    if config["dashboard"]["enable"]:
        launch_dashboard(X_selected, y, config["dashboard"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
