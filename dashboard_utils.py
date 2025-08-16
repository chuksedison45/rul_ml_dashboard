# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/16/2025
File: dashboard_utils.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

# dashboard_utils.py
import os
import json

def load_models(models_dir="models"):
    models = []
    for fname in os.listdir(models_dir):
        if fname.endswith(".json"):
            path = os.path.join(models_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    models.append(json.load(f))
            except Exception as e:
                print(f"⚠️ Failed to load {fname}: {e}")
    return models

def get_latest_model_timestamp(models_dir="models"):
    timestamps = []
    for fname in os.listdir(models_dir):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(models_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    ts = data.get("metrics", {}).get("timestamp")
                    if ts:
                        timestamps.append(ts)
            except Exception:
                continue
    return max(timestamps) if timestamps else "N/A"

def validate_model_outputs(model_configs, models_dir="models"):
    missing = []
    for cfg in model_configs:
        expected_path = os.path.join(models_dir, f"{cfg['name']}.json")
        if not os.path.exists(expected_path):
            missing.append(cfg['name'])
    return missing


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_feature_importance(feature_importance):
    if not feature_importance:
        st.info("No feature importance available.")
        return
    df = pd.DataFrame.from_dict(feature_importance, orient="index", columns=["Importance"])
    df = df.sort_values("Importance", ascending=False)
    st.bar_chart(df)

def plot_pca_variance(pca_variance):
    if not pca_variance:
        st.info("No PCA variance available.")
        return
    df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca_variance))],
        "Explained Variance": pca_variance
    })
    st.line_chart(df.set_index("Component"))

def plot_predictions_vs_actuals(predictions, actuals):
    if not predictions or not actuals:
        st.info("No predictions or actuals available.")
        return
    df = pd.DataFrame({"Predicted": predictions, "Actual": actuals})
    st.scatter_chart(df)

def export_json_button(data, label, filename):
    st.download_button(
        label=label,
        data=json.dumps(data, indent=2),
        file_name=filename,
        mime="application/json"
    )

def get_best_model(models, metric="R2"):
    scored = [
        (m["name"], m["metrics"].get(metric, 0), m)
        for m in models if metric in m["metrics"]
    ]
    if not scored:
        return None
    best = max(scored, key=lambda x: x[1])
    return best[2]  # return full model dict