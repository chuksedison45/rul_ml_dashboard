# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/16/2025
File: dashboard_refresh.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

import seaborn as sns
from packaging import version
# dashboard.py
import streamlit as st
import os
import json
from matplotlib import pyplot as plt

from utils.pipeline import run_model_pipeline_and_save_metrics
from utils.file_io import load_real_data
import pandas as pd
from model_runner import run_all_models
from dashboard_utils import load_models, get_latest_model_timestamp, get_best_model, export_json_button
from dashboard_utils import (
    plot_feature_importance,
    plot_pca_variance,
    plot_predictions_vs_actuals
)
from datetime import datetime

st.set_page_config(page_title="Model Dashboard", layout="wide")


# 🔄 Sidebar refresh
with st.sidebar:
    st.markdown("### 🔄 Refresh Models")
    if st.button("Run Training Pipeline"):
        with st.spinner("Running pipeline..."):
            run_model_pipeline_and_save_metrics(
                output_dir="outputs",
                data_path="data/processed/train.csv",
                config_path="config/model_configs.json"
            )
        st.success("✅ Pipeline complete. Models refreshed.")

    st.markdown(f"🕒 Last refresh: `{get_latest_model_timestamp()}`")

# 📊 Main dashboard
st.title("📈 Model Comparison Dashboard")



models = load_models()
#models = run_all_models(
#    config_path="sample_config.yaml",
 #   data_path="data/processed/train.csv"
#)

#models = load_models()
best_model = get_best_model(models, metric="R2")
if best_model:
    with st.expander("🏆 Best Model Summary"):
        st.markdown(f"### 🏆 Best Model: `{best_model['name']}`")
        st.write({
            "R²": best_model["metrics"]["R2"],
            "MSE": best_model["metrics"]["MSE"],
            "MAE": best_model["metrics"]["MAE"]
        })
        #export_json_button(best_model["metrics"], "📥 Download Metrics", f"metrics_{best_model['name']}.json")
        #export_json_button(best_model["config"], "📥 Download Config", f"config_{best_model['name']}.json")

#model_names = [m["name"] for m in models]
model_names = [
    f"{r['name']} ❌" if not r.get("metrics") else r["name"]
    for r in models
]

selected_name = st.selectbox("🎯 Select Model", model_names)
selected_model = next(m for m in models if m["name"] == selected_name)


def log_failed_config(config, error_msg, log_path="logs/failed_configs.jsonl"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "error": error_msg
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

if not models:
    st.warning("No models found. Please run the pipeline.")
else:
    #tabs = st.tabs([model["name"] for model in models])
    tabs_list = ["📈 Performance", "🔍 PCA Variance", "🔥 Feature Importance", "Model Accuracy",
                 "📜 Logs", "📂 Raw Dataset", "Visualization",  "❌ Failed Models" ]
    tab = st.tabs(tabs_list )

    with tab[0]:
        st.markdown(f"### ⚙️ Config")
        st.json(selected_model["config"])

        st.markdown("### 📊 Metrics")
        st.write({
            "MSE": selected_model["metrics"]["MSE"],
            "MAE": selected_model["metrics"]["MAE"],
            "R²": selected_model["metrics"]["R2"],
            "Timestamp": selected_model["metrics"]["timestamp"]
        })

        export_json_button(selected_model["metrics"], "📥 Download Metrics", f"metrics_{selected_name}.json")
        export_json_button(selected_model["config"], "📥 Download Config", f"config_{selected_name}.json")
    with tab[1]:
        st.markdown("### 📈 PCA Variance")
        plot_pca_variance(selected_model["metrics"].get("pca_variance"))

    with tab[2]:
        st.markdown("### 🔥 Feature Importance")
        plot_feature_importance(selected_model["metrics"].get("feature_importance"))


    with tab[3]:
        st.markdown("### 🔍 Predictions vs Actuals")
        plot_predictions_vs_actuals(
            selected_model["metrics"].get("predictions"),
            selected_model["metrics"].get("actuals")
        )
    # Pipeline Logs
    with tab[4]:
        st.markdown("### 🪵 Logs")
        st.code("\n".join(selected_model.get("logs", [])))

    # 📂 Raw Dataset Tab
    with tab[5]:
        st.subheader("📂 View Raw Dataset")
        try:
            df = pd.read_csv("data/processed/train.csv")
            st.write("🔹 Raw Dataset")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")
    with tab[6]:

        st.markdown("### 📊 Explore Variables")
        numeric_cols = [None] + df.select_dtypes(include="number").columns.tolist()
        selected_col = st.selectbox("Select a vertical (y) variable", numeric_cols)
        selected_row = st.selectbox("Select horizontal (x) variable", numeric_cols)

        if selected_col and selected_row:
            fig = sns.relplot(x=selected_row, y=selected_col, data=df)
            st.pyplot(fig)
        elif selected_col:
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
        elif selected_row:
            fig, ax = plt.subplots()
            sns.histplot(df[selected_row], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_row}")
            st.pyplot(fig)
    with tab[7]:
        st.markdown("### ❌ Failed Model Configs")
        try:
            with open("logs/failed_configs.jsonl", "r") as f:
                failed_entries = [json.loads(line) for line in f]
            for entry in failed_entries:
                st.markdown(f"**Timestamp:** {entry['timestamp']}")
                st.json(entry["config"])
                st.error(f"Error: {entry['error']}")
                st.markdown("---")
        except FileNotFoundError:
            st.info("No failed configs logged yet.")



