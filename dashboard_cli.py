# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/16/2025
File: dashboard_cli.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

# dashboard_cli.py
import argparse
import subprocess
import sys
from utils.pipeline import run_model_pipeline_and_save_metrics, load_model_configs, validate_output_files

def refresh_pipeline():
    print("🔄 Refreshing models via pipeline...")
    run_model_pipeline_and_save_metrics(
        output_dir="outputs",
        data_path="data/processed/train.csv",
        config_path="config/model_configs.json"
    )
    print("✅ Model refresh complete.")

def serve_dashboard():
    print("🚀 Launching Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_refresh.py"])

from dashboard_utils import validate_model_outputs
from utils.pipeline import load_model_configs

def test_pipeline_outputs():
    print("🧪 Validating model configs and output files...")
    configs = load_model_configs("config/model_configs.json")
    missing = validate_model_outputs(configs, models_dir="models")
    if missing:
        print(f"⚠️ Missing model files: {missing}")
    else:
        print("✅ All model files present.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashboard CLI")
    parser.add_argument("--refresh", action="store_true", help="Run training pipeline and refresh models")
    parser.add_argument("--serve", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--test", action="store_true", help="Validate configs and output files")

    args = parser.parse_args()

    if args.refresh:
        refresh_pipeline()
    if args.serve:
        serve_dashboard()
    if args.test:
        test_pipeline_outputs()
    if not any(vars(args).values()):
        parser.print_help()