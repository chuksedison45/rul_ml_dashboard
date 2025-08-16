# 🧠 Modular ML Dashboard

A reproducible, interactive dashboard for comparing machine learning models with PCA, feature importance, logging, and exportable metrics/configs.

# 🧠 Remaining Useful Life (RUL) Dashboard

This dashboard predicts the Remaining Useful Life of jet engines using the CMAPSS dataset — a NASA simulation of sensor readings under varied operating conditions.

Each engine starts healthy and degrades over time. The challenge? Predict how many cycles remain before failure using only sensor data.

## 🔍 Features

- PCA variance visualization
- Feature importance across models
- Model selection and comparison
- Exportable metrics and reproducible logs

Built with Streamlit, scikit-learn, and XGBoost — designed for transparency, onboarding clarity, and production-grade engineering.

🎯 [Launch the dashboard](https://edison-ai-rul-dashboard.streamlit.app)

## 🚀 Features

- 📂 Raw dataset visualization (distributions, correlations)
- 🎯 Model selection dropdown
- 🏆 Best model summary tab (based on R²)
- 📊 Metrics, config, and feature importance plots
- 📈 PCA variance and prediction analysis
- 📥 Export buttons for reproducibility
- 🪵 Logs for transparency and debugging

## 🛠️ Setup

```bash
git clone https://github.com/your-username/ml-dashboard.git
cd ml-dashboard
pip install -r requirements.txt
streamlit run dashboard.py