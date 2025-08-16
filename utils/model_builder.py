# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/15/2025
File: model_builder.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

def build_model(model_type, params):
    if model_type == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_type == "GradientBoosting":
        return GradientBoostingRegressor(**params)
    elif model_type == "Ridge":
        return Ridge(**params)
    elif model_type == "SVR":
        return SVR(**params)
    elif model_type == "LinearRegression":
        return LinearRegression()
    elif model_type == "Xgboost":
        return XGBRegressor( n_estimators= 100, learning_rate= 0.1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

