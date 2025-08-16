# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/16/2025
File: file_io.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_real_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at: {path}")

    df = pd.read_csv(path)
    if "RUL" not in df.columns:
        raise ValueError("❌ 'RUL' column missing in dataset.")

    X = df.drop(columns=["RUL"])
    y = df["RUL"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


