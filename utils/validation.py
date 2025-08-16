# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 8/16/2025
File: validation.py
PRODUCT: PyCharm
PROJECT: rul_dashboard
"""

def validate_model_config(cfg):
    required_keys = {"name", "model", "params"}
    if not isinstance(cfg, dict):
        return False
    if not required_keys.issubset(cfg.keys()):
        return False
    if not isinstance(cfg["params"], dict):
        return False
    return True
