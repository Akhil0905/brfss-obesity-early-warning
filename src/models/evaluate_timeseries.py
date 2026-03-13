"""
src/models/evaluate_timeseries.py
-----------------------------------
Evaluation metrics and visualization for LSTM models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Dict, Any, List
from src.utils.helpers import get_logger, print_section
from src.utils.paths import figures_path, metrics_path
import json

logger = get_logger(__name__)

def evaluate_lstm(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Evaluate LSTM model and save plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().squeeze()
        
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }
    
    logger.info(f"LSTM Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    if config["output"]["save_metrics"]:
        with open(metrics_path("lstm_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
    if config["output"]["save_figures"]:
        plot_lstm_results(y_test, y_pred)
        
    return metrics

def plot_lstm_results(y_true: np.ndarray, y_pred: np.ndarray):
    """Plot actual vs predicted values for LSTM."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Obesity Prevalence (%)")
    plt.ylabel("Predicted Obesity Prevalence (%)")
    plt.title("LSTM Sequence Prediction: Actual vs Predicted")
    plt.grid(True)
    
    save_path = figures_path("lstm_actual_vs_pred.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved LSTM plot to {save_path}")
