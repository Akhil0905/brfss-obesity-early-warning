"""
src/models/train_regression.py
--------------------------------
Regression model training for the BRFSS Obesity Early Warning project.

Trains multiple regression models to predict obesity prevalence (Data_Value):
  - Linear Regression (baseline)
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor (optional)

All models are trained on the temporal training split (year ≤ train_max_year)
and serialized to models/regression/. Hyperparameters are read from config.yaml.

Split strategy: temporal (train on earlier years, validate on later years).
See config.yaml → split and PROJECT_PLAN.md §7 for rationale.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import get_logger, load_config
from src.utils.paths import model_path

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model registry — maps config name → sklearn estimator factory
# ---------------------------------------------------------------------------

def _build_regression_model(name: str, params: dict, random_state: int) -> Any:
    """Instantiate a regression estimator from its config name and parameters.

    Args:
        name: Model name from config (e.g., 'random_forest').
        params: Hyperparameter dict from config.
        random_state: Global random seed.

    Returns:
        Fitted sklearn-compatible estimator (wrapped in Pipeline for linear models).
    """
    if name == "linear_regression":
        # Scale features for linear models
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    elif name == "ridge":
        alpha = params.get("alpha", 1.0)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])

    elif name == "lasso":
        alpha = params.get("alpha", 0.1)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=5000)),
        ])

    elif name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            random_state=random_state,
            n_jobs=-1,
        )

    elif name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            random_state=random_state,
        )

    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                random_state=random_state,
                verbosity=0,
            )
        except ImportError:
            logger.warning("xgboost not installed — skipping XGBRegressor.")
            return None

    else:
        raise ValueError(f"Unknown regression model name: '{name}'")


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Optional[dict] = None,
    feature_names: Optional[List[str]] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """Train all enabled regression models on the training set.

    Args:
        X_train: Feature matrix (training split).
        y_train: Target vector (Data_Value, training split).
        config: Pre-loaded config. If None, loaded from config.yaml.
        feature_names: List of feature column names (for logging).
        silent: If True, suppress logging and model saving (useful for CV).

    Returns:
        Dictionary mapping model name → fitted estimator.
    """
    if config is None:
        config = load_config()

    random_state = config["split"]["random_state"]
    model_configs = config["regression"]["models"]
    save_models = config["output"]["save_models"] and not silent

    if not silent:
        logger.info(
            f"Training regression models | "
            f"X_train={X_train.shape} | target rows with value={y_train.notna().sum():,}"
        )

    # Drop rows where target is missing (should already be clean, but guard)
    valid_mask = y_train.notna()
    X_fit = X_train[valid_mask]
    y_fit = y_train[valid_mask]

    trained_models: Dict[str, Any] = {}

    for model_cfg in model_configs:
        name = model_cfg["name"]
        enabled = model_cfg.get("enabled", True)

        if not enabled:
            if not silent:
                logger.info(f"  Skipping '{name}' (disabled in config)")
            continue

        if not silent:
            logger.info(f"  Training: {name}")
        params = {k: v for k, v in model_cfg.items() if k not in ("name", "enabled")}

        estimator = _build_regression_model(name, params, random_state)
        if estimator is None:
            continue  # skipped (e.g., xgboost not installed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_fit, y_fit)

        trained_models[name] = estimator
        if not silent:
            logger.info(f"    → '{name}' trained on {len(X_fit):,} samples")

        if save_models:
            save_path = model_path("regression", f"{name}.joblib")
            joblib.dump(estimator, save_path)
            if not silent:
                logger.info(f"    → Saved to {save_path}")

    if not silent:
        logger.info(f"Regression training complete | {len(trained_models)} model(s) trained")
    return trained_models
    return trained_models
