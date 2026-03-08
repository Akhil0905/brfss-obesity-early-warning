"""
src/models/train_classification.py
-------------------------------------
Classification model training for the BRFSS Obesity Early Warning project.

Trains binary classifiers to predict:
  - high_risk_obesity: whether an observation is above the prevalence threshold
  - early_warning (if feasible): whether a non-high-risk observation will
    become high-risk in the next observed year

Models trained:
  - Logistic Regression (baseline with class_weight='balanced')
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier (optional)

Split strategy: temporal (same boundaries as regression).
Class imbalance is handled via class_weight='balanced' for linear models
and noted in evaluation for tree-based models.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import get_logger, load_config
from src.utils.paths import model_path

logger = get_logger(__name__)


def _build_classification_model(name: str, params: dict, random_state: int) -> Any:
    """Instantiate a classifier from its config name and params.

    Args:
        name: Model name from config.
        params: Hyperparameter dict from config.
        random_state: Global random seed.

    Returns:
        sklearn-compatible classifier.
    """
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                class_weight=params.get("class_weight", "balanced"),
                random_state=random_state,
                solver="lbfgs",
            )),
        ])

    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            class_weight=params.get("class_weight", "balanced"),
            random_state=random_state,
            n_jobs=-1,
        )

    elif name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            random_state=random_state,
        )

    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                scale_pos_weight=params.get("scale_pos_weight", 1),
                random_state=random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        except ImportError:
            logger.warning("xgboost not installed — skipping XGBClassifier.")
            return None

    else:
        raise ValueError(f"Unknown classification model name: '{name}'")


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    classification_target: str,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Train all enabled classification models.

    Args:
        X_train: Feature matrix (training split).
        y_train: Binary target vector (training split).
        classification_target: Name of the target column (for logging).
        config: Pre-loaded config. If None, loaded from config.yaml.

    Returns:
        Dictionary mapping model name → fitted classifer.
    """
    if config is None:
        config = load_config()

    random_state = config["split"]["random_state"]
    model_configs = config["classification"]["models"]
    save_models = config["output"]["save_models"]

    # Drop rows where target is NaN (can happen for early_warning)
    valid_mask = y_train.notna()
    X_fit = X_train[valid_mask]
    y_fit = y_train[valid_mask].astype(int)

    n_pos = int(y_fit.sum())
    n_neg = int((y_fit == 0).sum())
    logger.info(
        f"Training classification models | target='{classification_target}' | "
        f"X_train={X_fit.shape} | positive={n_pos:,} | negative={n_neg:,} | "
        f"imbalance ratio={n_neg / max(n_pos, 1):.1f}:1"
    )

    trained_models: Dict[str, Any] = {}

    for model_cfg in model_configs:
        name = model_cfg["name"]
        enabled = model_cfg.get("enabled", True)

        if not enabled:
            logger.info(f"  Skipping '{name}' (disabled in config)")
            continue

        logger.info(f"  Training: {name}")
        params = {k: v for k, v in model_cfg.items() if k not in ("name", "enabled")}

        estimator = _build_classification_model(name, params, random_state)
        if estimator is None:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_fit, y_fit)

        trained_models[name] = estimator
        logger.info(f"    → '{name}' trained on {len(X_fit):,} samples")

        if save_models:
            save_path = model_path("classification", f"{name}.joblib")
            joblib.dump(estimator, save_path)
            logger.info(f"    → Saved to {save_path}")

    logger.info(f"Classification training complete | {len(trained_models)} model(s) trained")
    return trained_models
