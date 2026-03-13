"""
src/models/cross_validation.py
-------------------------------
Implementation of cross-validation strategies for model robustness.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Any
from src.utils.helpers import get_logger, print_section
from src.models.train_regression import train_regression_models
from src.models.train_classification import train_classification_models
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, roc_auc_score

logger = get_logger(__name__)

def run_cross_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    groups_col: str,
    track: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run GroupKFold cross-validation and return aggregated metrics.
    
    Args:
        df: Full modeling dataframe
        feature_cols: List of feature names
        target_col: Name of the target column
        groups_col: Column used for grouping (e.g., 'LocationDesc')
        track: 'regression' or 'classification'
        config: Configuration dictionary
    """
    n_splits = config.get("validation", {}).get("cv_folds", 5)
    gkf = GroupKFold(n_splits=n_splits)
    
    # Drop rows where target is missing (can happen for early_warning)
    valid_mask = df[target_col].notna()
    df_cv = df[valid_mask].copy()
    
    X = df_cv[feature_cols]
    y = df_cv[target_col]
    groups = df_cv[groups_col]
    
    fold_results = []
    
    print_section(f"Running {n_splits}-Fold Group CV ({track})")
    logger.info(f"Grouping by: {groups_col} | Rows after target NaN drop: {len(y)}")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        logger.info(f"Fold {fold}: Train size={len(X_train)}, Test size={len(X_test)}")
        
        if track == "regression":
            models = train_regression_models(X_train, y_train, config=config, silent=True)
            for name, model in models.items():
                y_pred = model.predict(X_test)
                fold_results.append({
                    "fold": fold,
                    "model": name,
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred)
                })
        else:
            models = train_classification_models(X_train, y_train, classification_target=target_col, config=config, silent=True)
            for name, model in models.items():
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                fold_results.append({
                    "fold": fold,
                    "model": name,
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                    "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
                })
                
    cv_df = pd.DataFrame(fold_results)
    summary = cv_df.groupby("model").mean().drop(columns="fold").to_dict(orient="index")
    
    return summary
