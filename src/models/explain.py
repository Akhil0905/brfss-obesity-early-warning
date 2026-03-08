"""
src/models/explain.py
----------------------
Model interpretability for the BRFSS Obesity Early Warning project.

Provides three complementary interpretability methods:
  1. **Feature importance** — for tree-based models (Random Forest, GBM, XGB)
  2. **Coefficient analysis** — for linear models (Linear/Ridge/Lasso Regression,
     Logistic Regression)
  3. **Permutation importance** — model-agnostic, measures loss increase when
     each feature is randomly permuted; works for all model types.
  4. **SHAP** — optional (disabled by default in config); requires `shap` package.

All plots are saved to reports/figures/ and can be embedded in notebooks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.utils.helpers import get_logger, load_config, save_metrics
from src.utils.paths import figures_path, metrics_path

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_models(
    trained_models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    track: str = "regression",
    config: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """Run interpretability analysis on all trained models.

    Args:
        trained_models: Dict of model_name → fitted estimator.
        X_val: Validation feature matrix (used for permutation importance).
        y_val: Validation targets.
        feature_names: List of feature column names (same order as X columns).
        track: 'regression' or 'classification' (affects labels and plot names).
        config: Pre-loaded config. If None, loaded from config.yaml.

    Returns:
        Dict of model_name → DataFrame of feature importances (descending order).
    """
    if config is None:
        config = load_config()

    interp_cfg = config.get("interpretability", {})
    top_n = interp_cfg.get("top_n_features", 20)
    run_perm = interp_cfg.get("run_permutation_importance", True)
    perm_repeats = interp_cfg.get("permutation_n_repeats", 10)
    run_shap = interp_cfg.get("run_shap", False)
    random_state = config["split"]["random_state"]
    dpi = config["output"].get("figure_dpi", 150)

    all_importances: Dict[str, pd.DataFrame] = {}

    for name, model in trained_models.items():
        logger.info(f"\nExplaining model: {name}")

        imp_df = None

        # 1. Built-in feature importance (tree-based models)
        if _has_feature_importances(model):
            imp_df = _get_tree_importance(model, feature_names, name, top_n, track, dpi)

        # 2. Coefficient analysis (linear models)
        elif _has_coef(model):
            imp_df = _get_linear_coef(model, feature_names, name, top_n, track, dpi)

        # 3. Permutation importance (always run if enabled)
        if run_perm:
            valid_mask = y_val.notna()
            perm_df = _get_permutation_importance(
                model=model,
                X=X_val[valid_mask],
                y=y_val[valid_mask],
                feature_names=feature_names,
                model_name=name,
                n_repeats=perm_repeats,
                top_n=top_n,
                track=track,
                dpi=dpi,
                random_state=random_state,
            )
            # Use permutation importance as primary if no built-in is available
            if imp_df is None:
                imp_df = perm_df

        if imp_df is not None:
            all_importances[name] = imp_df

        # 4. SHAP (optional)
        if run_shap:
            _run_shap(model, X_val, feature_names, name, config)

    # Save a consolidated importances JSON
    if config["output"]["save_metrics"]:
        importances_json = {
            name: df.set_index("feature")["importance"].to_dict()
            for name, df in all_importances.items()
        }
        save_metrics(importances_json, metrics_path(f"feature_importances_{track}.json"))

    return all_importances


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_feature_importances(model: Any) -> bool:
    """Return True if the model (or its final Pipeline step) has feature_importances_."""
    estimator = _unwrap(model)
    return hasattr(estimator, "feature_importances_")


def _has_coef(model: Any) -> bool:
    """Return True if the model (or its final Pipeline step) has coef_."""
    estimator = _unwrap(model)
    return hasattr(estimator, "coef_")


def _unwrap(model: Any) -> Any:
    """Extract the final estimator from a Pipeline, or return the model itself."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _get_tree_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    top_n: int,
    track: str,
    dpi: int,
) -> pd.DataFrame:
    """Extract and plot built-in feature importances from a tree-based model."""
    estimator = _unwrap(model)
    importances = estimator.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(
        imp_df["feature"][::-1],
        imp_df["importance"][::-1],
        color="#4C72B0",
        edgecolor="white",
    )
    ax.set_xlabel("Feature Importance (Gini / Variance Reduction)", fontsize=10)
    ax.set_title(
        f"Top {top_n} Feature Importances — {model_name.replace('_', ' ').title()} ({track.title()})",
        fontsize=11, fontweight="bold"
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig_path = figures_path(f"feature_importance_{track}_{model_name}.png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved feature importance plot: {fig_path}")

    return imp_df


def _get_linear_coef(
    model: Any,
    feature_names: List[str],
    model_name: str,
    top_n: int,
    track: str,
    dpi: int,
) -> pd.DataFrame:
    """Extract and plot coefficients from a linear model."""
    estimator = _unwrap(model)
    coef = estimator.coef_

    if coef.ndim > 1:
        coef = coef[0]  # binary classifier: take positive class coefficients

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": coef,
    }).assign(abs_coef=lambda x: x["importance"].abs())
    imp_df = imp_df.sort_values("abs_coef", ascending=False).head(top_n).reset_index(drop=True)

    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in imp_df["importance"]]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(
        imp_df["feature"][::-1],
        imp_df["importance"][::-1],
        color=colors[::-1],
        edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient Value", fontsize=10)
    ax.set_title(
        f"Top {top_n} Coefficients — {model_name.replace('_', ' ').title()} ({track.title()})",
        fontsize=11, fontweight="bold"
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig_path = figures_path(f"coefficients_{track}_{model_name}.png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved coefficient plot: {fig_path}")

    return imp_df.drop(columns=["abs_coef"])


def _get_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    model_name: str,
    n_repeats: int,
    top_n: int,
    track: str,
    dpi: int,
    random_state: int,
) -> pd.DataFrame:
    """Compute and plot permutation importance (model-agnostic)."""
    scoring = "r2" if track == "regression" else "f1"

    y_clean = y.astype(int) if track == "classification" else y

    result = permutation_importance(
        model, X, y_clean,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(
        perm_df["feature"][::-1],
        perm_df["importance"][::-1],
        xerr=perm_df["std"][::-1],
        color="#55A868",
        edgecolor="white",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Permutation Importance (Δ {scoring.upper()})", fontsize=10)
    ax.set_title(
        f"Permutation Importance — {model_name.replace('_', ' ').title()} ({track.title()})",
        fontsize=11, fontweight="bold"
    )
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig_path = figures_path(f"permutation_importance_{track}_{model_name}.png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved permutation importance plot: {fig_path}")

    return perm_df


def _run_shap(
    model: Any,
    X_val: pd.DataFrame,
    feature_names: List[str],
    model_name: str,
    config: dict,
) -> None:
    """Run SHAP analysis (optional — requires the shap package)."""
    dpi = config["output"].get("figure_dpi", 150)
    try:
        import shap
    except ImportError:
        logger.warning("SHAP requested but 'shap' package is not installed. Skipping.")
        return

    logger.info(f"  Running SHAP for {model_name}...")
    estimator = _unwrap(model)

    try:
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
        else:
            explainer = shap.LinearExplainer(estimator, X_val)

        shap_values = explainer.shap_values(X_val)

        # For binary classifiers, shap_values may be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val, feature_names=feature_names, show=False)
        fig_path = figures_path(f"shap_summary_{model_name}.png")
        plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved SHAP summary plot: {fig_path}")

    except Exception as exc:
        logger.warning(f"  SHAP failed for {model_name}: {exc}")
