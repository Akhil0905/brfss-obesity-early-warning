"""
src/models/evaluate_regression.py
-----------------------------------
Regression model evaluation for the BRFSS Obesity Early Warning project.

Computes MAE, RMSE, and R² for each trained regression model on the
validation and test sets. Results are saved to reports/metrics/ as JSON
and a comparison plot is saved to reports/figures/.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.helpers import get_logger, load_config, save_metrics
from src.utils.paths import figures_path, metrics_path

logger = get_logger(__name__)


def evaluate_regression_models(
    trained_models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    config: Optional[dict] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all trained regression models on validation (and optionally test) data.

    Args:
        trained_models: Dict of model_name → fitted estimator.
        X_val: Validation feature matrix.
        y_val: Validation targets.
        X_test: Test feature matrix (optional).
        y_test: Test targets (optional).
        config: Pre-loaded config. If None, loaded from config.yaml.

    Returns:
        Nested dict: model_name → {'val_mae': ..., 'val_rmse': ..., 'val_r2': ...}
    """
    if config is None:
        config = load_config()

    save_fig = config["output"]["save_figures"]
    save_met = config["output"]["save_metrics"]

    all_metrics: Dict[str, Dict[str, float]] = {}

    logger.info("=" * 60)
    logger.info("REGRESSION EVALUATION")
    logger.info("=" * 60)

    valid_val_mask = y_val.notna()
    X_val_e = X_val[valid_val_mask]
    y_val_e = y_val[valid_val_mask]

    for name, model in trained_models.items():
        metrics: Dict[str, float] = {}

        # Validation metrics
        y_pred_val = model.predict(X_val_e)
        metrics["val_mae"] = float(mean_absolute_error(y_val_e, y_pred_val))
        metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val_e, y_pred_val)))
        metrics["val_r2"] = float(r2_score(y_val_e, y_pred_val))

        # Test metrics (if provided)
        if X_test is not None and y_test is not None:
            valid_test_mask = y_test.notna()
            y_pred_test = model.predict(X_test[valid_test_mask])
            metrics["test_mae"] = float(mean_absolute_error(y_test[valid_test_mask], y_pred_test))
            metrics["test_rmse"] = float(np.sqrt(mean_squared_error(y_test[valid_test_mask], y_pred_test)))
            metrics["test_r2"] = float(r2_score(y_test[valid_test_mask], y_pred_test))

        all_metrics[name] = metrics
        _log_regression_metrics(name, metrics)

    # Summary table to console
    _print_regression_summary(all_metrics)

    if save_met:
        met_path = metrics_path("regression_metrics.json")
        save_metrics(all_metrics, met_path)

    if save_fig:
        _plot_regression_comparison(all_metrics, config)

    return all_metrics


def _log_regression_metrics(name: str, metrics: Dict[str, float]) -> None:
    """Log metrics for a single model."""
    logger.info(f"\n  Model: {name}")
    logger.info(f"    Val  → MAE={metrics['val_mae']:.3f} | RMSE={metrics['val_rmse']:.3f} | R²={metrics['val_r2']:.4f}")
    if "test_mae" in metrics:
        logger.info(f"    Test → MAE={metrics['test_mae']:.3f} | RMSE={metrics['test_rmse']:.3f} | R²={metrics['test_r2']:.4f}")


def _print_regression_summary(all_metrics: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted summary table."""
    rows = []
    for name, metrics in all_metrics.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("model")
    print("\n" + "=" * 70)
    print("REGRESSION RESULTS SUMMARY")
    print("=" * 70)
    print(summary_df.round(4).to_string())
    print("=" * 70 + "\n")


def _plot_regression_comparison(
    all_metrics: Dict[str, Dict[str, float]],
    config: dict,
) -> None:
    """Save a bar chart comparing validation MAE / RMSE / R² across models."""
    models = list(all_metrics.keys())
    val_mae = [all_metrics[m].get("val_mae", 0) for m in models]
    val_rmse = [all_metrics[m].get("val_rmse", 0) for m in models]
    val_r2 = [all_metrics[m].get("val_r2", 0) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Regression Model Comparison (Validation Set)", fontsize=13, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"][:len(models)]

    def _bar_plot(ax, values, title, ylabel, is_r2=False):
        bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8, rotation=15)
        ax.tick_params(axis="x", length=0)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.002 if not is_r2 else 0.005),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    _bar_plot(axes[0], val_mae, "MAE (lower is better)", "MAE")
    _bar_plot(axes[1], val_rmse, "RMSE (lower is better)", "RMSE")
    _bar_plot(axes[2], val_r2, "R² (higher is better)", "R²", is_r2=True)

    plt.tight_layout()
    fig_path = figures_path("regression_model_comparison.png")
    dpi = config["output"].get("figure_dpi", 150)
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved regression comparison plot to {fig_path}")
