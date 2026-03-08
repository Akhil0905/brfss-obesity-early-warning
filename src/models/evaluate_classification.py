"""
src/models/evaluate_classification.py
----------------------------------------
Classification model evaluation for the BRFSS Obesity Early Warning project.

Computes and saves:
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion matrix (per model, saved as individual figures)
  - ROC curve comparison plot across all models

All metrics are saved to reports/metrics/classification_metrics.json.
Plots are saved to reports/figures/.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.helpers import get_logger, load_config, save_metrics
from src.utils.paths import figures_path, metrics_path

logger = get_logger(__name__)


def evaluate_classification_models(
    trained_models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    classification_target: str = "high_risk_obesity",
    config: Optional[dict] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all trained classifiers.

    Args:
        trained_models: Dict of model_name → fitted classifier.
        X_val: Validation feature matrix.
        y_val: Validation binary targets.
        X_test: Test feature matrix (optional).
        y_test: Test binary targets (optional).
        classification_target: Name of the classification target (for labels).
        config: Pre-loaded config. If None, loaded from config.yaml.

    Returns:
        Nested dict: model_name → metric_name → value.
    """
    if config is None:
        config = load_config()

    save_fig = config["output"]["save_figures"]
    save_met = config["output"]["save_metrics"]
    dpi = config["output"].get("figure_dpi", 150)

    all_metrics: Dict[str, Dict[str, float]] = {}
    roc_data: Dict[str, tuple] = {}  # for the combined ROC plot

    logger.info("=" * 60)
    logger.info("CLASSIFICATION EVALUATION")
    logger.info(f"Target: {classification_target}")
    logger.info("=" * 60)

    valid_val = y_val.notna()
    X_val_e = X_val[valid_val]
    y_val_e = y_val[valid_val].astype(int)

    for name, model in trained_models.items():
        metrics: Dict[str, float] = {}

        # Predictions
        y_pred = model.predict(X_val_e)
        y_proba = _get_proba(model, X_val_e)

        # Standard metrics
        metrics["val_accuracy"] = float(accuracy_score(y_val_e, y_pred))
        metrics["val_precision"] = float(precision_score(y_val_e, y_pred, zero_division=0))
        metrics["val_recall"] = float(recall_score(y_val_e, y_pred, zero_division=0))
        metrics["val_f1"] = float(f1_score(y_val_e, y_pred, zero_division=0))

        if y_proba is not None:
            auc = roc_auc_score(y_val_e, y_proba)
            metrics["val_roc_auc"] = float(auc)
            fpr, tpr, _ = roc_curve(y_val_e, y_proba)
            roc_data[name] = (fpr, tpr, auc)

        # Test metrics if provided
        if X_test is not None and y_test is not None:
            valid_test = y_test.notna()
            X_test_e = X_test[valid_test]
            y_test_e = y_test[valid_test].astype(int)

            y_pred_test = model.predict(X_test_e)
            y_proba_test = _get_proba(model, X_test_e)

            metrics["test_accuracy"] = float(accuracy_score(y_test_e, y_pred_test))
            metrics["test_precision"] = float(precision_score(y_test_e, y_pred_test, zero_division=0))
            metrics["test_recall"] = float(recall_score(y_test_e, y_pred_test, zero_division=0))
            metrics["test_f1"] = float(f1_score(y_test_e, y_pred_test, zero_division=0))
            if y_proba_test is not None:
                metrics["test_roc_auc"] = float(roc_auc_score(y_test_e, y_proba_test))

        all_metrics[name] = metrics
        _log_classification_metrics(name, metrics)

        if save_fig:
            _plot_confusion_matrix(model, X_val_e, y_val_e, name, dpi)

    _print_classification_summary(all_metrics)

    if save_fig and roc_data:
        _plot_roc_curves(roc_data, classification_target, dpi)

    if save_met:
        met_path = metrics_path("classification_metrics.json")
        save_metrics(all_metrics, met_path)

    return all_metrics


def _get_proba(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return predicted probabilities for the positive class if available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def _log_classification_metrics(name: str, metrics: Dict[str, float]) -> None:
    logger.info(f"\n  Model: {name}")
    logger.info(
        f"    Val  → Acc={metrics['val_accuracy']:.3f} | "
        f"P={metrics['val_precision']:.3f} | R={metrics['val_recall']:.3f} | "
        f"F1={metrics['val_f1']:.3f} | "
        f"AUC={metrics.get('val_roc_auc', float('nan')):.3f}"
    )
    if "test_accuracy" in metrics:
        logger.info(
            f"    Test → Acc={metrics['test_accuracy']:.3f} | "
            f"P={metrics['test_precision']:.3f} | R={metrics['test_recall']:.3f} | "
            f"F1={metrics['test_f1']:.3f} | "
            f"AUC={metrics.get('test_roc_auc', float('nan')):.3f}"
        )


def _print_classification_summary(all_metrics: Dict[str, Dict[str, float]]) -> None:
    rows = [{"model": name, **met} for name, met in all_metrics.items()]
    df = pd.DataFrame(rows).set_index("model")
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 70)
    print(df.round(4).to_string())
    print("=" * 70 + "\n")


def _plot_confusion_matrix(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    dpi: int,
) -> None:
    """Save a confusion matrix plot for a single model."""
    cm = confusion_matrix(y, model.predict(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}", fontsize=11)
    plt.tight_layout()
    fig_path = figures_path(f"confusion_matrix_{model_name}.png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {fig_path}")


def _plot_roc_curves(
    roc_data: Dict[str, tuple],
    target_name: str,
    dpi: int,
) -> None:
    """Save a combined ROC curve plot for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})", color=color, linewidth=1.8)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves — {target_name.replace('_', ' ').title()} (Validation)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = figures_path(f"roc_curves_{target_name}.png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves: {fig_path}")
