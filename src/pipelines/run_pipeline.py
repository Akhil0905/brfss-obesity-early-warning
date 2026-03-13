"""
src/pipelines/run_pipeline.py
-------------------------------
End-to-end pipeline runner for the BRFSS Obesity Early Warning project.

This script orchestrates the full machine learning pipeline in sequence:
  1. Load & filter raw data
  2. Validate raw data
  3. Preprocess (clean, encode, add region)
  4. Build features (lag, rolling, trend)
  5. Build targets (regression, high-risk, early warning)
  6. Temporal train/val/test split
  7. Train + evaluate regression models
  8. Train + evaluate classification models
  9. Run interpretability analysis

Usage:
    python -m src.pipelines.run_pipeline
    python -m src.pipelines.run_pipeline --track regression
    python -m src.pipelines.run_pipeline --track classification
    python -m src.pipelines.run_pipeline --track both  (default)

All outputs are written to:
  - data/interim/    — cleaned data
  - data/processed/  — feature matrix + targets
  - models/          — serialized model files
  - reports/metrics/ — JSON metrics
  - reports/figures/ — PNG plots
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd

from src.data.load_data import filter_obesity_question, load_raw_data
from src.data.preprocess import preprocess, split_by_year
from src.data.validate_data import validate_raw
from src.features.build_features import build_features, get_feature_columns
from src.features.build_targets import build_targets, select_classification_target
from src.models.evaluate_classification import evaluate_classification_models
from src.models.evaluate_regression import evaluate_regression_models
from src.models.evaluate_timeseries import evaluate_lstm
from src.models.explain import explain_models
from src.models.train_classification import train_classification_models
from src.models.train_regression import train_regression_models
from src.models.train_timeseries import prepare_sequences, train_lstm
from src.models.cross_validation import run_cross_validation
from src.utils.helpers import get_logger, load_config, print_section, set_pandas_display
from src.utils.paths import ensure_dirs

logger = get_logger(__name__)


def run_pipeline(track: str = "both", cv: bool = False, config_override: Optional[dict] = None) -> None:
    """Execute the BRFSS Obesity Early Warning pipeline end-to-end.

    Args:
        track: Which modeling track(s) to run.
            - 'regression': regression models only
            - 'classification': classification models only
            - 'both': run both tracks (default)
        cv: Whether to run cross-validation in addition to temporal split.
        config_override: Optional dict to override loaded config values.
            Useful for testing with modified parameters without editing config.yaml.
    """
    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    set_pandas_display()
    ensure_dirs()

    config = load_config()
    if config_override:
        _deep_update(config, config_override)

    print_section("BRFSS OBESITY EARLY WARNING PIPELINE")
    logger.info(f"Pipeline track: {track}")

    # -----------------------------------------------------------------------
    # Stage 1: Load raw data
    # -----------------------------------------------------------------------
    print_section("Stage 1: Load & Filter")
    raw_df = load_raw_data(config=config)
    df = filter_obesity_question(raw_df, config=config)

    # -----------------------------------------------------------------------
    # Stage 2: Validate
    # -----------------------------------------------------------------------
    print_section("Stage 2: Validate")
    validate_raw(df, strict=False)

    # -----------------------------------------------------------------------
    # Stage 3: Preprocess
    # -----------------------------------------------------------------------
    print_section("Stage 3: Preprocess")
    clean_df = preprocess(df, config=config, save_interim=True)

    # -----------------------------------------------------------------------
    # Stage 4: Build features
    # -----------------------------------------------------------------------
    print_section("Stage 4: Build Features")
    feature_df = build_features(clean_df, config=config, save_output=True)

    # -----------------------------------------------------------------------
    # Stage 5: Build targets
    # -----------------------------------------------------------------------
    print_section("Stage 5: Build Targets")
    # Use year mask to define training rows for leakage-safe threshold computation
    train_max_year = config["split"]["train_max_year"]
    train_mask = clean_df["YearStart"] <= train_max_year

    target_df, target_metadata = build_targets(
        df=clean_df,
        train_mask=train_mask,
        config=config,
        save_output=True,
    )

    classification_target = select_classification_target(target_metadata, config=config)
    logger.info(f"Classification target selected: '{classification_target}'")

    # -----------------------------------------------------------------------
    # Stage 6: Merge features + targets, then split
    # -----------------------------------------------------------------------
    print_section("Stage 6: Temporal Split")

    # Merge features and targets on their shared key columns
    key_cols = ["YearStart", "LocationAbbr", "stratum_category", "stratum_value"]
    # We also need LocationDesc for GroupKFold if running CV
    group_cols = ["LocationDesc"]
    
    key_cols = [c for c in key_cols if c in feature_df.columns and c in target_df.columns]

    modeling_df = feature_df.merge(
        target_df[key_cols + group_cols + ["Data_Value", config["targets"]["high_risk_col"]]
                  + ([classification_target] if classification_target not in
                     [config["targets"]["high_risk_col"], "Data_Value"] else [])],
        on=key_cols,
        how="inner",
    ).drop_duplicates(subset=key_cols)

    logger.info(f"Modeling dataset shape: {modeling_df.shape}")

    train_df, val_df, test_df = split_by_year(modeling_df, config=config)

    # Determine feature columns (numeric only, excluding key cols)
    feature_cols = get_feature_columns(feature_df)
    # Remove target columns if they slipped in
    target_col_names = {"Data_Value", config["targets"]["high_risk_col"],
                        config["targets"]["early_warning_col"], classification_target}
    feature_cols = [c for c in feature_cols if c not in target_col_names and c in train_df.columns]

    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols] if len(test_df) > 0 else None

    # -----------------------------------------------------------------------
    # Track 1: Regression
    # -----------------------------------------------------------------------
    if track in ("regression", "both"):
        print_section("Track 1: Regression")

        y_train_reg = train_df["Data_Value"]
        y_val_reg = val_df["Data_Value"]
        y_test_reg = test_df["Data_Value"] if X_test is not None else None

        reg_models = train_regression_models(X_train, y_train_reg, config=config, feature_names=feature_cols)
        reg_metrics = evaluate_regression_models(
            reg_models, X_val, y_val_reg,
            X_test=X_test, y_test=y_test_reg,
            config=config,
        )

        if cv:
            cv_metrics = run_cross_validation(
                df=modeling_df,
                feature_cols=feature_cols,
                target_col="Data_Value",
                groups_col="LocationDesc",
                track="regression",
                config=config
            )
            logger.info(f"Cross-Validation Summary (Regression): {cv_metrics}")

        print_section("Track 1 Interpretability: Regression")
        explain_models(
            reg_models, X_val, y_val_reg,
            feature_names=feature_cols,
            track="regression",
            config=config,
        )

    # -----------------------------------------------------------------------
    # Track 2: Classification
    # -----------------------------------------------------------------------
    if track in ("classification", "both"):
        print_section("Track 2: Classification")

        if classification_target not in train_df.columns:
            logger.error(
                f"Classification target '{classification_target}' not found in training data. "
                f"Skipping classification track."
            )
        else:
            y_train_clf = train_df[classification_target]
            y_val_clf = val_df[classification_target]
            y_test_clf = test_df[classification_target] if X_test is not None else None

            clf_models = train_classification_models(
                X_train, y_train_clf,
                classification_target=classification_target,
                config=config,
            )
            clf_metrics = evaluate_classification_models(
                clf_models, X_val, y_val_clf,
                X_test=X_test, y_test=y_test_clf,
                classification_target=classification_target,
                config=config,
            )

            if cv:
                cv_metrics = run_cross_validation(
                    df=modeling_df,
                    feature_cols=feature_cols,
                    target_col=classification_target,
                    groups_col="LocationDesc",
                    track="classification",
                    config=config
                )
                logger.info(f"Cross-Validation Summary (Classification): {cv_metrics}")

            print_section("Track 2 Interpretability: Classification")
            explain_models(
                clf_models, X_val, y_val_clf,
                feature_names=feature_cols,
                track="classification",
                config=config,
            )

    # -----------------------------------------------------------------------
    # Stage 7: Time Series (Deep Learning) Track
    # -----------------------------------------------------------------------
    if config.get("timeseries", {}).get("enabled", False) and track in ("regression", "both"):
        print_section("Stage 7: Time Series (LSTM)")
        seq_length = config["timeseries"]["seq_length"]
        
        # Prepare sequences using the full modeling_df to preserve temporal flow
        X_seq_all, y_seq_all, years_seq = prepare_sequences(
            modeling_df, feature_cols, "Data_Value", seq_length=seq_length, return_years=True
        )
        
        # Split sequences based on the year of the prediction (year of the target)
        train_mask = years_seq <= config["split"]["train_max_year"]
        val_mask = (years_seq > config["split"]["train_max_year"]) & (years_seq <= config["split"]["val_max_year"])
        test_mask = years_seq > config["split"]["val_max_year"]
        
        X_seq_train, y_seq_train = X_seq_all[train_mask], y_seq_all[train_mask]
        X_seq_val, y_seq_val = X_seq_all[val_mask], y_seq_all[val_mask]
        X_seq_test, y_seq_test = X_seq_all[test_mask], y_seq_all[test_mask]
        
        logger.info(f"LSTM sequences: train={len(X_seq_train)}, val={len(X_seq_val)}, test={len(X_seq_test)}")
        
        if len(X_seq_train) > 0:
            lstm_model = train_lstm(X_seq_train, y_seq_train, X_seq_val, y_seq_val, config=config)
            
            if len(X_seq_test) > 0:
                logger.info("Evaluating LSTM on test set...")
                evaluate_lstm(lstm_model, X_seq_test, y_seq_test, config=config)
            else:
                logger.info("Evaluating LSTM on validation set (no test sequences)...")
                evaluate_lstm(lstm_model, X_seq_val, y_seq_val, config=config)
        else:
            logger.warning("Not enough data to create time-series sequences. Skipping LSTM.")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    print_section("Pipeline Complete")
    logger.info("All outputs written to models/, reports/metrics/, reports/figures/")
    logger.info("Run notebooks/ for visualization and narrative analysis.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update base dict with values from update dict."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BRFSS Obesity Early Warning — End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--track",
        choices=["regression", "classification", "both"],
        default="both",
        help="Which modeling track to run (default: both)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run GroupKFold cross-validation (grouped by state)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(track=args.track, cv=args.cv)
