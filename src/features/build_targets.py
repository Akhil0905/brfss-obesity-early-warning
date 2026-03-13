"""
src/features/build_targets.py
-------------------------------
Target variable construction for the BRFSS Obesity Early Warning project.

This module builds:
  1. **Regression target** — the raw `Data_Value` (obesity prevalence %)
  2. **Classification target** — `high_risk_obesity` (1 if prevalence exceeds
     the top-quartile threshold computed on the training set)
  3. **Early warning target** — `early_warning` (1 if a state/group combo that
     is NOT currently high-risk will *become* high-risk in the next observed year)

Early Warning Design:
  The early warning target requires finding future-year observations for the
  same (LocationAbbr, stratum_category, stratum_value) panel unit. If fewer
  than `early_warning_min_pairs` valid pairs exist, the pipeline logs a warning
  and returns only the `high_risk_obesity` target.

Leakage prevention:
  The high-risk threshold is computed **on training data only** and then applied
  to validation and test data. The threshold is returned alongside the targets
  so it can be saved and reused consistently.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, load_config, write_csv
from src.utils.paths import processed_data_path

logger = get_logger(__name__)


def build_targets(
    df: pd.DataFrame,
    train_mask: Optional[pd.Series] = None,
    config: Optional[dict] = None,
    save_output: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Build all target variables from the cleaned DataFrame.

    Args:
        df: Cleaned DataFrame (output of :func:`src.data.preprocess.preprocess`).
            Must contain 'Data_Value', 'YearStart', 'LocationAbbr',
            'stratum_category', 'stratum_value'.
        train_mask: Boolean Series indicating training rows (used to compute
            the high-risk threshold without data leakage). If None, the
            threshold is computed on the full dataset.
        config: Pre-loaded config dict. If None, loaded from config.yaml.
        save_output: Whether to save the target DataFrame to data/processed/.

    Returns:
        Tuple of:
          - :class:`pandas.DataFrame` with target columns added
          - ``metadata`` dict containing the high-risk threshold and a flag
            indicating whether the early warning target was successfully built
    """
    if config is None:
        config = load_config()

    target_cfg = config["targets"]
    percentile = target_cfg["high_risk_percentile"]
    high_risk_col = target_cfg["high_risk_col"]
    ew_col = target_cfg["early_warning_col"]
    ew_min_pairs = target_cfg["early_warning_min_pairs"]

    logger.info(f"Building targets | input shape: {df.shape}")
    df = df.copy()

    # -----------------------------------------------------------------------
    # 1. Regression target: Data_Value (already present — just verify)
    # -----------------------------------------------------------------------
    regression_target = "Data_Value"
    if regression_target not in df.columns:
        raise ValueError("'Data_Value' column not found. Cannot build regression target.")
    null_target = df[regression_target].isnull().sum()
    if null_target > 0:
        logger.warning(f"{null_target:,} rows have missing Data_Value (regression target).")

    # -----------------------------------------------------------------------
    # 2. High-risk threshold (leakage-safe)
    # -----------------------------------------------------------------------
    if train_mask is not None:
        train_values = df.loc[train_mask, regression_target].dropna()
        logger.info(
            f"Computing {percentile}th-percentile threshold on {len(train_values):,} training rows."
        )
    else:
        train_values = df[regression_target].dropna()
        logger.warning(
            "No train_mask provided. Computing threshold on full dataset — "
            "this may introduce leakage if applied to val/test."
        )

    threshold = float(np.percentile(train_values, percentile))
    logger.info(f"High-risk threshold (P{percentile}): {threshold:.2f}%")

    # -----------------------------------------------------------------------
    # 3. high_risk_obesity classification target
    # -----------------------------------------------------------------------
    df[high_risk_col] = (df[regression_target] >= threshold).astype(int)
    n_high_risk = df[high_risk_col].sum()
    pct_high_risk = 100 * n_high_risk / len(df)
    logger.info(
        f"high_risk_obesity: {n_high_risk:,} high-risk rows ({pct_high_risk:.1f}% of total)"
    )

    # -----------------------------------------------------------------------
    # 4. Early Warning target (next-year high-risk crossing)
    # -----------------------------------------------------------------------
    ew_built = False
    group_key = config["features"]["group_key"]
    available_key = [c for c in group_key if c in df.columns]

    if not available_key:
        logger.warning(
            "Cannot build early warning target: group key columns not present."
        )
    else:
        ew_df, ew_built, n_pairs = _build_early_warning(
            df=df,
            group_key=available_key,
            high_risk_col=high_risk_col,
            ew_col=ew_col,
            min_pairs=ew_min_pairs,
        )
        if ew_built:
            df = ew_df

    # -----------------------------------------------------------------------
    # 5. Assemble target DataFrame
    # -----------------------------------------------------------------------
    target_cols = ["YearStart", "LocationAbbr", "LocationDesc", "stratum_category", "stratum_value",
                   regression_target, high_risk_col]
    if ew_built and ew_col in df.columns:
        target_cols.append(ew_col)

    available_target_cols = [c for c in target_cols if c in df.columns]
    target_df = df[available_target_cols].copy()

    metadata = {
        "high_risk_threshold": threshold,
        "high_risk_percentile": percentile,
        "early_warning_built": ew_built,
        "n_total_rows": len(target_df),
        "n_high_risk": int(df[high_risk_col].sum()),
    }

    logger.info(f"Targets built | shape: {target_df.shape} | early_warning={ew_built}")

    if save_output:
        out_path = processed_data_path(config["data"]["targets_filename"])
        write_csv(target_df, out_path)

    return target_df, metadata


def _build_early_warning(
    df: pd.DataFrame,
    group_key: List[str],
    high_risk_col: str,
    ew_col: str,
    min_pairs: int,
) -> Tuple[pd.DataFrame, bool, int]:
    """Attempt to build the early warning target.

    For each panel unit (state/subgroup), this finds pairs of consecutive
    observed years and marks year T as early_warning=1 if:
      - The unit is NOT high-risk at year T (high_risk_obesity == 0)
      - The unit IS high-risk at year T+1

    This is a lead (forward) shift applied within each panel group.

    Args:
        df: DataFrame with sorted years and high_risk_col.
        group_key: Group key columns identifying a panel unit.
        high_risk_col: Name of the high-risk binary column.
        ew_col: Name of the new early warning column to add.
        min_pairs: Minimum number of valid pairs required to enable this target.

    Returns:
        Tuple of (modified_df, success_flag, n_valid_pairs).
    """
    logger.info("Attempting to build early warning target...")

    df = df.sort_values(group_key + ["YearStart"]).copy()

    # For each group, shift high_risk by -1 (look at next year's value)
    next_year_high_risk = df.groupby(group_key, sort=False)[high_risk_col].shift(-1)

    # Early warning applies only to currently non-high-risk rows
    currently_not_high_risk = df[high_risk_col] == 0
    will_be_high_risk = next_year_high_risk == 1

    ew_series = pd.Series(np.nan, index=df.index)
    # For non-high-risk rows with a valid next-year observation:
    ew_series[currently_not_high_risk & next_year_high_risk.notna()] = 0
    ew_series[currently_not_high_risk & will_be_high_risk] = 1

    n_valid_pairs = int(ew_series.notna().sum())
    n_positive_pairs = int((ew_series == 1).sum())

    logger.info(
        f"  Early warning pairs: {n_valid_pairs:,} valid | "
        f"{n_positive_pairs:,} positive transitions (will cross into high-risk)"
    )

    if n_valid_pairs < min_pairs:
        logger.warning(
            f"  Early warning: only {n_valid_pairs:,} valid pairs found "
            f"(minimum required: {min_pairs:,}). "
            f"Pipeline will use 'high_risk_obesity' as the classification target instead."
        )
        return df, False, n_valid_pairs

    logger.info(
        f"  Early warning target built successfully | "
        f"class balance: {n_positive_pairs / n_valid_pairs * 100:.1f}% positive"
    )
    df[ew_col] = ew_series
    return df, True, n_valid_pairs


def select_classification_target(
    metadata: Dict,
    config: Optional[dict] = None,
) -> str:
    """Decide which classification target to use based on config + feasibility.

    Args:
        metadata: Output from :func:`build_targets` (must contain 'early_warning_built').
        config: Pre-loaded config. If None, loaded from config.yaml.

    Returns:
        Column name of the selected classification target.
    """
    if config is None:
        config = load_config()

    target_pref = config["classification"]["target"]
    high_risk_col = config["targets"]["high_risk_col"]
    ew_col = config["targets"]["early_warning_col"]

    if target_pref == "early_warning":
        if metadata.get("early_warning_built"):
            logger.info(f"Classification target: '{ew_col}' (as configured)")
            return ew_col
        else:
            logger.warning(
                f"Config requested early_warning target but it was not built. "
                f"Falling back to '{high_risk_col}'."
            )
            return high_risk_col
    elif target_pref == "high_risk_obesity":
        logger.info(f"Classification target: '{high_risk_col}' (as configured)")
        return high_risk_col
    else:  # "auto"
        if metadata.get("early_warning_built"):
            logger.info(f"Classification target: '{ew_col}' (auto-selected — early warning feasible)")
            return ew_col
        else:
            logger.info(f"Classification target: '{high_risk_col}' (auto-selected — early warning not feasible)")
            return high_risk_col
