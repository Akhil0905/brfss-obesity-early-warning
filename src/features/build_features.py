"""
src/features/build_features.py
--------------------------------
Feature construction for the BRFSS Obesity Early Warning project.

This module takes the cleaned interim DataFrame and produces the final
model-ready feature matrix. Features include:

  - Temporal: year (numeric), decade indicator
  - Geographic: encoded state (LocationAbbr_enc), encoded region (region_enc)
  - Demographic: stratum_category_enc, stratum_value_enc, income_enc,
    education_enc, gender_enc
  - Lag features: Data_Value at year T-1, T-2 for same (state, category, subgroup)
  - Rolling mean: rolling 3-year average of Data_Value
  - Trend: year-over-year change in Data_Value (T vs T-1)
  - Confidence interval width (High_CI - Low_CI) as a proxy for estimation precision

Design notes:
  - Lag/rolling features are built BEFORE splitting so that we can leverage
    the full timeseries. However, the lag values themselves only use
    *past* data relative to each observation — no look-ahead.
  - After building features, rows with insufficient lag history (NaN lag values)
    retain NaNs which are imputed with the group median during final assembly.
  - The final feature matrix excludes unneeded string/metadata columns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, load_config, write_csv
from src.utils.paths import processed_data_path

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Core feature columns included in the final feature matrix
# ---------------------------------------------------------------------------

_FEATURE_CANDIDATES = [
    # Temporal
    "YearStart",
    "year_normalized",
    # Geographic
    "LocationAbbr_enc",
    "region_enc",
    # Demographic / stratification
    "stratum_category_enc",
    "stratum_value_enc",
    "income_enc",
    "education_enc",
    "gender_enc",
    # Lag / rolling / trend (added dynamically)
    "lag_1",
    "lag_2",
    "rolling_mean_3",
    "trend_1yr",
    # Precision proxy
    "ci_width",
]


def build_features(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix from the cleaned interim DataFrame.

    Args:
        df: Cleaned DataFrame from :func:`src.data.preprocess.preprocess`.
        config: Pre-loaded config dict. If None, loaded from config.yaml.
        save_output: Whether to save the feature matrix to data/processed/.

    Returns:
        :class:`pandas.DataFrame` containing only the final feature columns
        and the index/key columns needed for joining back to targets.
    """
    if config is None:
        config = load_config()

    logger.info(f"Building features | input shape: {df.shape}")
    df = df.copy()

    # 1. Temporal features
    df = _add_temporal_features(df)

    # 2. Confidence interval width
    df = _add_ci_width(df)

    # 3. Lag / rolling / trend features
    group_key: List[str] = config["features"]["group_key"]
    lag_years: List[int] = config["features"]["lag_years"]
    rolling_window: int = config["features"]["rolling_window"]
    df = _add_lag_features(df, group_key, lag_years, rolling_window)

    # 4. Assemble the final feature matrix
    feature_df = _select_features(df)

    # 5. Final null imputation: fill remaining NaNs with column median
    #    (only for lag/rolling columns where early years have no history)
    numeric_cols = feature_df.select_dtypes(include="number").columns
    null_before = feature_df[numeric_cols].isnull().sum().sum()
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(
        feature_df[numeric_cols].median()
    )
    if null_before > 0:
        logger.info(f"Imputed {null_before:,} NaN(s) with column medians (lag/rolling history gaps)")

    logger.info(f"Feature matrix built | shape: {feature_df.shape}")

    if save_output:
        out_path = processed_data_path(config["data"]["features_filename"])
        write_csv(feature_df, out_path)

    return feature_df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features derived from YearStart."""
    df["year_normalized"] = (df["YearStart"] - df["YearStart"].min()) / max(
        df["YearStart"].max() - df["YearStart"].min(), 1
    )
    return df


def _add_ci_width(df: pd.DataFrame) -> pd.DataFrame:
    """Add confidence interval width as a survey precision proxy."""
    if "High_Confidence_Limit" in df.columns and "Low_Confidence_Limit" in df.columns:
        df["ci_width"] = df["High_Confidence_Limit"] - df["Low_Confidence_Limit"]
    else:
        df["ci_width"] = np.nan
    return df


def _add_lag_features(
    df: pd.DataFrame,
    group_key: List[str],
    lag_years: List[int],
    rolling_window: int,
) -> pd.DataFrame:
    """Add lag, rolling mean, and trend features for Data_Value.

    Uses a **merge-on-year-offset** approach — no groupby, no shift, no lambdas.
    For each lag L, we create a copy of (group_key, YearStart, Data_Value),
    rename YearStart to YearStart+L, then merge back to the original.
    This is a vectorized hash join and runs in milliseconds on 20k rows.

    Args:
        df: Cleaned DataFrame with YearStart and Data_Value.
        group_key: Columns identifying a unique panel unit across years.
        lag_years: List of lag offsets (in years) to compute.
        rolling_window: Number of years for the rolling mean (unused directly —
            rolling_mean is the mean of available lag columns).

    Returns:
        DataFrame with new lag / rolling_mean / trend columns added.
    """
    available_key_cols = [c for c in group_key if c in df.columns]
    if not available_key_cols:
        logger.warning(f"Group key columns {group_key} not in data. Skipping lag features.")
        for lag in lag_years:
            df[f"lag_{lag}"] = np.nan
        df[f"rolling_mean_{rolling_window}"] = np.nan
        df["trend_1yr"] = np.nan
        return df

    if len(available_key_cols) < len(group_key):
        logger.warning(
            f"Missing group key cols: {set(group_key) - set(available_key_cols)}. "
            f"Using: {available_key_cols}"
        )

    # Ensure YearStart is int for reliable merge arithmetic
    df = df.copy()
    df["YearStart"] = pd.to_numeric(df["YearStart"], errors="coerce").astype("Int64")

    merge_cols = available_key_cols + ["YearStart", "Data_Value"]
    ref = df[merge_cols].copy()

    for lag in lag_years:
        col_name = f"lag_{lag}"
        # Build a lookup table: shift YearStart forward by `lag` so it aligns
        # with the row at year T when merged on T (the observation year)
        shifted = ref.rename(columns={"Data_Value": col_name}).copy()
        shifted["YearStart"] = shifted["YearStart"] + lag

        df = df.merge(
            shifted[available_key_cols + ["YearStart", col_name]],
            on=available_key_cols + ["YearStart"],
            how="left",
        )
        n_valid = df[col_name].notna().sum()
        logger.info(f"  Lag-{lag}: {n_valid:,} non-null values (merge approach)")

    # Rolling mean = arithmetic mean of available lag columns (no groupby needed)
    lag_cols_present = [f"lag_{l}" for l in lag_years if f"lag_{l}" in df.columns]
    if lag_cols_present:
        df[f"rolling_mean_{rolling_window}"] = df[lag_cols_present].mean(axis=1)
    else:
        df[f"rolling_mean_{rolling_window}"] = np.nan

    # Trend = lag_1 - lag_2 (year-over-year change as seen from past)
    if "lag_1" in df.columns and "lag_2" in df.columns:
        df["trend_1yr"] = df["lag_1"] - df["lag_2"]
    else:
        df["trend_1yr"] = np.nan

    return df




def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and return only the final feature columns plus key identifiers.

    Key identifiers (YearStart, LocationAbbr) are kept so we can join back
    to targets later without rebuilding them.
    """
    # Always keep these for joining
    keep_always = ["YearStart", "LocationAbbr", "stratum_category", "stratum_value"]
    keep_always = [c for c in keep_always if c in df.columns]

    feature_cols = [c for c in _FEATURE_CANDIDATES if c in df.columns]
    all_cols = list(dict.fromkeys(keep_always + feature_cols))  # ordered, deduped

    missing = [c for c in _FEATURE_CANDIDATES if c not in df.columns]
    if missing:
        logger.debug(f"Feature candidates not available in data: {missing}")

    return df[all_cols].copy()


def get_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    """Return the list of numeric feature columns (excluding key identifiers).

    Args:
        feature_df: Feature matrix from :func:`build_features`.

    Returns:
        List of column names that should be passed to model ``.fit(X, y)``.
    """
    exclude = {"YearStart", "LocationAbbr", "stratum_category", "stratum_value"}
    return [c for c in feature_df.columns if c not in exclude]
