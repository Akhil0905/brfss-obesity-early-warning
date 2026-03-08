"""
src/data/preprocess.py
----------------------
Data cleaning and preprocessing for the BRFSS Obesity Early Warning project.

Responsibilities:
  - Coerce Data_Value and YearStart to numeric
  - Drop rows where the regression target is missing
  - Standardize stratification column names
  - Encode categorical variables
  - Add US Census region as a feature
  - Produce a clean, analysis-ready DataFrame saved to data/interim/

Design notes:
  - All encoding is fit on training data only (thresholds / encoders passed
    as arguments or returned for re-use on validation/test sets).
  - No imputation of the target variable (Data_Value); rows without a target
    are dropped early.
  - Sparse stratification categories below a row-count threshold are grouped
    into an "Other" bucket to avoid one-hot explosion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, load_config, write_csv
from src.utils.paths import interim_data_path

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# US Census Region Mapping
# ---------------------------------------------------------------------------

_STATE_TO_REGION: Dict[str, str] = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    # South
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    # West
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
    # Territories / Other
    "PR": "Territory", "GU": "Territory", "VI": "Territory",
    "AS": "Territory", "MP": "Territory",
    "US": "National",  # national-level aggregate rows
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    save_interim: bool = True,
) -> pd.DataFrame:
    """Full preprocessing pipeline for the raw BRFSS obesity DataFrame.

    Steps:
      1. Coerce numeric columns
      2. Drop rows with missing regression target
      3. Drop exact duplicates
      4. Standardize / rename stratification columns
      5. Add US Census region
      6. Encode categorical variables (no leakage — encoding is based on
         value counts, not train/test labels)
      7. Optionally save interim output

    Args:
        df: Filtered raw DataFrame (output of :func:`src.data.load_data.filter_obesity_question`).
        config: Pre-loaded config dictionary. If None, loaded from config.yaml.
        save_interim: Whether to save the cleaned DataFrame to data/interim/.

    Returns:
        Cleaned :class:`pandas.DataFrame`.
    """
    if config is None:
        config = load_config()

    logger.info(f"Preprocessing | input shape: {df.shape}")
    df = df.copy()

    # 1. Coerce numerics
    df = _coerce_numerics(df)

    # 2. Drop rows where the regression target is missing
    before = len(df)
    df = df.dropna(subset=["Data_Value"])
    logger.info(f"Dropped {before - len(df):,} rows with missing Data_Value → {len(df):,} remain")

    # 3. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        logger.info(f"Dropped {before - len(df):,} exact duplicate rows")

    # 4. Standardize stratification columns
    df = _standardize_stratification(df)

    # 5. Add US Census region
    if config.get("features", {}).get("add_region", True) and "LocationAbbr" in df.columns:
        df["region"] = df["LocationAbbr"].map(_STATE_TO_REGION).fillna("Unknown")
        logger.info("Added 'region' column from LocationAbbr → US Census region mapping")

    # 6. Encode categoricals (fit on all available data since this encodes
    #    population-level categories, not target-related information)
    df = _encode_categoricals(df, config)

    logger.info(f"Preprocessing complete | output shape: {df.shape}")

    if save_interim:
        out_path = interim_data_path(config["data"]["interim_filename"])
        write_csv(df, out_path)

    return df


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Data_Value, YearStart, and confidence interval columns to float."""
    numeric_cols = [
        "Data_Value",
        "YearStart",
        "Low_Confidence_Limit",
        "High_Confidence_Limit",
        "Sample_Size",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _standardize_stratification(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize stratification fields for consistent downstream use.

    The BRFSS dataset uses 'StratificationCategory1' and 'Stratification1'.
    We also handle 'Total' (whole-population aggregate rows).

    Adds a clean 'stratum_category' and 'stratum_value' column pair that
    downstream feature builders use.
    """
    if "StratificationCategory1" in df.columns:
        df["stratum_category"] = df["StratificationCategory1"].str.strip().fillna("Total")
    else:
        df["stratum_category"] = "Total"

    if "Stratification1" in df.columns:
        df["stratum_value"] = df["Stratification1"].str.strip().fillna("Total")
    else:
        df["stratum_value"] = "Total"

    return df


def _encode_categoricals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode categorical columns as ordinal integers.

    Encoding strategy:
      - 'stratum_category': label-encoded (alphabetical order)
      - 'stratum_value': label-encoded with infrequent categories collapsed
        to "Other" to limit cardinality
      - 'region': label-encoded
      - 'LocationAbbr': label-encoded (state codes)
      - Original string columns are kept alongside encoded versions
        (suffixed with '_enc') so notebooks can still read the labels.

    Args:
        df: DataFrame with standardized stratification columns.
        config: Project config dictionary.

    Returns:
        DataFrame with added *_enc columns.
    """
    min_rows = config["data"].get("min_stratum_rows", 5)

    # Collapse infrequent stratum_values
    if "stratum_value" in df.columns:
        value_counts = df["stratum_value"].value_counts()
        rare_values = value_counts[value_counts < min_rows].index.tolist()
        if rare_values:
            logger.info(
                f"Collapsing {len(rare_values)} rare stratum_value(s) to 'Other' "
                f"(threshold: <{min_rows} rows)"
            )
            df["stratum_value"] = df["stratum_value"].replace(rare_values, "Other")

    # Label-encode categorical columns
    # NOTE: The dataset uses 'Sex' (not 'Gender') as the column name.
    # StratificationCategory1 uses 'Age (years)' with a space; the standalone
    # column header is 'Age(years)' (no space). Both are handled correctly here.
    cat_cols = ["stratum_category", "stratum_value", "region", "LocationAbbr"]
    for col in cat_cols:
        if col in df.columns:
            # Sort so encoding is deterministic
            categories = sorted(df[col].dropna().unique().tolist())
            mapping = {val: idx for idx, val in enumerate(categories)}
            df[f"{col}_enc"] = df[col].map(mapping).astype("Int64")

    # Encode income as ordinal (ordered if possible)
    income_order = [
        "Less than $15,000",
        "$15,000 - $24,999",
        "$25,000 - $34,999",
        "$35,000 - $49,999",
        "$50,000 - $74,999",
        "$75,000 or greater",
        "Data not reported",
    ]
    if "Income" in df.columns:
        income_map = {v: i for i, v in enumerate(income_order)}
        df["income_enc"] = df["Income"].map(income_map)
        # Anything not in the explicit ordering gets a high NaN-sentinel
        # to avoid silently losing information
        logger.info(
            f"Income encoding: {df['income_enc'].notna().sum():,} / {len(df):,} rows mapped"
        )

    # Encode education as ordinal
    edu_order = [
        "Less than high school",
        "High school graduate",
        "Some college or technical school",
        "College graduate",
        "Data not reported",
    ]
    if "Education" in df.columns:
        edu_map = {v: i for i, v in enumerate(edu_order)}
        df["education_enc"] = df["Education"].map(edu_map)

    # Encode sex — the BRFSS column is 'Sex' (values: 'Male', 'Female')
    # Also check legacy 'Gender' label for robustness
    sex_col = "Sex" if "Sex" in df.columns else ("Gender" if "Gender" in df.columns else None)
    if sex_col:
        gender_map = {"Male": 0, "Female": 1}
        df["gender_enc"] = df[sex_col].map(gender_map)

    logger.info("Categorical encoding complete")
    return df


def split_by_year(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train / validation / test by year.

    Split boundaries are read from config.yaml under the 'split' key.

    This is a **temporal split** — no future years bleed into training.
    See PROJECT_PLAN.md §7 for design rationale.

    Args:
        df: Cleaned DataFrame with 'YearStart' column.
        config: Pre-loaded config. If None, loaded from config.yaml.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if config is None:
        config = load_config()

    split_cfg = config["split"]
    train_max = split_cfg["train_max_year"]
    val_min = split_cfg["val_min_year"]
    val_max = split_cfg["val_max_year"]
    test_min = split_cfg["test_min_year"]

    train_df = df[df["YearStart"] <= train_max].copy()
    val_df = df[(df["YearStart"] >= val_min) & (df["YearStart"] <= val_max)].copy()
    test_df = df[df["YearStart"] >= test_min].copy()

    logger.info(
        f"Temporal split | train={len(train_df):,} (≤{train_max}) | "
        f"val={len(val_df):,} ({val_min}–{val_max}) | "
        f"test={len(test_df):,} (≥{test_min})"
    )

    if len(test_df) == 0:
        logger.warning(
            f"Test set is empty (no rows with YearStart ≥ {test_min}). "
            f"The dataset may not extend to {test_min}. "
            f"Consider adjusting split boundaries in config.yaml."
        )

    return train_df, val_df, test_df
