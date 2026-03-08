"""
src/data/load_data.py
---------------------
Raw data loading for the BRFSS Obesity Early Warning project.

Responsibilities:
  - Locate and load the raw CSV from data/raw/
  - Apply a minimal column filter (keep only needed columns)
  - Return a raw DataFrame for downstream validation & preprocessing

The dataset is the CDC/Data.gov "Nutrition, Physical Activity, and Obesity —
Behavioral Risk Factor Surveillance System" CSV export.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.helpers import get_logger, load_config, read_csv
from src.utils.paths import raw_data_path

logger = get_logger(__name__)


def load_raw_data(
    filename: Optional[str] = None,
    keep_columns: Optional[List[str]] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Load the raw BRFSS CSV from data/raw/.

    Args:
        filename: Name of the CSV file inside data/raw/. If None, the value
            is read from configs/config.yaml.
        keep_columns: List of columns to retain. If None, uses the list
            specified in config.yaml under ``data.keep_columns``.
        config: Pre-loaded config dictionary. If None, config.yaml is loaded
            automatically.

    Returns:
        Raw :class:`pandas.DataFrame` with only the requested columns.

    Raises:
        FileNotFoundError: If the CSV file is not found in data/raw/.
        ValueError: If one or more required columns are missing from the file.
    """
    if config is None:
        config = load_config()

    if filename is None:
        filename = config["data"]["raw_filename"]

    if keep_columns is None:
        keep_columns = config["data"].get("keep_columns")

    raw_path: Path = raw_data_path(filename)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"\n\nRaw data file not found:\n  {raw_path}\n\n"
            "Please download the BRFSS dataset from Data.gov and place it in data/raw/.\n"
            "Expected filename:\n  Nutrition__Physical_Activity__and_Obesity__-_"
            "Behavioral_Risk_Factor_Surveillance_System.csv\n"
            "Download URL: https://catalog.data.gov/dataset/"
            "nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system"
        )

    df = read_csv(raw_path, low_memory=False)

    # Strip whitespace from all column names — the raw CSV has a trailing
    # space in 'High_Confidence_Limit ' (column 16) which would otherwise
    # cause silent column-not-found errors downstream.
    df.columns = df.columns.str.strip()

    # -----------------------------------------------------------------------
    # Column filtering — keep only what we need to reduce memory and noise
    # -----------------------------------------------------------------------
    if keep_columns:
        available = set(df.columns)
        requested = set(keep_columns)
        missing_cols = requested - available

        if missing_cols:
            logger.warning(
                f"The following requested columns are NOT present in the raw file "
                f"and will be skipped: {sorted(missing_cols)}"
            )

        cols_to_keep = [c for c in keep_columns if c in available]
        df = df[cols_to_keep]
        logger.info(f"Retained {len(cols_to_keep)} of {len(keep_columns)} requested columns")

    logger.info(
        f"Raw data loaded | shape={df.shape} | "
        f"years={_year_range(df)} | "
        f"locations={_n_locations(df)}"
    )
    return df


def filter_obesity_question(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Filter the raw DataFrame to the primary obesity prevalence question.

    The BRFSS dataset contains many indicators. This function subsets to:
      - Class == config.data.target_class  (e.g. "Obesity / Weight Status")
      - Question == config.data.target_question

    Args:
        df: Raw or partially filtered DataFrame.
        config: Pre-loaded config. If None, loaded from config.yaml.

    Returns:
        Filtered :class:`pandas.DataFrame`.
    """
    if config is None:
        config = load_config()

    target_class = config["data"]["target_class"]
    target_question = config["data"]["target_question"]

    before = len(df)

    # Filter by Class if the column exists
    if "Class" in df.columns:
        df = df[df["Class"] == target_class].copy()
        logger.info(f"After Class filter ('{target_class}'): {len(df):,} rows (removed {before - len(df):,})")

    # Filter by Question if the column exists
    if "Question" in df.columns:
        before_q = len(df)
        df = df[df["Question"] == target_question].copy()
        logger.info(
            f"After Question filter: {len(df):,} rows (removed {before_q - len(df):,})"
        )

    if len(df) == 0:
        raise ValueError(
            f"No rows remain after filtering to Class='{target_class}' and "
            f"Question='{target_question}'. Check that the dataset matches the "
            f"expected format, or update target_class / target_question in config.yaml."
        )

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _year_range(df: pd.DataFrame) -> str:
    """Return a string summary of the year range in the DataFrame."""
    year_col = "YearStart"
    if year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        if len(years) > 0:
            return f"{int(years.min())}–{int(years.max())}"
    return "unknown"


def _n_locations(df: pd.DataFrame) -> str:
    """Return the number of unique locations in the DataFrame."""
    for col in ("LocationAbbr", "LocationDesc"):
        if col in df.columns:
            return str(df[col].nunique())
    return "unknown"
