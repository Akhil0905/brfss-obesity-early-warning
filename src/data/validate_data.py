"""
src/data/validate_data.py
-------------------------
Data quality and schema validation for the BRFSS Obesity Early Warning project.

Validation checks for each stage of the pipeline are collected here.
All checks log warnings/errors but — by default — do not halt execution,
so the pipeline can continue and surface a complete picture of data quality
rather than crashing on the first issue.

Set ``strict=True`` to raise on failure instead of warning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected dtypes / columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: List[str] = [
    "YearStart",
    "LocationAbbr",
    "Data_Value",
    "StratificationCategory1",
    "Stratification1",
]

NUMERIC_COLUMNS: List[str] = [
    "Data_Value",
    "YearStart",
]


# ---------------------------------------------------------------------------
# Public validation API
# ---------------------------------------------------------------------------


def validate_raw(df: pd.DataFrame, strict: bool = False) -> Dict[str, Any]:
    """Run data quality checks on the raw/filtered DataFrame.

    Checks performed:
      1. Required columns present
      2. Data_Value is (or can be coerced to) numeric
      3. Missing value summary
      4. Year range sanity (2000 – present)
      5. Data_Value range sanity (0 – 100, as it is a percentage)
      6. Duplicate rows check

    Args:
        df: DataFrame to validate.
        strict: If True, raise :exc:`ValueError` on any failed check.

    Returns:
        Dictionary of validation results including counts and flags.
    """
    results: Dict[str, Any] = {}

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    results["missing_required_columns"] = missing_cols
    _log_check(
        name="required_columns",
        passed=(len(missing_cols) == 0),
        message=(
            f"All required columns present."
            if not missing_cols
            else f"Missing columns: {missing_cols}"
        ),
        strict=strict,
    )

    # 2. Coerce Data_Value to numeric
    if "Data_Value" in df.columns:
        data_val_numeric = pd.to_numeric(df["Data_Value"], errors="coerce")
        n_unparseable = data_val_numeric.isna().sum() - df["Data_Value"].isna().sum()
        results["data_value_unparseable_count"] = int(max(0, n_unparseable))
        _log_check(
            name="data_value_numeric",
            passed=(n_unparseable == 0),
            message=(
                "Data_Value is fully numeric."
                if n_unparseable == 0
                else f"{n_unparseable:,} rows have non-numeric Data_Value (will be treated as NaN)"
            ),
            strict=False,  # non-strict; we handle NAs downstream
        )

    # 3. Missing value summary
    null_counts = df.isnull().sum()
    null_summary = null_counts[null_counts > 0].to_dict()
    results["null_counts"] = {k: int(v) for k, v in null_summary.items()}
    null_pct = (null_counts / len(df) * 100).round(1)

    # NOTE: In the BRFSS dataset, the unstacked demographic columns
    # (Sex, Income, Education, Age(years), Race/Ethnicity, Total) are
    # STRUCTURALLY sparse — each row is stratified by exactly ONE category,
    # so all other demographic columns will be blank for that row.
    # This is expected behaviour, not a data quality problem.
    # We use a higher threshold (80%) to avoid false-alarm warnings on these.
    _structurally_sparse = {"Total", "Age(years)", "Education", "Sex", "Income",
                            "Race/Ethnicity", "Gender"}
    high_null_cols = [
        c for c in null_pct[null_pct > 80].index.tolist()
        if c not in _structurally_sparse
    ]
    structurally_sparse_high = [
        c for c in null_pct[null_pct > 50].index.tolist()
        if c in _structurally_sparse
    ]
    if structurally_sparse_high:
        logger.info(
            f"Structurally sparse columns (expected by BRFSS design): "
            f"{structurally_sparse_high} — these are blank for rows stratified "
            f"by a different category."
        )
    if high_null_cols:
        logger.warning(
            f"Columns with >80%% missing values (unexpected): {high_null_cols}. "
            f"These may need inspection before modeling."
        )
    results["high_missingness_columns"] = high_null_cols

    # 4. Year range sanity
    if "YearStart" in df.columns:
        years = pd.to_numeric(df["YearStart"], errors="coerce").dropna()
        min_year, max_year = int(years.min()), int(years.max())
        results["year_range"] = (min_year, max_year)
        _log_check(
            name="year_range",
            passed=(2000 <= min_year and max_year <= 2030),
            message=f"Year range: {min_year}–{max_year}",
            strict=False,
        )

    # 5. Data_Value range sanity
    if "Data_Value" in df.columns:
        vals = pd.to_numeric(df["Data_Value"], errors="coerce").dropna()
        out_of_range = ((vals < 0) | (vals > 100)).sum()
        results["data_value_out_of_range"] = int(out_of_range)
        _log_check(
            name="data_value_range",
            passed=(out_of_range == 0),
            message=(
                "All Data_Value entries are in [0, 100]."
                if out_of_range == 0
                else f"{out_of_range:,} Data_Value entries are outside [0, 100]."
            ),
            strict=strict,
        )

    # 6. Duplicate rows
    n_dupes = int(df.duplicated().sum())
    results["duplicate_rows"] = n_dupes
    _log_check(
        name="duplicates",
        passed=(n_dupes == 0),
        message=(
            "No duplicate rows detected."
            if n_dupes == 0
            else f"{n_dupes:,} exact duplicate rows found."
        ),
        strict=False,
    )

    logger.info(
        f"Validation complete | rows={len(df):,} | "
        f"null_cols={len(null_summary)} | "
        f"issues={'none' if not high_null_cols and not missing_cols else 'see above'}"
    )
    return results


def validate_features(
    df: pd.DataFrame,
    expected_features: Optional[List[str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate the model-ready feature matrix.

    Args:
        df: Feature DataFrame produced by :mod:`src.features.build_features`.
        expected_features: Optional explicit list of required column names.
        strict: Raise on failure if True.

    Returns:
        Dictionary of validation results.
    """
    results: Dict[str, Any] = {}

    # Check shape
    results["n_rows"] = len(df)
    results["n_features"] = len(df.columns)
    if len(df) == 0:
        _log_check("non_empty", False, "Feature matrix is empty!", strict=True)

    # No remaining object columns (should all be numeric by this stage)
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    results["remaining_object_columns"] = obj_cols
    _log_check(
        "all_numeric",
        passed=(len(obj_cols) == 0),
        message=(
            "Feature matrix is fully numeric."
            if not obj_cols
            else f"Non-numeric columns remain: {obj_cols}"
        ),
        strict=strict,
    )

    # No infinite values
    import numpy as np
    inf_count = int(np.isinf(df.select_dtypes(include="number")).sum().sum())
    results["infinite_values"] = inf_count
    _log_check(
        "no_infinities",
        passed=(inf_count == 0),
        message=(
            "No infinite values."
            if inf_count == 0
            else f"{inf_count:,} infinite values present."
        ),
        strict=strict,
    )

    # Null count in feature matrix
    null_total = int(df.isnull().sum().sum())
    results["null_total"] = null_total
    if null_total > 0:
        logger.warning(f"Feature matrix contains {null_total:,} null values. Verify imputation is complete.")

    # Optional: expected features present
    if expected_features:
        missing = [f for f in expected_features if f not in df.columns]
        results["missing_expected_features"] = missing
        _log_check(
            "expected_features",
            passed=(len(missing) == 0),
            message=(
                "All expected features present."
                if not missing
                else f"Missing expected features: {missing}"
            ),
            strict=strict,
        )

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _log_check(name: str, passed: bool, message: str, strict: bool) -> None:
    """Log a validation check result and optionally raise on failure.

    Args:
        name: Short identifier for the check.
        passed: Whether the check passed.
        message: Human-readable result description.
        strict: If True and check failed, raise :exc:`ValueError`.
    """
    status = "PASS" if passed else "FAIL"
    log_fn = logger.info if passed else logger.warning
    log_fn(f"  [{status}] {name}: {message}")

    if not passed and strict:
        raise ValueError(f"Validation check failed [{name}]: {message}")
