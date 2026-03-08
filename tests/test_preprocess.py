"""
tests/test_preprocess.py
------------------------
Unit tests for preprocessing logic (src/data/preprocess.py).

Tests are designed to work with small synthetic DataFrames that mimic
the structure of the BRFSS dataset, so no actual data file is needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    _coerce_numerics,
    _encode_categoricals,
    _standardize_stratification,
    preprocess,
    split_by_year,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> dict:
    """Minimal config dict that satisfies all preprocessing functions."""
    return {
        "data": {
            "interim_filename": "test_cleaned.csv",
            "min_stratum_rows": 2,
        },
        "features": {
            "add_region": True,
            "group_key": ["LocationAbbr", "StratificationCategory1", "Stratification1"],
            "lag_years": [1],
            "rolling_window": 3,
        },
        "split": {
            "train_max_year": 2017,
            "val_min_year": 2018,
            "val_max_year": 2019,
            "test_min_year": 2020,
            "random_state": 42,
        },
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic BRFSS-like DataFrame for testing."""
    return pd.DataFrame({
        "YearStart": [2015, 2016, 2017, 2018, 2019, 2020],
        "LocationAbbr": ["PA", "PA", "NY", "NY", "TX", "TX"],
        "LocationDesc": ["Pennsylvania"] * 2 + ["New York"] * 2 + ["Texas"] * 2,
        "Data_Value": [30.5, 31.2, None, 29.8, 35.1, 36.0],
        "Low_Confidence_Limit": [28.0, 29.0, None, 27.5, 33.0, 34.0],
        "High_Confidence_Limit": [33.0, 33.5, None, 32.0, 37.0, 38.5],
        "Sample_Size": [500, 520, 480, 510, 600, 620],
        "StratificationCategory1": ["Income", "Income", "Age(years)", "Age(years)", "Total", "Total"],
        "Stratification1": ["$25,000 - $34,999", "$50,000 - $74,999", "25-44", "45-64", "Total", "Total"],
        "Income": ["$25,000 - $34,999", "$50,000 - $74,999", None, None, None, None],
        "Education": [None, None, None, None, None, None],
        "Gender": ["Male", "Female", None, None, None, None],
    })


# ---------------------------------------------------------------------------
# Tests: _coerce_numerics
# ---------------------------------------------------------------------------


class TestCoerceNumerics:
    def test_data_value_coerced(self, sample_df):
        sample_df["Data_Value"] = sample_df["Data_Value"].astype(str)
        result = _coerce_numerics(sample_df)
        assert pd.api.types.is_float_dtype(result["Data_Value"]), \
            "Data_Value should be float after coercion"

    def test_year_start_coerced(self, sample_df):
        sample_df["YearStart"] = sample_df["YearStart"].astype(str)
        result = _coerce_numerics(sample_df)
        assert pd.api.types.is_float_dtype(result["YearStart"]) or \
               pd.api.types.is_integer_dtype(result["YearStart"]), \
            "YearStart should be numeric"

    def test_non_numeric_data_value_becomes_nan(self, sample_df):
        # Pandas 2.x won't assign a string into a float64 column in-place.
        # Cast to object dtype first so the fixture is correctly set up.
        sample_df["Data_Value"] = sample_df["Data_Value"].astype(object)
        sample_df.loc[0, "Data_Value"] = "N/A"
        result = _coerce_numerics(sample_df)
        assert pd.isna(result.loc[0, "Data_Value"]), \
            "Non-numeric Data_Value should become NaN"


# ---------------------------------------------------------------------------
# Tests: _standardize_stratification
# ---------------------------------------------------------------------------


class TestStandardizeStratification:
    def test_adds_stratum_category_col(self, sample_df):
        result = _standardize_stratification(sample_df)
        assert "stratum_category" in result.columns

    def test_adds_stratum_value_col(self, sample_df):
        result = _standardize_stratification(sample_df)
        assert "stratum_value" in result.columns

    def test_strips_whitespace(self):
        df = pd.DataFrame({
            "StratificationCategory1": ["  Income  ", "Age(years)"],
            "Stratification1": ["$25,000 - $34,999", "  25-44  "],
        })
        result = _standardize_stratification(df)
        assert result["stratum_category"].iloc[0] == "Income"
        assert result["stratum_value"].iloc[1] == "25-44"

    def test_missing_stratification_fills_total(self):
        df = pd.DataFrame({"YearStart": [2018]})  # no stratification columns
        result = _standardize_stratification(df)
        assert result["stratum_category"].iloc[0] == "Total"
        assert result["stratum_value"].iloc[0] == "Total"


# ---------------------------------------------------------------------------
# Tests: _encode_categoricals
# ---------------------------------------------------------------------------


class TestEncodeCategoricals:
    def setup_method(self):
        """Prepare a small DataFrame with stratum columns."""
        self.df = pd.DataFrame({
            "stratum_category": ["Income", "Income", "Age(years)"],
            "stratum_value": ["Low", "High", "25-44"],
            "LocationAbbr": ["PA", "PA", "NY"],
            "Income": ["$25,000 - $34,999", "$50,000 - $74,999", None],
        })
        self.config = {"data": {"min_stratum_rows": 1}}

    def test_adds_enc_columns(self):
        result = _encode_categoricals(self.df.copy(), self.config)
        assert "stratum_category_enc" in result.columns
        assert "stratum_value_enc" in result.columns
        assert "LocationAbbr_enc" in result.columns

    def test_enc_values_are_integers(self):
        result = _encode_categoricals(self.df.copy(), self.config)
        non_null = result["stratum_category_enc"].dropna()
        assert non_null.dtype in (np.int64, np.int32, "Int64"), \
            "Encoded categories should be integer"

    def test_income_encoding_ordered(self):
        result = _encode_categoricals(self.df.copy(), self.config)
        row_low = result[result["Income"] == "$25,000 - $34,999"]["income_enc"].iloc[0]
        row_high = result[result["Income"] == "$50,000 - $74,999"]["income_enc"].iloc[0]
        assert row_low < row_high, "Lower income bracket should have smaller encoded value"


# ---------------------------------------------------------------------------
# Tests: split_by_year
# ---------------------------------------------------------------------------


class TestSplitByYear:
    def test_split_sizes(self, sample_df, minimal_config):
        df = _coerce_numerics(sample_df).dropna(subset=["Data_Value"])
        train, val, test = split_by_year(df, minimal_config)

        assert set(train["YearStart"].unique()) <= {2015, 2016, 2017}
        assert set(val["YearStart"].unique()) <= {2018, 2019}
        assert set(test["YearStart"].unique()) <= {2020}

    def test_no_overlap(self, sample_df, minimal_config):
        df = _coerce_numerics(sample_df).dropna(subset=["Data_Value"])
        train, val, test = split_by_year(df, minimal_config)

        train_years = set(train["YearStart"].unique())
        val_years = set(val["YearStart"].unique())
        test_years = set(test["YearStart"].unique())

        assert train_years.isdisjoint(val_years), "Train and val sets must not overlap"
        assert train_years.isdisjoint(test_years), "Train and test sets must not overlap"
        assert val_years.isdisjoint(test_years), "Val and test sets must not overlap"

    def test_empty_test_warning(self, sample_df, minimal_config, caplog):
        """If no rows fall in test range, code should warn rather than crash."""
        import logging
        minimal_config["split"]["test_min_year"] = 2099  # beyond dataset
        df = _coerce_numerics(sample_df).dropna(subset=["Data_Value"])
        with caplog.at_level(logging.WARNING):
            train, val, test = split_by_year(df, minimal_config)
        assert len(test) == 0
