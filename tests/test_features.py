"""
tests/test_features.py
-----------------------
Unit tests for feature and target engineering (src/features/).

Uses small synthetic DataFrames that mimic the panel structure of BRFSS data.
No actual dataset or trained models are needed for these tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    _add_ci_width,
    _add_lag_features,
    _add_temporal_features,
    get_feature_columns,
)
from src.features.build_targets import _build_early_warning, build_targets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config() -> dict:
    """Minimal config dict for feature/target tests."""
    return {
        "data": {
            "targets_filename": "test_targets.csv",
            "features_filename": "test_features.csv",
        },
        "features": {
            "group_key": ["LocationAbbr", "StratificationCategory1", "Stratification1"],
            "lag_years": [1, 2],
            "rolling_window": 3,
            "add_region": True,
        },
        "targets": {
            "high_risk_percentile": 75,
            "high_risk_col": "high_risk_obesity",
            "early_warning_col": "early_warning",
            "early_warning_min_pairs": 2,
        },
        "split": {
            "train_max_year": 2017,
            "random_state": 42,
        },
        "output": {
            "save_models": False,
            "save_metrics": False,
            "save_figures": False,
        },
    }


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Synthetic panel DataFrame: 2 states × 1 subgroup × 5 years."""
    rows = []
    for state in ["PA", "NY"]:
        for year in [2014, 2015, 2016, 2017, 2018]:
            rows.append({
                "LocationAbbr": state,
                "YearStart": year,
                "StratificationCategory1": "Total",
                "Stratification1": "Total",
                "stratum_category": "Total",
                "stratum_value": "Total",
                "Data_Value": 28.0 + (year - 2014) * 1.5 + (3.0 if state == "NY" else 0.0),
                "High_Confidence_Limit": 32.0,
                "Low_Confidence_Limit": 26.0,
                "LocationAbbr_enc": 0 if state == "PA" else 1,
                "stratum_category_enc": 0,
                "stratum_value_enc": 0,
                "region_enc": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: _add_temporal_features
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    def test_adds_year_normalized(self, panel_df):
        result = _add_temporal_features(panel_df)
        assert "year_normalized" in result.columns

    def test_year_normalized_range(self, panel_df):
        result = _add_temporal_features(panel_df)
        col = result["year_normalized"]
        assert col.min() >= 0.0, "year_normalized should be >= 0"
        assert col.max() <= 1.0, "year_normalized should be <= 1"

    def test_year_normalized_monotone_within_state(self, panel_df):
        result = _add_temporal_features(panel_df)
        pa = result[result["LocationAbbr"] == "PA"].sort_values("YearStart")
        assert (pa["year_normalized"].diff().dropna() >= 0).all(), \
            "year_normalized should increase with YearStart"


# ---------------------------------------------------------------------------
# Tests: _add_ci_width
# ---------------------------------------------------------------------------


class TestCIWidth:
    def test_adds_ci_width(self, panel_df):
        result = _add_ci_width(panel_df)
        assert "ci_width" in result.columns

    def test_ci_width_values(self, panel_df):
        result = _add_ci_width(panel_df)
        # High_CI (32) - Low_CI (26) = 6
        expected = 32.0 - 26.0
        assert (result["ci_width"] == expected).all(), \
            f"CI width should be {expected}"

    def test_ci_width_nan_when_cols_missing(self):
        df = pd.DataFrame({"YearStart": [2018], "Data_Value": [30.0]})
        result = _add_ci_width(df)
        assert "ci_width" in result.columns
        assert pd.isna(result["ci_width"].iloc[0])


# ---------------------------------------------------------------------------
# Tests: _add_lag_features
# ---------------------------------------------------------------------------


class TestLagFeatures:
    def test_adds_lag_columns(self, panel_df):
        result = _add_lag_features(
            panel_df,
            group_key=["LocationAbbr", "StratificationCategory1", "Stratification1"],
            lag_years=[1, 2],
            rolling_window=3,
        )
        assert "lag_1" in result.columns
        assert "lag_2" in result.columns

    def test_lag_1_no_leakage(self, panel_df):
        """Lag-1 for the earliest year in a group should be NaN."""
        result = _add_lag_features(
            panel_df,
            group_key=["LocationAbbr", "StratificationCategory1", "Stratification1"],
            lag_years=[1],
            rolling_window=3,
        )
        earliest = result.groupby(["LocationAbbr", "StratificationCategory1", "Stratification1"])[
            "YearStart"
        ].transform("min")
        first_rows = result[result["YearStart"] == earliest]
        assert first_rows["lag_1"].isna().all(), \
            "Lag-1 for the first observed year in each group must be NaN (no historical data)"

    def test_lag_1_correct_value(self, panel_df):
        """Lag-1 value at year T should equal Data_Value at year T-1."""
        result = _add_lag_features(
            panel_df,
            group_key=["LocationAbbr", "StratificationCategory1", "Stratification1"],
            lag_years=[1],
            rolling_window=3,
        )
        pa = result[result["LocationAbbr"] == "PA"].sort_values("YearStart")
        # Lag-1 at 2015 should equal Data_Value at 2014
        val_2014 = pa[pa["YearStart"] == 2014]["Data_Value"].values[0]
        lag1_at_2015 = pa[pa["YearStart"] == 2015]["lag_1"].values[0]
        assert abs(lag1_at_2015 - val_2014) < 1e-6, \
            "lag_1 at year T should equal Data_Value at year T-1"

    def test_rolling_mean_added(self, panel_df):
        result = _add_lag_features(
            panel_df,
            group_key=["LocationAbbr", "StratificationCategory1", "Stratification1"],
            lag_years=[1],
            rolling_window=3,
        )
        assert "rolling_mean_3" in result.columns


# ---------------------------------------------------------------------------
# Tests: build_targets
# ---------------------------------------------------------------------------


class TestBuildTargets:
    def test_adds_high_risk_col(self, panel_df, base_config):
        train_mask = panel_df["YearStart"] <= 2017
        result_df, meta = build_targets(panel_df, train_mask=train_mask, config=base_config, save_output=False)
        assert "high_risk_obesity" in result_df.columns

    def test_high_risk_is_binary(self, panel_df, base_config):
        train_mask = panel_df["YearStart"] <= 2017
        result_df, _ = build_targets(panel_df, train_mask=train_mask, config=base_config, save_output=False)
        unique_vals = set(result_df["high_risk_obesity"].dropna().unique())
        assert unique_vals <= {0, 1}, f"high_risk_obesity must be binary, got: {unique_vals}"

    def test_threshold_from_train_only(self, panel_df, base_config):
        """Threshold should equal the P75 of training Data_Value, not full dataset."""
        train_mask = panel_df["YearStart"] <= 2017
        _, meta = build_targets(panel_df, train_mask=train_mask, config=base_config, save_output=False)
        train_vals = panel_df.loc[train_mask, "Data_Value"].dropna()
        expected_threshold = float(np.percentile(train_vals, 75))
        assert abs(meta["high_risk_threshold"] - expected_threshold) < 1e-6, \
            "High-risk threshold must be computed on training data only"

    def test_metadata_keys_present(self, panel_df, base_config):
        train_mask = panel_df["YearStart"] <= 2017
        _, meta = build_targets(panel_df, train_mask=train_mask, config=base_config, save_output=False)
        for key in ("high_risk_threshold", "high_risk_percentile", "early_warning_built"):
            assert key in meta, f"Missing metadata key: {key}"


# ---------------------------------------------------------------------------
# Tests: _build_early_warning
# ---------------------------------------------------------------------------


class TestEarlyWarning:
    def test_early_warning_built_with_enough_pairs(self, panel_df, base_config):
        """Construct high_risk_obesity so some groups will have transitions."""
        panel_df = panel_df.copy()
        panel_df["high_risk_obesity"] = (panel_df["Data_Value"] >= 32.0).astype(int)
        result_df, success, n_pairs = _build_early_warning(
            df=panel_df,
            group_key=["LocationAbbr", "StratificationCategory1", "Stratification1"],
            high_risk_col="high_risk_obesity",
            ew_col="early_warning",
            min_pairs=2,
        )
        assert success, "Early warning should be built when enough pairs exist"
        assert "early_warning" in result_df.columns

    def test_early_warning_skipped_with_too_few_pairs(self, panel_df):
        """When min_pairs is much larger than available pairs, fallback should trigger."""
        panel_df = panel_df.copy()
        panel_df["high_risk_obesity"] = 0  # nobody transitions
        result_df, success, n_pairs = _build_early_warning(
            df=panel_df,
            group_key=["LocationAbbr", "StratificationCategory1", "Stratification1"],
            high_risk_col="high_risk_obesity",
            ew_col="early_warning",
            min_pairs=9999,  # impossible to reach
        )
        assert not success, "Early warning should not be built with too few pairs"


# ---------------------------------------------------------------------------
# Tests: get_feature_columns
# ---------------------------------------------------------------------------


class TestGetFeatureColumns:
    def test_excludes_key_identifiers(self):
        df = pd.DataFrame(columns=[
            "YearStart", "LocationAbbr", "stratum_category", "stratum_value",
            "lag_1", "region_enc", "stratum_value_enc",
        ])
        result = get_feature_columns(df)
        assert "YearStart" not in result
        assert "LocationAbbr" not in result
        assert "stratum_category" not in result
        assert "stratum_value" not in result

    def test_includes_numeric_features(self):
        df = pd.DataFrame(columns=[
            "YearStart", "LocationAbbr", "stratum_category", "stratum_value",
            "lag_1", "rolling_mean_3", "region_enc",
        ])
        result = get_feature_columns(df)
        assert "lag_1" in result
        assert "rolling_mean_3" in result
        assert "region_enc" in result
