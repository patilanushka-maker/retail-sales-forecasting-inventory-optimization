"""Automated unit and integration tests for time-series feature engineering.

Verifies:
1. Correct lag_1 values matching t-1.
2. Correct lag_7 values matching t-7.
3. Correct lag_14 and lag_28 values matching t-14 and t-28.
4. Rolling means strictly exclude target date t (target shifted by 1 row).
5. Rolling standard deviations strictly exclude target date t.
6. Features are calculated independently for distinct products without cross-contamination.
7. Input series is sorted chronologically prior to feature calculation.
8. Duplicate product-date rows raise ValueError.
9. Missing lag history is retained as NaN (not silently zero-filled).
10. Calendar features (day_of_week, month, day_of_month, is_weekend) are correct.
11. Input DataFrame is not mutated in-place.
12. Empty input is handled gracefully without crashing.
13. Missing columns or invalid dates raise clear, informative errors.
14. Feature pipeline output matches documented schema and marks model eligibility.
15. Product-history diagnostics accurately distinguish calendar span from positive-demand days.
16. Strict Leakage Test: Modifying demand on a future date leaves all prior feature values unchanged.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from retail_forecasting.features.build_features import (
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    build_feature_pipeline,
    analyze_product_histories,
)


@pytest.fixture
def single_product_demand() -> pd.DataFrame:
    """Fixture providing 40 days of continuous demand for a single product."""
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    # Demand pattern: day index + 10 (e.g. 10, 11, 12, ..., 49)
    demand = [float(i + 10) for i in range(40)]
    return pd.DataFrame({
        "date": dates,
        "product_id": ["SKU_A"] * 40,
        "demand": demand,
    })


@pytest.fixture
def multi_product_demand() -> pd.DataFrame:
    """Fixture providing 35 days of demand for two distinct products."""
    dates = pd.date_range("2021-01-01", periods=35, freq="D")
    df_a = pd.DataFrame({
        "date": dates,
        "product_id": ["SKU_A"] * 35,
        "demand": [10.0] * 35,  # constant 10
    })
    df_b = pd.DataFrame({
        "date": dates,
        "product_id": ["SKU_B"] * 35,
        "demand": [100.0] * 35,  # constant 100
    })
    return pd.concat([df_a, df_b], ignore_index=True)


# =========================================================================
# 1, 2, 3: Lag Feature Tests
# =========================================================================

def test_lag_1_values_match_previous_day(single_product_demand: pd.DataFrame) -> None:
    """lag_1 at row index i must strictly equal demand at row index i-1."""
    df_lags = create_lag_features(single_product_demand, lags=[1])
    assert pd.isna(df_lags.loc[0, "lag_1"])
    for i in range(1, len(df_lags)):
        assert df_lags.loc[i, "lag_1"] == single_product_demand.loc[i - 1, "demand"]


def test_lag_7_values_match_seven_days_prior(single_product_demand: pd.DataFrame) -> None:
    """lag_7 at row index i must strictly equal demand at row index i-7."""
    df_lags = create_lag_features(single_product_demand, lags=[7])
    # First 7 rows (0 to 6) must be NaN
    assert df_lags.loc[:6, "lag_7"].isna().all()
    for i in range(7, len(df_lags)):
        assert df_lags.loc[i, "lag_7"] == single_product_demand.loc[i - 7, "demand"]


def test_lag_14_and_lag_28_values(single_product_demand: pd.DataFrame) -> None:
    """lag_14 and lag_28 must match observations 14 and 28 days prior."""
    df_lags = create_lag_features(single_product_demand, lags=[14, 28])
    assert df_lags.loc[:13, "lag_14"].isna().all()
    assert df_lags.loc[:27, "lag_28"].isna().all()
    assert df_lags.loc[14, "lag_14"] == single_product_demand.loc[0, "demand"]
    assert df_lags.loc[28, "lag_28"] == single_product_demand.loc[0, "demand"]
    assert df_lags.loc[35, "lag_28"] == single_product_demand.loc[7, "demand"]


# =========================================================================
# 4, 5: Rolling Statistics & Leakage Prevention Tests
# =========================================================================

def test_rolling_mean_excludes_target_date(single_product_demand: pd.DataFrame) -> None:
    """Rolling mean at date t must strictly exclude demand at date t."""
    df_rolling = create_rolling_features(
        single_product_demand,
        windows=[7],
        min_periods={7: 7}
    )
    # Row 0 to 6 have fewer than 7 prior observations (since target date is excluded)
    assert df_rolling.loc[:6, "rolling_mean_7"].isna().all()

    # At row 7, prior 7 observations are rows 0 to 6 (demands: 10, 11, 12, 13, 14, 15, 16)
    expected_mean_row_7 = np.mean([10, 11, 12, 13, 14, 15, 16])  # 13.0
    assert df_rolling.loc[7, "rolling_mean_7"] == pytest.approx(expected_mean_row_7)

    # Crucial check: row 7 demand is 17.0. If leakage occurred, mean would be 14.0.
    assert df_rolling.loc[7, "rolling_mean_7"] != pytest.approx(14.0)


def test_rolling_std_excludes_target_date(single_product_demand: pd.DataFrame) -> None:
    """Rolling std at date t must strictly exclude demand at date t."""
    df_rolling = create_rolling_features(
        single_product_demand,
        windows=[7],
        min_periods={7: 7}
    )
    assert df_rolling.loc[:6, "rolling_std_7"].isna().all()

    expected_std_row_7 = np.std([10, 11, 12, 13, 14, 15, 16], ddof=1)
    assert df_rolling.loc[7, "rolling_std_7"] == pytest.approx(expected_std_row_7)


# =========================================================================
# 6: Independent Calculation Across Products
# =========================================================================

def test_features_calculated_independently_per_product(multi_product_demand: pd.DataFrame) -> None:
    """Features for SKU_A must never use data from SKU_B, and vice versa."""
    pipeline_out = build_feature_pipeline(multi_product_demand, lags=[1], windows=[7])

    sku_a_out = pipeline_out[pipeline_out["product_id"] == "SKU_A"].reset_index(drop=True)
    sku_b_out = pipeline_out[pipeline_out["product_id"] == "SKU_B"].reset_index(drop=True)

    # SKU_A constant demand is 10.0; lag_1 must be 10.0 and rolling_mean_7 must be 10.0
    assert sku_a_out.loc[1:, "lag_1"].dropna().unique() == [10.0]
    assert sku_a_out.loc[7:, "rolling_mean_7"].dropna().unique() == [10.0]

    # SKU_B constant demand is 100.0; lag_1 must be 100.0 and rolling_mean_7 must be 100.0
    assert sku_b_out.loc[1:, "lag_1"].dropna().unique() == [100.0]
    assert sku_b_out.loc[7:, "rolling_mean_7"].dropna().unique() == [100.0]


# =========================================================================
# 7, 8: Sorting & Duplicate Detection
# =========================================================================

def test_chronological_sorting_before_feature_calculation(single_product_demand: pd.DataFrame) -> None:
    """Input shuffled out of order must be sorted chronologically before lagging."""
    shuffled = single_product_demand.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_lags = create_lag_features(shuffled, lags=[1])

    # Result should be sorted by date
    assert df_lags["date"].is_monotonic_increasing
    assert df_lags.loc[1, "lag_1"] == single_product_demand.loc[0, "demand"]


def test_duplicate_product_date_raises_error() -> None:
    """Duplicate (product_id, date) rows must raise a descriptive ValueError."""
    df_dup = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-01")],
        "product_id": ["SKU_A", "SKU_A"],
        "demand": [10.0, 15.0],
    })
    with pytest.raises(ValueError, match="Detected 1 duplicate entries"):
        create_lag_features(df_dup, lags=[1])


# =========================================================================
# 9: Missing History Retention
# =========================================================================

def test_missing_lag_history_not_silently_converted_to_zero(single_product_demand: pd.DataFrame) -> None:
    """Missing historical values must remain NaN and not be converted to 0.0."""
    df_lags = create_lag_features(single_product_demand, lags=[1, 7, 14, 28])
    assert pd.isna(df_lags.loc[0, "lag_1"])
    assert pd.isna(df_lags.loc[0, "lag_7"])
    assert pd.isna(df_lags.loc[0, "lag_14"])
    assert pd.isna(df_lags.loc[0, "lag_28"])


# =========================================================================
# 10: Calendar Features Correctness
# =========================================================================

def test_calendar_features_accuracy() -> None:
    """Verify exact day_of_week, month, day_of_month, and is_weekend mappings."""
    # 2021-01-01 was a Friday (dayofweek = 4, weekend = 0)
    # 2021-01-02 was a Saturday (dayofweek = 5, weekend = 1)
    # 2021-01-03 was a Sunday (dayofweek = 6, weekend = 1)
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-02"),
            pd.Timestamp("2021-01-03"),
        ]
    })
    cal = create_calendar_features(df)
    assert list(cal["day_of_week"]) == [4, 5, 6]
    assert list(cal["month"]) == [1, 1, 1]
    assert list(cal["day_of_month"]) == [1, 2, 3]
    assert list(cal["is_weekend"]) == [0, 1, 1]


# =========================================================================
# 11, 12, 13: Invariance, Edge Cases & Error Handling
# =========================================================================

def test_input_dataframe_not_mutated(single_product_demand: pd.DataFrame) -> None:
    """The input DataFrame must remain unchanged after feature pipeline execution."""
    original_copy = single_product_demand.copy(deep=True)
    _ = build_feature_pipeline(single_product_demand)
    pd.testing.assert_frame_equal(single_product_demand, original_copy)


def test_empty_input_handled_gracefully() -> None:
    """Empty DataFrame must return an empty DataFrame with expected feature columns."""
    empty_df = pd.DataFrame(columns=["date", "product_id", "demand"])
    out = build_feature_pipeline(empty_df)
    assert out.empty
    assert "lag_1" in out.columns
    assert "rolling_mean_7" in out.columns
    assert "day_of_week" in out.columns


def test_missing_column_raises_error() -> None:
    """Missing required column raises descriptive KeyError."""
    df = pd.DataFrame({"product_id": ["SKU_1"], "demand": [10.0]})
    with pytest.raises(KeyError, match="date"):
        create_lag_features(df, lags=[1])


# =========================================================================
# 14, 15: Schema & Product History Diagnostics
# =========================================================================

def test_feature_pipeline_output_schema_and_eligibility(single_product_demand: pd.DataFrame) -> None:
    """Pipeline output must include all lag, rolling, calendar features, and is_model_eligible."""
    out = build_feature_pipeline(single_product_demand)

    expected_cols = [
        "date", "product_id", "demand",
        "day_of_week", "month", "day_of_month", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        "rolling_mean_28", "rolling_std_28",
        "is_model_eligible",
    ]
    for col in expected_cols:
        assert col in out.columns

    # Single product has 40 rows. Rows 0 to 27 have missing lag_28 -> not eligible.
    # Rows 28 to 39 have full 28-day history -> eligible (12 rows).
    assert out["is_model_eligible"].sum() == 12


def test_product_history_diagnostics_accuracy() -> None:
    """Diagnostics must accurately count calendar observations and positive days."""
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=10, freq="D"),
        "product_id": ["TEST_SKU"] * 10,
        "demand": [0.0, 5.0, 0.0, 10.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    })
    diag_df, summary = analyze_product_histories(df, min_history_days=7)
    row = diag_df.iloc[0]

    assert row["calendar_observations"] == 10
    assert row["positive_demand_days"] == 3
    assert row["zero_demand_days"] == 7
    assert bool(row["has_enough_history_for_lag28"]) is False
    assert summary["history_tiers"]["7 to 28 days (short, insufficient for lag_28)"] == 1


# =========================================================================
# 16: Strict Leakage Audit Test
# =========================================================================

def test_future_target_modification_does_not_affect_past_features(
    single_product_demand: pd.DataFrame
) -> None:
    """Strict Leakage Test: Modifying demand at date T_future must NOT alter any feature at t < T_future."""
    pipeline_orig = build_feature_pipeline(single_product_demand)

    # Select cutoff index 25
    cutoff_idx = 25
    features_before_cutoff_orig = pipeline_orig.loc[:cutoff_idx - 1].copy()

    # Modify future demand at cutoff_idx and beyond to drastically different values
    tampered_demand = single_product_demand.copy(deep=True)
    tampered_demand.loc[cutoff_idx:, "demand"] = 999999.0

    pipeline_tampered = build_feature_pipeline(tampered_demand)
    features_before_cutoff_tampered = pipeline_tampered.loc[:cutoff_idx - 1].copy()

    # Assert exact equality of all feature values before the cutoff
    pd.testing.assert_frame_equal(features_before_cutoff_orig, features_before_cutoff_tampered)
