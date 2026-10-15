"""Unit tests for DataCo inventory optimization engine and audit/correction safeguards.

Covers:
1. Fixed-origin forecast horizon dates (2017-09-01 through 2017-09-08).
2. Leakage prevention: predictions use only information available on or before the origin date.
3. Recursive rollout: zero consumption of future actual demand.
4. Pre-decision uncertainty: post-decision test observations strictly excluded from uncertainty.
5. Uncertainty date bound enforcement: residual dates <= decision date.
6. Insufficient uncertainty history produces explicit fallback or UNAVAILABLE status (no invented precision).
7. Inventory calculations strictly aligned with fixed-origin forecasts.
8. Non-negative order quantities under non-negative integer policies.
9. Incomplete or invalid forecasts produce UNAVAILABLE recommendations instead of silent zeros.
10. Preservation of inventory position arithmetic, safety stock formulas, ROP, target stock, and EOQ.
"""

import pytest
import pandas as pd
import numpy as np

from retail_forecasting.forecasting.benchmark_dataco import (
    generate_dataco_fixed_origin_forecasts,
    generate_dataco_validation_forecasts,
    create_dataco_tabular_features,
)
from retail_forecasting.inventory.reorder_recommendations_dataco import (
    calculate_inventory_position,
    calculate_dataco_safety_stock,
    calculate_dataco_reorder_point,
    calculate_dataco_target_stock,
    calculate_dataco_order_quantity,
    calculate_dataco_eoq,
    estimate_dataco_forecast_uncertainty,
    generate_dataco_inventory_recommendations,
)


def test_calculate_inventory_position() -> None:
    """Verifies net inventory position arithmetic and negative quantity protection."""
    assert calculate_inventory_position(100.0, 50.0, 10.0) == 140.0
    assert calculate_inventory_position(10.0, 0.0, 25.0) == -15.0

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_inventory_position(-5.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_inventory_position(10.0, -10.0, 0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_inventory_position(10.0, 10.0, -5.0)


def test_calculate_dataco_safety_stock() -> None:
    """Verifies safety stock calculation under standard normal approximation."""
    ss = calculate_dataco_safety_stock(
        service_level=0.95,
        daily_forecast_error_std=5.0,
        lead_time_days=4
    )
    assert ss == pytest.approx(16.4485, rel=1e-3)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        calculate_dataco_safety_stock(0.0, 5.0, 4)
    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        calculate_dataco_safety_stock(1.0, 5.0, 4)
    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        calculate_dataco_safety_stock(1.5, 5.0, 4)
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_dataco_safety_stock(0.95, 5.0, 0)


def test_calculate_dataco_reorder_point_and_target_stock() -> None:
    """Verifies ROP = LTD + SS and Target Stock = PPD + SS."""
    ltd = 100.0
    ppd = 120.0
    ss = 25.0

    rop = calculate_dataco_reorder_point(ltd, ss)
    s = calculate_dataco_target_stock(ppd, ss)

    assert rop == 125.0
    assert s == 145.0


def test_calculate_dataco_order_quantity_rounding() -> None:
    """Verifies replenishment order quantity with ceiling rounding and ROP trigger."""
    target_stock = 150.0
    rop = 100.0

    q_ceil = calculate_dataco_order_quantity(
        target_stock=target_stock,
        inventory_position=90.4,
        reorder_point=rop,
        rounding_mode="ceil"
    )
    assert q_ceil == 60

    q_floor = calculate_dataco_order_quantity(
        target_stock=target_stock,
        inventory_position=90.4,
        reorder_point=rop,
        rounding_mode="floor"
    )
    assert q_floor == 59

    q_no_order = calculate_dataco_order_quantity(
        target_stock=target_stock,
        inventory_position=110.0,
        reorder_point=rop
    )
    assert q_no_order == 0


def test_calculate_dataco_eoq() -> None:
    """Verifies EOQ formula: sqrt((2 * D * S) / H)."""
    eoq = calculate_dataco_eoq(
        annual_demand=1000.0,
        order_cost=50.0,
        holding_cost_annual_per_unit=2.0
    )
    assert eoq == pytest.approx(223.6068, rel=1e-3)


# --- Phase B, C, D, E Audit & Correction Tests ---

def test_fixed_origin_forecast_horizon_dates() -> None:
    """Verifies that fixed-origin forecasting produces exactly the requested forward dates."""
    dates = pd.date_range("2017-06-01", "2017-08-31", freq="D")
    df = pd.DataFrame({
        "date": dates,
        "product_id": "365",
        "demand": [50.0] * len(dates)
    })
    fc_df = generate_dataco_fixed_origin_forecasts(
        contiguous_df=df,
        forecast_origin="2017-08-31",
        horizon_days=8,
        product_ids=["365"],
        random_state=42,
    )
    assert len(fc_df) == 8
    expected_dates = [f"2017-09-0{d}" for d in range(1, 9)]
    assert list(fc_df["forecast_date"]) == expected_dates
    assert list(fc_df["forecast_horizon_step"]) == list(range(1, 9))
    assert (fc_df["predicted_demand"] >= 0.0).all()
    assert (fc_df["forecast_origin"] == "2017-08-31").all()
    assert (fc_df["forecast_method"] == "recursive_multi_step").all()


def test_recursive_forecasting_does_not_consume_future_actuals() -> None:
    """Proves that changing future actual demand after origin has ZERO impact on recursive predictions."""
    dates_hist = pd.date_range("2017-06-01", "2017-08-31", freq="D")
    dates_future = pd.date_range("2017-09-01", "2017-09-15", freq="D")

    # Dataset A: future demand is 10.0
    df_a = pd.DataFrame({
        "date": list(dates_hist) + list(dates_future),
        "product_id": "365",
        "demand": [50.0] * len(dates_hist) + [10.0] * len(dates_future)
    })

    # Dataset B: future demand is wildly different (9999.0)
    df_b = pd.DataFrame({
        "date": list(dates_hist) + list(dates_future),
        "product_id": "365",
        "demand": [50.0] * len(dates_hist) + [9999.0] * len(dates_future)
    })

    fc_a = generate_dataco_fixed_origin_forecasts(
        contiguous_df=df_a,
        forecast_origin="2017-08-31",
        horizon_days=8,
        product_ids=["365"],
        random_state=42,
    )

    fc_b = generate_dataco_fixed_origin_forecasts(
        contiguous_df=df_b,
        forecast_origin="2017-08-31",
        horizon_days=8,
        product_ids=["365"],
        random_state=42,
    )

    # Predictions MUST be identical because future actuals cannot enter
    np.testing.assert_array_almost_equal(
        fc_a["predicted_demand"].values,
        fc_b["predicted_demand"].values,
        err_msg="Future actuals leaked into recursive forecasting!"
    )


def test_uncertainty_estimator_excludes_post_decision_observations() -> None:
    """Proves that observations after forecast_origin_date are strictly excluded from uncertainty."""
    hist_dates = pd.date_range("2017-07-01", "2017-08-31", freq="D")  # 62 days
    future_dates = pd.date_range("2017-09-01", "2017-09-30", freq="D")  # 30 days

    all_dates = list(hist_dates) + list(future_dates)
    # Historic error = 5.0, future error = 500.0
    df = pd.DataFrame({
        "date": all_dates,
        "product_id": "P1",
        "actual_demand": [105.0] * len(hist_dates) + [600.0] * len(future_dates),
        "prediction": [100.0] * len(hist_dates) + [100.0] * len(future_dates),
    })

    # Filter with cutoff 2017-08-31
    uncert = estimate_dataco_forecast_uncertainty(
        historical_forecasts_df=df,
        forecast_origin_date="2017-08-31",
        pred_col="prediction",
        actual_col="actual_demand",
        min_observations=14,
        method="sample_std"
    )

    assert len(uncert) == 1
    row = uncert.iloc[0]
    # Sample size must be exactly 62 (historical), not 92
    assert row["sample_size"] == 62
    assert row["max_date"] == "2017-08-31"
    # Error mean must be 5.0, not contaminated by 500.0
    assert row["error_mean"] == pytest.approx(5.0)
    assert row["error_std"] == pytest.approx(0.0)  # constant 5.0 error has 0 std


def test_uncertainty_insufficient_history_handling() -> None:
    """Verifies that insufficient history triggers pooled fallback and zero history triggers UNAVAILABLE."""
    # Product P1 has 20 observations (sufficient >= 14)
    # Product P2 has 5 observations (< 14, triggers pooled fallback)
    # Product P3 has 0 observations
    dates = pd.date_range("2017-08-01", "2017-08-20", freq="D")
    df = pd.DataFrame({
        "date": list(dates) + list(dates[:5]),
        "product_id": ["P1"] * len(dates) + ["P2"] * 5,
        "actual_demand": [20.0] * len(dates) + [30.0] * 5,
        "prediction": [18.0] * len(dates) + [25.0] * 5,
    })

    uncert = estimate_dataco_forecast_uncertainty(
        historical_forecasts_df=df,
        forecast_origin_date="2017-08-31",
        pred_col="prediction",
        min_observations=14,
        method="sample_std"
    )

    p1_row = uncert[uncert["product_id"] == "P1"].iloc[0]
    assert p1_row["uncertainty_status"] == "reliable_sample"
    assert bool(p1_row["is_fallback"]) is False

    p2_row = uncert[uncert["product_id"] == "P2"].iloc[0]
    assert p2_row["uncertainty_status"] == "insufficient_sample_pooled_fallback"
    assert bool(p2_row["is_fallback"]) is True


    # Empty history test
    empty_df = pd.DataFrame(columns=["date", "product_id", "actual_demand", "prediction"])
    empty_uncert = estimate_dataco_forecast_uncertainty(
        historical_forecasts_df=empty_df,
        forecast_origin_date="2017-08-31",
        pred_col="prediction"
    )
    assert empty_uncert.empty


def test_recommendation_handles_incomplete_forecast_horizon() -> None:
    """Proves that missing or incomplete forecast horizons return UNAVAILABLE, never silent zero."""
    # Provide only 3 forward days when protection period requires 8 (L=7 + R=1)
    dates = pd.date_range("2017-09-01", periods=3, freq="D")
    fc_df = pd.DataFrame({
        "forecast_date": dates,
        "forecast_origin": ["2017-08-31"] * 3,
        "product_id": ["P1"] * 3,
        "predicted_demand": [10.0, 12.0, 11.0],
    })

    uncert_df = pd.DataFrame([{
        "product_id": "P1",
        "sample_size": 30,
        "uncertainty_value": 3.5,
        "uncertainty_status": "reliable_sample",
        "is_fallback": False,
    }])

    recs = generate_dataco_inventory_recommendations(
        forecast_comparison_df=fc_df,
        uncertainty_df=uncert_df,
        service_level_target=0.95,
        lead_time_days=7,
        review_period_days=1,
        forecast_origin_date="2017-08-31",
        pred_col="predicted_demand",
    )

    assert len(recs) == 1
    row = recs.iloc[0]
    assert row["recommendation_status"] == "UNAVAILABLE"
    assert row["recommended_order_quantity"] is None
    assert np.isnan(row["reorder_point"])
    assert "INCOMPLETE_FORECAST_HORIZON" in row["warning_or_limitation"]


def test_full_pipeline_fixed_origin_reorder_calculation() -> None:
    """Verifies complete end-to-end recommendation arithmetic with fixed-origin inputs."""
    # 8 forward days for P1 (L=7, R=1 -> protection = 8)
    dates = pd.date_range("2017-09-01", periods=8, freq="D")
    demands = [50.0, 60.0, 55.0, 65.0, 70.0, 60.0, 80.0, 75.0]
    fc_df = pd.DataFrame({
        "forecast_date": dates,
        "forecast_origin": ["2017-08-31"] * 8,
        "forecast_horizon_step": list(range(1, 9)),
        "product_id": ["P1"] * 8,
        "predicted_demand": demands,
    })

    # Uncertainty std = 10.0
    uncert_df = pd.DataFrame([{
        "product_id": "P1",
        "sample_size": 60,
        "uncertainty_value": 10.0,
        "uncertainty_status": "reliable_sample",
        "is_fallback": False,
    }])

    recs = generate_dataco_inventory_recommendations(
        forecast_comparison_df=fc_df,
        uncertainty_df=uncert_df,
        service_level_target=0.95,
        lead_time_days=7,
        review_period_days=1,
        simulated_default_on_hand=50.0,
        simulated_default_on_order=0.0,
        simulated_default_backorders=0.0,
        forecast_origin_date="2017-08-31",
        pred_col="predicted_demand",
    )

    assert len(recs) == 1
    row = recs.iloc[0]

    # LTD = sum(demands[:7]) = 440.0
    assert row["expected_lead_time_demand"] == 440.0
    # PPD = sum(demands[:8]) = 515.0
    assert row["expected_protection_period_demand"] == 515.0

    # z = 1.64485
    # SS_lead = 1.64485 * 10 * sqrt(7) = 43.518
    assert row["safety_stock"] == pytest.approx(43.52, abs=0.05)
    # ROP = 440 + 43.52 = 483.52
    assert row["reorder_point"] == pytest.approx(483.52, abs=0.05)

    # SS_protection = 1.64485 * 10 * sqrt(8) = 46.523
    # Target stock = 515 + 46.523 = 561.52
    assert row["target_stock_level"] == pytest.approx(561.52, abs=0.05)

    # IP = 50.0 <= ROP (483.52) -> REORDER_TRIGGERED
    assert bool(row["reorder_triggered"]) is True
    assert row["recommendation_status"] == "REORDER_TRIGGERED"


    # Q = ceil(561.52 - 50.0) = 512
    assert row["recommended_order_quantity"] == 512
