"""Comprehensive unit and integration tests for Phase 6 Inventory Optimization Engine.

Verifies:
1. Inventory-position arithmetic (on_hand + on_order - backorders).
2. Rejection of invalid stock inputs (negative, non-finite, non-numeric).
3. Safety-stock calculation against independent analytical formula.
4. Service-level validation (strictly 0 < service_level < 1).
5. Zero forecast-error variability behavior (SS = 0.0).
6. Invalid uncertainty estimates (negative, NaN, Inf).
7. Lead-time demand calculated strictly from the first L forecast days.
8. Rejection of insufficient forecast horizons (H < L).
9. Reorder-point arithmetic (LTD + SS).
10. Target-stock protection-period consistency (P = L + R).
11. No double-counting of lead-time demand in target stock.
12. Order quantity never negative (Q = max(0, S - IP)).
13. Correct rounding to whole units ('ceil', 'round', 'floor').
14. EOQ calculation and validation (handling D=0 and cost validation).
15. Forecast-error uncertainty estimates derived from genuine out-of-sample errors.
16. Sparse-history fallback behavior for products with insufficient observations.
17. Missing or non-finite forecast handling (rejects NaN, Inf, negatives).
18. Separation of actual historical data from simulated operational inputs.
19. Reproducible results under fixed random_state.
20. Complete small end-to-end inventory recommendation example with manually verifiable numbers.
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from retail_forecasting.inventory.calculate_stock import (
    calculate_safety_stock,
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_target_stock,
    calculate_order_quantity,
    calculate_economic_order_quantity,
)
from retail_forecasting.inventory.reorder_recommendations import (
    inventory_position,
    evaluate_inventory_position,
    estimate_forecast_uncertainty,
    generate_inventory_recommendations,
    generate_replenishment_orders,
)


# 1. Inventory-position arithmetic
def test_inventory_position_arithmetic() -> None:
    """Verify net inventory position formula: IP = on_hand + on_order - backorders."""
    # Standard positive stock
    assert inventory_position(on_hand=100.0, on_order=50.0, backorders=20.0) == 130.0
    # Zero in transit and backorders
    assert inventory_position(on_hand=75.0, on_order=0.0, backorders=0.0) == 75.0
    # Net negative inventory position when backorders exceed stock
    assert inventory_position(on_hand=10.0, on_order=5.0, backorders=30.0) == -15.0
    # Alias compatibility
    assert evaluate_inventory_position(100.0, 50.0, 20.0) == 130.0


# 2. Rejection of invalid stock inputs
def test_rejection_of_invalid_stock_inputs() -> None:
    """Verify rejection of negative, non-finite, and non-numeric stock inputs."""
    with pytest.raises(ValueError, match="cannot be negative"):
        inventory_position(on_hand=-5.0, on_order=10.0, backorders=0.0)

    with pytest.raises(ValueError, match="cannot be negative"):
        inventory_position(on_hand=10.0, on_order=-1.0, backorders=0.0)

    with pytest.raises(ValueError, match="cannot be negative"):
        inventory_position(on_hand=10.0, on_order=0.0, backorders=-10.0)

    with pytest.raises(ValueError, match="must be finite"):
        inventory_position(on_hand=float("nan"), on_order=10.0)

    with pytest.raises(ValueError, match="must be finite"):
        inventory_position(on_hand=10.0, on_order=float("inf"))

    with pytest.raises(TypeError, match="numeric"):
        inventory_position(on_hand="100", on_order=10.0)


# 3. Safety-stock calculation against an independently calculated example
def test_safety_stock_calculation_independent() -> None:
    """Validate safety stock against independent analytical formula: SS = z * sigma * sqrt(L)."""
    service_level = 0.95
    sigma_d = 4.0
    lead_time = 9.0

    # Independent hand calculation
    z_expected = norm.ppf(0.95)  # 1.6448536269514722
    expected_ss = z_expected * sigma_d * math.sqrt(lead_time)  # 1.6448536... * 4 * 3 = 19.7382435...

    actual_ss = calculate_safety_stock(
        service_level=service_level,
        daily_forecast_error_std=sigma_d,
        lead_time_days=lead_time,
    )
    assert pytest.approx(actual_ss, rel=1e-6) == expected_ss


# 4. Service-level validation
def test_service_level_validation() -> None:
    """Verify service level must be strictly between 0 and 1 exclusive."""
    # Boundary and out-of-range values
    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        calculate_safety_stock(service_level=0.0, daily_forecast_error_std=2.0, lead_time_days=7)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        calculate_safety_stock(service_level=1.0, daily_forecast_error_std=2.0, lead_time_days=7)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        calculate_safety_stock(service_level=-0.05, daily_forecast_error_std=2.0, lead_time_days=7)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        calculate_safety_stock(service_level=1.05, daily_forecast_error_std=2.0, lead_time_days=7)


# 5. Zero forecast-error variability behavior
def test_zero_forecast_error_variability_behavior() -> None:
    """Verify that zero forecast error volatility results in zero safety stock."""
    ss = calculate_safety_stock(
        service_level=0.99,
        daily_forecast_error_std=0.0,
        lead_time_days=14,
    )
    assert ss == 0.0


# 6. Invalid uncertainty estimates
def test_invalid_uncertainty_estimates() -> None:
    """Verify rejection of negative or non-finite forecast error standard deviations."""
    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_safety_stock(service_level=0.95, daily_forecast_error_std=-1.5, lead_time_days=7)

    with pytest.raises(ValueError, match="must be finite"):
        calculate_safety_stock(service_level=0.95, daily_forecast_error_std=float("nan"), lead_time_days=7)

    with pytest.raises(ValueError, match="must be finite"):
        calculate_safety_stock(service_level=0.95, daily_forecast_error_std=float("inf"), lead_time_days=7)


# 7. Lead-time demand calculated from correct forecast days
def test_lead_time_demand_correct_days() -> None:
    """Verify that LTD sums strictly the first L days and ignores subsequent days."""
    forecasts = [10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    lead_time = 5
    # Expected sum: 10 + 12 + 15 + 20 + 25 = 82.0
    ltd = calculate_lead_time_demand(forecasts, lead_time_days=lead_time)
    assert ltd == 82.0


# 8. Rejection of insufficient forecast horizons
def test_rejection_of_insufficient_forecast_horizons() -> None:
    """Verify that an error is raised when forecast horizon is shorter than lead time."""
    short_forecasts = [10.0, 12.0, 15.0]
    with pytest.raises(ValueError, match="shorter than required lead time"):
        calculate_lead_time_demand(short_forecasts, lead_time_days=7)


# 9. Reorder-point arithmetic
def test_reorder_point_arithmetic() -> None:
    """Verify ROP = LTD + SS."""
    rop = calculate_reorder_point(lead_time_demand=82.0, safety_stock=18.5)
    assert rop == 100.5

    # Backwards-compatible mean * L + SS
    rop_mean = calculate_reorder_point(mean_daily_demand=10.0, lead_time_days=7.0, safety_stock=15.0)
    assert rop_mean == 85.0


# 10. Target-stock protection-period consistency
def test_target_stock_protection_period_consistency() -> None:
    """Verify target stock (S) over protection period P = L + R."""
    lead_time = 7
    review_period = 1
    protection_period = lead_time + review_period  # 8 days

    forecasts = [10.0] * 10
    pp_demand = float(np.sum(forecasts[:protection_period]))  # 80.0
    ss_pp = 20.0

    target_stock = calculate_target_stock(protection_period_demand=pp_demand, safety_stock=ss_pp)
    assert target_stock == 100.0


# 11. No double-counting of lead-time demand
def test_no_double_counting_of_lead_time_demand() -> None:
    """Verify that target stock does not erroneously add LTD and protection period demand together."""
    lead_time_demand = 70.0
    protection_period_demand = 80.0
    safety_stock = 15.0

    target_stock = calculate_target_stock(
        protection_period_demand=protection_period_demand,
        safety_stock=safety_stock,
    )
    # Target stock must be 95.0, NOT 70 + 80 + 15 = 165.0
    assert target_stock == 95.0
    assert target_stock != (lead_time_demand + protection_period_demand + safety_stock)


# 12. Order quantity never negative
def test_order_quantity_never_negative() -> None:
    """Verify that order quantity is zero when inventory position exceeds target stock."""
    # When IP (120) > target stock (100), net deficit is -20 -> Q must be 0
    q = calculate_order_quantity(target_stock=100.0, inventory_position=120.0)
    assert q == 0
    assert isinstance(q, int)


# 13. Correct rounding to whole units
def test_correct_rounding_to_whole_units() -> None:
    """Verify that ceil rounding correctly protects safety stock buffer."""
    target_stock = 100.2
    inventory_pos = 80.0
    # Deficit = 20.2 -> ceil is 21, round is 20, floor is 20
    assert calculate_order_quantity(target_stock, inventory_pos, rounding_mode="ceil") == 21
    assert calculate_order_quantity(target_stock, inventory_pos, rounding_mode="round") == 20
    assert calculate_order_quantity(target_stock, inventory_pos, rounding_mode="floor") == 20


# 14. EOQ calculation and validation
def test_eoq_calculation_and_validation() -> None:
    """Verify classical Wilson EOQ = sqrt(2*D*S / H) and input edge cases."""
    # D = 1000 units/year, S = $50/order, H = $2.5/unit/year
    # EOQ = sqrt(2 * 1000 * 50 / 2.5) = sqrt(40000) = 200.0
    eoq = calculate_economic_order_quantity(
        annual_demand=1000.0,
        order_cost=50.0,
        annual_holding_cost_per_unit=2.5,
    )
    assert pytest.approx(eoq) == 200.0

    # Zero annual demand returns 0.0
    assert calculate_economic_order_quantity(0.0, 50.0, 2.5) == 0.0

    # Invalid cost parameters
    with pytest.raises(ValueError, match="order_cost must be strictly positive"):
        calculate_economic_order_quantity(1000.0, 0.0, 2.5)

    with pytest.raises(ValueError, match="annual_holding_cost_per_unit must be strictly positive"):
        calculate_economic_order_quantity(1000.0, 50.0, -1.0)


# 15. Forecast-error estimates derived from out-of-sample errors
def test_forecast_error_estimates_out_of_sample() -> None:
    """Verify uncertainty estimation calculates sample std (ddof=1) of out-of-sample errors."""
    records = []
    # 10 days for SKU_A
    actuals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    preds =   [12.0, 18.0, 33.0, 38.0, 52.0, 58.0, 73.0, 78.0, 92.0,  98.0]
    for d, (a, p) in enumerate(zip(actuals, preds)):
        records.append({
            "product_id": "SKU_A",
            "date": pd.to_datetime(f"2021-01-{d+1:02d}"),
            "actual_demand": a,
            "random_forest_pred": p,
        })
    df = pd.DataFrame(records)

    errors = np.array(actuals) - np.array(preds)
    expected_std = float(np.std(errors, ddof=1))

    unc_df = estimate_forecast_uncertainty(
        df,
        actual_col="actual_demand",
        pred_col="random_forest_pred",
        product_col="product_id",
        min_observations=7,
    )
    sku_row = unc_df[unc_df["product_id"] == "SKU_A"].iloc[0]
    assert sku_row["error_history_count"] == 10
    assert bool(sku_row["is_uncertainty_fallback"]) is False
    assert pytest.approx(sku_row["forecast_error_std"], rel=1e-3) == expected_std


# 16. Sparse-history fallback behavior
def test_sparse_history_fallback_behavior() -> None:
    """Verify that products with insufficient history (< min_observations) use catalog pooled fallback."""
    # Product 1 has 10 days, Product 2 has only 3 days
    df = pd.DataFrame({
        "product_id": ["P1"] * 10 + ["P2"] * 3,
        "date": pd.date_range("2021-01-01", periods=10).tolist() + pd.date_range("2021-01-01", periods=3).tolist(),
        "actual_demand": [10.0] * 10 + [5.0, 10.0, 15.0],
        "random_forest_pred": [8.0] * 10 + [5.0, 8.0, 12.0],
    })

    unc_df = estimate_forecast_uncertainty(df, min_observations=7, fallback_policy="catalog_pooled")
    p2_row = unc_df[unc_df["product_id"] == "P2"].iloc[0]

    assert p2_row["error_history_count"] == 3
    assert bool(p2_row["is_uncertainty_fallback"]) is True
    assert p2_row["uncertainty_status"] == "CATALOG_POOLED_FALLBACK"
    assert np.isfinite(p2_row["forecast_error_std"])


# 17. Missing or non-finite forecast handling
def test_missing_or_non_finite_forecast_handling() -> None:
    """Verify that daily forecast validation rejects NaN, Inf, and negative forecasts."""
    with pytest.raises(ValueError, match="non-finite"):
        calculate_lead_time_demand([10.0, np.nan, 12.0], lead_time_days=3)

    with pytest.raises(ValueError, match="non-finite"):
        calculate_lead_time_demand([10.0, np.inf, 12.0], lead_time_days=3)

    with pytest.raises(ValueError, match="negative"):
        calculate_lead_time_demand([10.0, -2.0, 12.0], lead_time_days=3)


# 18. Separation of actual historical data from simulated inventory inputs
def test_separation_of_actual_from_simulated_inputs() -> None:
    """Verify that recommendations table explicitly flags simulated operational fields."""
    forecast_df = pd.DataFrame({
        "product_id": ["SKU_X"] * 14,
        "date": pd.date_range("2021-01-01", periods=14),
        "actual_demand": [10.0] * 14,
        "random_forest_pred": [10.0] * 14,
    })

    rec_df = generate_inventory_recommendations(forecast_df, lead_time_days=7, review_period_days=1)
    assert "is_simulated_operational_input" in rec_df.columns
    assert (rec_df["is_simulated_operational_input"] == True).all()

    # Check that simulated stock fields are distinct from forecast demand
    assert "simulated_on_hand" in rec_df.columns
    assert "simulated_on_order" in rec_df.columns
    assert "simulated_backorders" in rec_df.columns
    assert "expected_lead_time_demand" in rec_df.columns


# 19. Reproducible results where randomness is used
def test_reproducible_simulated_results() -> None:
    """Verify deterministic simulation reproducibility when random_state is fixed."""
    forecast_df = pd.DataFrame({
        "product_id": ["SKU_1"] * 14 + ["SKU_2"] * 14,
        "date": pd.date_range("2021-01-01", periods=14).tolist() * 2,
        "actual_demand": [15.0] * 28,
        "random_forest_pred": [14.0] * 28,
    })

    res1 = generate_inventory_recommendations(forecast_df, random_state=42)
    res2 = generate_inventory_recommendations(forecast_df, random_state=42)
    pd.testing.assert_frame_equal(res1, res2)


# 20. A complete small end-to-end inventory recommendation example with manually verifiable expected results
def test_end_to_end_manual_verifiable_recommendation_example() -> None:
    """End-to-end multi-step inventory recommendation with hand-calculated expected values.

    Scenario Parameters:
    - Product: 'BENCHMARK_SKU'
    - Daily forecast for 10 upcoming days: exactly 10.0 units/day.
    - Historical forecast error std sigma_d = 2.0 (pre-evaluated).
    - Lead time L = 7 days.
    - Review period R = 1 day -> Protection period P = 8 days.
    - Service level = 0.95 (z = 1.6448536...).
    - Simulated stock: On-Hand = 50.0, On-Order = 10.0, Backorders = 0.0.

    Hand Calculations:
    1. Inventory Position: IP = 50 + 10 - 0 = 60.0.
    2. Expected Lead-Time Demand (7 days): 7 * 10.0 = 70.0.
    3. Safety Stock (L=7): z * sigma * sqrt(7) = 1.6448536... * 2.0 * 2.6457513... = 8.7037...
    4. Reorder Point: ROP = 70.0 + 8.7037 = 78.7037...
    5. Expected Protection-Period Demand (8 days): 8 * 10.0 = 80.0.
    6. Safety Stock for Protection Period (P=8): z * sigma * sqrt(8) = 1.6448536... * 2.0 * 2.828427... = 9.3047...
    7. Target Stock (S): 80.0 + 9.3047 = 89.3047...
    8. Reorder Triggered: IP (60.0) <= ROP (78.7037) -> TRUE.
    9. Suggested Order Quantity: ceil(89.3047 - 60.0) = ceil(29.3047) = 30 units.
    """
    forecast_df = pd.DataFrame({
        "product_id": ["BENCHMARK_SKU"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "random_forest_pred": [10.0] * 10,
    })

    uncertainty_df = pd.DataFrame([{
        "product_id": "BENCHMARK_SKU",
        "error_history_count": 28,
        "error_mean": 0.0,
        "forecast_error_std": 2.0,
        "is_uncertainty_fallback": False,
        "uncertainty_status": "PRODUCT_SPECIFIC_ESTIMATE",
    }])

    inventory_state_df = pd.DataFrame([{
        "product_id": "BENCHMARK_SKU",
        "on_hand": 50.0,
        "on_order": 10.0,
        "backorders": 0.0,
    }])

    recs = generate_inventory_recommendations(
        forecast_df=forecast_df,
        inventory_state_df=inventory_state_df,
        uncertainty_df=uncertainty_df,
        service_level=0.95,
        lead_time_days=7,
        review_period_days=1,
        rounding_mode="ceil",
    )

    assert len(recs) == 1
    row = recs.iloc[0]

    z_val = norm.ppf(0.95)
    expected_ss_lt = z_val * 2.0 * math.sqrt(7)
    expected_rop = 70.0 + expected_ss_lt
    expected_ss_pp = z_val * 2.0 * math.sqrt(8)
    expected_target_stock = 80.0 + expected_ss_pp
    expected_order_qty = math.ceil(expected_target_stock - 60.0)  # 30

    assert row["inventory_position"] == 60.0
    assert row["expected_lead_time_demand"] == 70.0
    assert pytest.approx(row["safety_stock"], rel=1e-3) == expected_ss_lt
    assert pytest.approx(row["reorder_point"], rel=1e-3) == expected_rop
    assert pytest.approx(row["expected_protection_period_demand"], rel=1e-3) == 80.0
    assert pytest.approx(row["target_stock"], rel=1e-3) == expected_target_stock
    assert bool(row["reorder_triggered"]) is True
    assert row["suggested_order_qty"] == 30
    assert row["recommendation_status"] == "REORDER_NOW"

    # Also verify that generate_replenishment_orders filters this order
    orders = generate_replenishment_orders(recs)
    assert len(orders) == 1
    assert orders.iloc[0]["suggested_order_qty"] == 30


# 21. Ground-truth actual demand cannot be substituted for predictions
def test_actual_demand_cannot_be_substituted_for_predictions() -> None:
    """Verify that using actual_demand as pred_col raises ValueError to prevent leakage."""
    forecast_df = pd.DataFrame({
        "product_id": ["SKU_1"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "actual_demand": [10.0] * 10,
    })
    with pytest.raises(ValueError, match="cannot be identical"):
        generate_inventory_recommendations(
            forecast_df,
            pred_col="actual_demand",
            actual_col="actual_demand",
        )


# 22. Retrospective demonstration labeling and date semantics
def test_retrospective_demonstration_labeling_and_dates() -> None:
    """Verify that historical test recommendations are explicitly marked as retrospective backtests."""
    forecast_df = pd.DataFrame({
        "product_id": ["SKU_1"] * 10,
        "date": pd.date_range("2011-11-12", periods=10),
        "actual_demand": [10.0] * 10,
        "random_forest_pred": [8.0] * 10,
    })
    recs = generate_inventory_recommendations(
        forecast_df,
        as_of_date="2011-11-11",
        is_retrospective_demonstration=True,
    )
    row = recs.iloc[0]
    assert bool(row["is_retrospective_demonstration"]) is True
    assert row["recommendation_type"] == "RETROSPECTIVE_DEMONSTRATION"
    assert row["as_of_date"] == "2011-11-11"
    assert row["forecast_start_date"] == "2011-11-12"
    assert row["forecast_end_date"] == "2011-11-21"


# 23. Missing uncertainty produces unavailable status instead of zero safety stock
def test_missing_uncertainty_produces_unavailable_status_not_zero_safety_stock() -> None:
    """Verify that unresolvable uncertainty sets status UNAVAILABLE_MISSING_UNCERTAINTY, not SS=0."""
    forecast_df = pd.DataFrame({
        "product_id": ["SKU_SPARSE"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "random_forest_pred": [10.0] * 10,
    })
    # Manually provide uncertainty with NaN std and no fallback
    uncertainty_df = pd.DataFrame([{
        "product_id": "SKU_SPARSE",
        "error_history_count": 1,
        "error_mean": 0.0,
        "forecast_error_std": float("nan"),
        "is_uncertainty_fallback": True,
        "uncertainty_status": "INSUFFICIENT_HISTORY",
    }])

    recs = generate_inventory_recommendations(
        forecast_df,
        uncertainty_df=uncertainty_df,
    )
    row = recs.iloc[0]
    assert row["recommendation_status"] == "UNAVAILABLE_MISSING_UNCERTAINTY"
    assert np.isnan(row["safety_stock"])
    assert np.isnan(row["reorder_point"])
    assert np.isnan(row["target_stock"])
    assert row["suggested_order_qty"] == 0
    assert bool(row["reorder_triggered"]) is False


# 24. Rejection of negative or non-finite forecasts in protection period
def test_protection_period_negative_forecast_rejection() -> None:
    """Verify that negative forecasts in the protection period mark recommendations UNAVAILABLE_INVALID_FORECAST."""
    # 8-day protection period with day 3 negative
    preds = [10.0, 10.0, -5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    forecast_df = pd.DataFrame({
        "product_id": ["SKU_NEG"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "actual_demand": [10.0] * 10,
        "random_forest_pred": preds,
    })

    recs = generate_inventory_recommendations(forecast_df)
    row = recs.iloc[0]
    assert row["recommendation_status"] == "UNAVAILABLE_INVALID_FORECAST"
    assert np.isnan(row["expected_lead_time_demand"])
    assert row["suggested_order_qty"] == 0


# 25. Replenishment trigger condition and order filtering
def test_replenishment_trigger_condition_and_order_filtering() -> None:
    """Verify replenishment orders filter on IP <= ROP and Q > 0 strictly."""
    inv_df = pd.DataFrame([
        # Product 1: IP <= ROP, Q > 0 -> must be included
        {"product_id": "P1", "reorder_point": 100.0, "inventory_position": 80.0, "suggested_order_qty": 25},
        # Product 2: IP > ROP -> must be excluded
        {"product_id": "P2", "reorder_point": 50.0, "inventory_position": 60.0, "suggested_order_qty": 0},
        # Product 3: IP <= ROP but Q == 0 (e.g. edge case) -> must be excluded
        {"product_id": "P3", "reorder_point": 75.0, "inventory_position": 75.0, "suggested_order_qty": 0},
    ])
    orders = generate_replenishment_orders(inv_df)
    assert len(orders) == 1
    assert orders.iloc[0]["product_id"] == "P1"
    assert orders.iloc[0]["suggested_order_qty"] == 25


# 26. Uncertainty estimation uses strictly out-of-sample forecast errors
def test_uncertainty_estimation_uses_strictly_out_of_sample_errors() -> None:
    """Verify that altering actual demand alters forecast_error_std, but not expected demand."""
    base_actuals = [10.0] * 10
    preds = [10.0] * 10
    df_base = pd.DataFrame({
        "product_id": ["SKU_TEST"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "actual_demand": base_actuals,
        "random_forest_pred": preds,
    })

    mutated_actuals = [10.0, 50.0, 10.0, 80.0, 10.0, 10.0, 90.0, 10.0, 10.0, 10.0]
    df_mutated = pd.DataFrame({
        "product_id": ["SKU_TEST"] * 10,
        "date": pd.date_range("2021-01-01", periods=10),
        "actual_demand": mutated_actuals,
        "random_forest_pred": preds,
    })

    rec_base = generate_inventory_recommendations(df_base)
    rec_mutated = generate_inventory_recommendations(df_mutated)

    # Forecast demands must be identical
    assert rec_base.iloc[0]["expected_lead_time_demand"] == rec_mutated.iloc[0]["expected_lead_time_demand"]
    assert rec_base.iloc[0]["expected_protection_period_demand"] == rec_mutated.iloc[0]["expected_protection_period_demand"]

    # Uncertainty std must be different because actuals changed the errors
    assert rec_base.iloc[0]["forecast_error_std"] != rec_mutated.iloc[0]["forecast_error_std"]

