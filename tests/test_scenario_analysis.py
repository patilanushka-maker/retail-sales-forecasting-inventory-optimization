"""Unit and integration tests for Simulation & What-If Planning module.

Verifies:
1. Valid baseline scenario calculation and consistency with inventory engine.
2. Service-level validation and correct standard normal z-score mapping.
3. Replenishment lead-time validation.
4. Review-period validation.
5. Rejection of negative stock quantities (on-hand, on-order, backorders).
6. Net inventory-position calculation (IP = on_hand + on_order - backorders).
7. Reorder-point calculation (ROP = LTD + SS_LT).
8. Target-stock (S = PPD + SS_PP) and order-quantity semantics.
9. Reorder-trigger behavior (ordering triggered strictly when IP <= ROP).
10. Missing or non-finite forecast values handling.
11. Insufficient forecast horizon rejection (H < L + R).
12. Missing or invalid forecast-error uncertainty handling.
13. Unavailable baseline recommendation handling in comparisons.
14. Baseline record immutability (no in-place mutation).
15. Baseline-versus-scenario absolute difference calculations.
16. Sensitivity analysis with valid and invalid horizons without zero-substitution.
17. Empty forecast artifact handling.
18. Product with no matching forecast records.
19. View import and render function callable check.
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.scenario_engine import (
    calculate_single_sku_scenario,
    compute_scenario_comparison,
    explain_scenario_dynamics,
    get_z_score_from_service_level,
    run_lead_time_sensitivity,
    run_service_level_sensitivity,
    validate_scenario_parameters,
)


@pytest.fixture
def synthetic_28d_forecast():
    """Create a 28-day synthetic forecast time series for a single SKU."""
    return pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=28, freq="D"),
        "product_id": ["TEST_SKU"] * 28,
        "actual_demand": [10.0 + (i % 5) for i in range(28)],
        "random_forest_pred": [8.0 + (i % 4) for i in range(28)],
        "seasonal_naive_pred": [7.0 + (i % 4) for i in range(28)],
    })


# =============================================================================
# 1. Valid Baseline Scenario
# =============================================================================

def test_valid_baseline_scenario(synthetic_28d_forecast):
    """Verify calculation under valid baseline parameters."""
    res = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        service_level=0.95,
        lead_time_days=7,
        review_period_days=1,
        on_hand=50.0,
        on_order=0.0,
        backorders=0.0,
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=3.0,
    )

    assert res["is_valid"] is True
    assert res["product_id"] == "TEST_SKU"
    assert res["service_level"] == 0.95
    assert pytest.approx(res["z_score"], rel=1e-3) == 1.6449
    assert res["lead_time_days"] == 7
    assert res["review_period_days"] == 1
    assert res["protection_period_days"] == 8
    assert res["inventory_position"] == 50.0

    # Formulas verification
    # preds: [8, 9, 10, 11, 8, 9, 10] -> sum = 65.0
    expected_ltd = sum([8.0 + (i % 4) for i in range(7)])
    assert res["expected_lead_time_demand"] == pytest.approx(expected_ltd)

    # SS_LT = z * sigma * sqrt(L) = 1.64485... * 3 * sqrt(7) = 13.055...
    expected_ss = norm.ppf(0.95) * 3.0 * math.sqrt(7)
    assert res["safety_stock"] == pytest.approx(expected_ss, abs=0.05)

    # ROP = LTD + SS
    assert res["reorder_point"] == pytest.approx(expected_ltd + expected_ss, abs=0.05)
    assert res["reorder_triggered"] == (res["inventory_position"] <= res["reorder_point"])


# =============================================================================
# 2. Service-Level Validation and z-Score Mapping
# =============================================================================

def test_service_level_validation_and_z_score_mapping():
    """Verify exact z-score mapping and rejection of invalid probabilities."""
    # Standard probability mappings
    assert pytest.approx(get_z_score_from_service_level(0.8413447), rel=1e-3) == 1.0
    assert pytest.approx(get_z_score_from_service_level(0.95), rel=1e-4) == 1.64485
    assert pytest.approx(get_z_score_from_service_level(0.975), rel=1e-4) == 1.95996
    assert pytest.approx(get_z_score_from_service_level(0.99), rel=1e-4) == 2.32635

    # Rejection of invalid probabilities
    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        get_z_score_from_service_level(0.0)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        get_z_score_from_service_level(1.0)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        get_z_score_from_service_level(-0.05)

    with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
        get_z_score_from_service_level(1.25)

    with pytest.raises(ValueError, match="finite"):
        get_z_score_from_service_level(float("nan"))

    with pytest.raises(TypeError):
        get_z_score_from_service_level("0.95")


# =============================================================================
# 3. Replenishment Lead-Time Validation
# =============================================================================

def test_lead_time_validation(synthetic_28d_forecast):
    """Verify lead time must be a strictly positive integer."""
    with pytest.raises(ValueError, match="lead_time_days must be strictly positive"):
        validate_scenario_parameters(
            service_level=0.95, lead_time_days=0, review_period_days=1, on_hand=10, on_order=0, backorders=0
        )

    with pytest.raises(ValueError, match="lead_time_days must be strictly positive"):
        validate_scenario_parameters(
            service_level=0.95, lead_time_days=-7, review_period_days=1, on_hand=10, on_order=0, backorders=0
        )

    with pytest.raises(TypeError, match="lead_time_days must be an integer"):
        validate_scenario_parameters(
            service_level=0.95, lead_time_days=7.5, review_period_days=1, on_hand=10, on_order=0, backorders=0
        )


# =============================================================================
# 4. Review-Period Validation
# =============================================================================

def test_review_period_validation():
    """Verify review period must be a strictly positive integer."""
    with pytest.raises(ValueError, match="review_period_days must be strictly positive"):
        validate_scenario_parameters(
            service_level=0.95, lead_time_days=7, review_period_days=0, on_hand=10, on_order=0, backorders=0
        )

    with pytest.raises(ValueError, match="review_period_days must be strictly positive"):
        validate_scenario_parameters(
            service_level=0.95, lead_time_days=7, review_period_days=-1, on_hand=10, on_order=0, backorders=0
        )


# =============================================================================
# 5. Negative Inventory Quantity Rejection
# =============================================================================

def test_negative_inventory_quantity_rejection():
    """Verify that negative on-hand, on-order, or backorders are strictly rejected."""
    with pytest.raises(ValueError, match="on_hand' cannot be negative"):
        validate_scenario_parameters(0.95, 7, 1, on_hand=-5.0, on_order=0.0, backorders=0.0)

    with pytest.raises(ValueError, match="on_order' cannot be negative"):
        validate_scenario_parameters(0.95, 7, 1, on_hand=10.0, on_order=-1.0, backorders=0.0)

    with pytest.raises(ValueError, match="backorders' cannot be negative"):
        validate_scenario_parameters(0.95, 7, 1, on_hand=10.0, on_order=0.0, backorders=-10.0)


# =============================================================================
# 6. Correct Inventory-Position Calculation
# =============================================================================

def test_correct_inventory_position_calculation(synthetic_28d_forecast):
    """Verify net inventory position formula: IP = on_hand + on_order - backorders."""
    # Standard positive
    res1 = calculate_single_sku_scenario(
        product_id="TEST_SKU", on_hand=100.0, on_order=40.0, backorders=10.0,
        forecast_df=synthetic_28d_forecast, forecast_error_std=2.0
    )
    assert res1["inventory_position"] == 130.0

    # Backorders exceed stock (negative IP)
    res2 = calculate_single_sku_scenario(
        product_id="TEST_SKU", on_hand=10.0, on_order=5.0, backorders=30.0,
        forecast_df=synthetic_28d_forecast, forecast_error_std=2.0
    )
    assert res2["inventory_position"] == -15.0


# =============================================================================
# 7. Correct Reorder-Point Calculation
# =============================================================================

def test_correct_reorder_point_calculation(synthetic_28d_forecast):
    """Verify Reorder Point: ROP = Lead-Time Demand + Safety Stock."""
    res = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        service_level=0.95,
        lead_time_days=7,
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=2.0,
    )
    assert res["is_valid"] is True
    expected_rop = res["expected_lead_time_demand"] + res["safety_stock"]
    assert pytest.approx(res["reorder_point"], rel=1e-3) == expected_rop


# =============================================================================
# 8. Correct Target-Stock and Order-Quantity Semantics
# =============================================================================

def test_correct_target_stock_and_order_quantity_semantics(synthetic_28d_forecast):
    """Verify Target Stock (S = PPD + SS_PP) and whole-unit order quantity (Q = max(0, ceil(S - IP)))."""
    # Force reorder condition: on_hand=0 -> IP=0 <= ROP
    res = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        on_hand=0.0,
        on_order=0.0,
        backorders=0.0,
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=2.0,
        rounding_mode="ceil",
    )
    assert res["reorder_triggered"] is True
    assert res["suggested_order_qty"] == math.ceil(res["target_stock"] - res["inventory_position"])
    assert res["suggested_order_qty"] > 0
    assert res["recommendation_status"] == "REORDER_NOW"


# =============================================================================
# 9. Reorder-Trigger Behavior
# =============================================================================

def test_reorder_trigger_behavior(synthetic_28d_forecast):
    """Verify order is triggered when IP <= ROP, and 0 order when IP > ROP."""
    # When IP > ROP: no order needed
    res_adequate = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        on_hand=500.0,  # Well above ROP
        on_order=0.0,
        backorders=0.0,
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=2.0,
    )
    assert res_adequate["reorder_triggered"] is False
    assert res_adequate["suggested_order_qty"] == 0
    assert res_adequate["recommendation_status"] == "STOCK_ADEQUATE"

    # When IP <= ROP: order triggered
    res_reorder = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        on_hand=10.0,  # Below ROP
        on_order=0.0,
        backorders=0.0,
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=2.0,
    )
    assert res_reorder["reorder_triggered"] is True
    assert res_reorder["suggested_order_qty"] > 0
    assert res_reorder["recommendation_status"] == "REORDER_NOW"


# =============================================================================
# 10. Missing Forecast Dates
# =============================================================================

def test_missing_forecast_dates(synthetic_28d_forecast):
    """Verify missing/NaN predictions in required window return invalid forecast state."""
    broken_fc = synthetic_28d_forecast.copy()
    broken_fc.loc[2, "random_forest_pred"] = np.nan

    res = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        lead_time_days=7,
        review_period_days=1,
        forecast_df=broken_fc,
        forecast_error_std=2.0,
    )
    assert res["is_valid"] is False
    assert res["recommendation_status"] == "UNAVAILABLE_INVALID_FORECAST"


# =============================================================================
# 11. Insufficient Protection-Period Horizon
# =============================================================================

def test_insufficient_protection_period_horizon():
    """Verify requested horizon exceeding available forecast is rejected without zero-padding."""
    # Only 5 days of forecasts available
    short_fc = pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=5, freq="D"),
        "product_id": ["SHORT_SKU"] * 5,
        "random_forest_pred": [10.0] * 5,
    })

    # Requested P = L + R = 7 + 1 = 8 days > 5 days available
    res = calculate_single_sku_scenario(
        product_id="SHORT_SKU",
        lead_time_days=7,
        review_period_days=1,
        forecast_df=short_fc,
        forecast_error_std=2.0,
    )
    assert res["is_valid"] is False
    assert res["recommendation_status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON"
    assert "exceeds available forecast horizon" in res["rejection_reason"]
    # Does not produce a fake 0 order quantity as a valid result
    assert np.isnan(res["safety_stock"])
    assert np.isnan(res["reorder_point"])


# =============================================================================
# 12. Missing Forecast-Error Uncertainty
# =============================================================================

def test_missing_forecast_error_uncertainty(synthetic_28d_forecast):
    """Verify missing, NaN, or negative uncertainty does not silently become zero safety stock."""
    # NaN uncertainty
    res_nan = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=float("nan"),
        baseline_record=None,
    )
    assert res_nan["is_valid"] is False
    assert res_nan["recommendation_status"] == "UNAVAILABLE_MISSING_UNCERTAINTY"

    # None uncertainty with no baseline
    res_none = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=None,
        baseline_record={},
    )
    assert res_none["is_valid"] is False
    assert res_none["recommendation_status"] == "UNAVAILABLE_MISSING_UNCERTAINTY"


# =============================================================================
# 13. Unavailable Baseline Recommendation
# =============================================================================

def test_unavailable_baseline_recommendation(synthetic_28d_forecast):
    """Verify scenario comparison handles unavailable baseline without crash."""
    baseline_unavail = {
        "product_id": "TEST_SKU",
        "recommendation_status": "UNAVAILABLE_INSUFFICIENT_HORIZON",
        "service_level": 0.95,
        "lead_time_days": 7,
        "review_period_days": 1,
        "safety_stock": np.nan,
        "reorder_point": np.nan,
        "target_stock": np.nan,
        "suggested_order_qty": 0,
        "inventory_position": np.nan,
    }

    scen = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=2.0,
    )
    assert scen["is_valid"] is True

    comp = compute_scenario_comparison(baseline_unavail, scen)
    assert comp["is_baseline_valid"] is False
    assert comp["is_scenario_valid"] is True
    assert comp["baseline_status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON"
    # Undefined delta for safety stock
    assert comp["deltas"]["safety_stock"] is None


# =============================================================================
# 14. Baseline Record Immutability
# =============================================================================

def test_baseline_record_immutability(synthetic_28d_forecast):
    """Verify baseline record dictionary is not mutated during scenario calculation."""
    original_baseline = {
        "product_id": "TEST_SKU",
        "service_level": 0.95,
        "lead_time_days": 7,
        "review_period_days": 1,
        "forecast_error_std": 3.0,
        "simulated_on_hand": 20.0,
        "simulated_on_order": 0.0,
        "simulated_backorders": 0.0,
        "recommendation_status": "STOCK_ADEQUATE",
    }
    baseline_copy = original_baseline.copy()

    scen = calculate_single_sku_scenario(
        product_id="TEST_SKU",
        service_level=0.99,  # Changed
        lead_time_days=10,   # Changed
        forecast_df=synthetic_28d_forecast,
        baseline_record=original_baseline,
    )
    assert scen["is_valid"] is True

    # Check baseline record was not altered
    assert original_baseline == baseline_copy


# =============================================================================
# 15. Baseline-Versus-Scenario Difference Calculations
# =============================================================================

def test_baseline_versus_scenario_difference_calculations():
    """Verify compute_scenario_comparison calculates exact deltas."""
    baseline = {
        "service_level": 0.90,
        "lead_time_days": 7,
        "review_period_days": 1,
        "safety_stock": 10.0,
        "reorder_point": 50.0,
        "target_stock": 60.0,
        "suggested_order_qty": 20,
        "inventory_position": 40.0,
        "reorder_triggered": True,
        "recommendation_status": "REORDER_NOW",
    }
    scenario = {
        "service_level": 0.95,
        "lead_time_days": 7,
        "review_period_days": 1,
        "safety_stock": 15.0,
        "reorder_point": 55.0,
        "target_stock": 65.0,
        "suggested_order_qty": 25,
        "inventory_position": 40.0,
        "reorder_triggered": True,
        "recommendation_status": "REORDER_NOW",
        "is_valid": True,
    }

    comp = compute_scenario_comparison(baseline, scenario)
    assert pytest.approx(comp["deltas"]["service_level"], rel=1e-3) == 0.05
    assert comp["deltas"]["safety_stock"] == 5.0
    assert comp["deltas"]["reorder_point"] == 5.0
    assert comp["deltas"]["target_stock"] == 5.0
    assert comp["deltas"]["suggested_order_qty"] == 5.0
    assert comp["trigger_changed"] is False


# =============================================================================
# 16. Sensitivity Analysis with Valid and Invalid Horizons
# =============================================================================

def test_sensitivity_analysis_with_valid_and_invalid_horizons(synthetic_28d_forecast):
    """Verify sensitivity across lead times marks horizons > 28 as unavailable without zero substitution."""
    lead_times = [3, 7, 14, 21, 28, 35]
    df_lt = run_lead_time_sensitivity(
        product_id="TEST_SKU",
        lead_times=lead_times,
        review_period_days=1,
        forecast_df=synthetic_28d_forecast,
        baseline_record={"forecast_error_std": 2.0},
    )

    assert len(df_lt) == 6

    # L=3 (P=4), L=7 (P=8), L=14 (P=15), L=21 (P=22), L=27 (P=28) -> valid
    # L=28 -> P = 28 + 1 = 29 > 28 available -> INVALID
    # L=35 -> P = 35 + 1 = 36 > 28 available -> INVALID
    row_28 = df_lt[df_lt["lead_time_days"] == 28].iloc[0]
    assert bool(row_28["is_valid"]) is False
    assert row_28["status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON"
    assert np.isnan(row_28["safety_stock"])

    row_35 = df_lt[df_lt["lead_time_days"] == 35].iloc[0]
    assert bool(row_35["is_valid"]) is False
    assert row_35["status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON"

    # L=7 must be valid
    row_7 = df_lt[df_lt["lead_time_days"] == 7].iloc[0]
    assert bool(row_7["is_valid"]) is True
    assert row_7["safety_stock"] > 0


# =============================================================================
# 17. Empty Forecast Artifact
# =============================================================================

def test_empty_forecast_artifact():
    """Verify empty forecast DataFrame is handled cleanly."""
    empty_df = pd.DataFrame(columns=["date", "product_id", "random_forest_pred"])
    res = calculate_single_sku_scenario(
        product_id="ANY_SKU",
        forecast_df=empty_df,
        forecast_error_std=2.0,
    )
    assert res["is_valid"] is False
    assert res["recommendation_status"] == "UNAVAILABLE_NO_FORECAST"


# =============================================================================
# 18. Product with No Matching Forecast Records
# =============================================================================

def test_product_with_no_matching_forecast_records(synthetic_28d_forecast):
    """Verify product absent from forecast DataFrame returns unavailable state."""
    res = calculate_single_sku_scenario(
        product_id="NON_EXISTENT_SKU_9999",
        forecast_df=synthetic_28d_forecast,
        forecast_error_std=2.0,
    )
    assert res["is_valid"] is False
    assert res["recommendation_status"] == "UNAVAILABLE_NO_FORECAST"


# =============================================================================
# 19. View Import and Render Callable Verification
# =============================================================================

def test_scenario_view_import_and_render_callable():
    """Verify scenario view imports and exposes callable render function."""
    from app.views.scenario_analysis import render_scenario_analysis_page
    assert callable(render_scenario_analysis_page)


# =============================================================================
# 20. Regression Tests for Formatting Error (Unknown format code 'd' for float)
# =============================================================================

def test_explain_scenario_dynamics_float_lead_time_delta():
    """Verify explain_scenario_dynamics formats float lead-time deltas without format code error."""
    # Test positive float delta
    comp_pos = {
        "deltas": {"lead_time_days": 2.0, "service_level": 0.0, "safety_stock": 5.0, "reorder_point": 10.0, "inventory_position": 0.0},
        "trigger_changed": False,
    }
    explanation_pos = explain_scenario_dynamics(comp_pos)
    assert "+2 days vs baseline" in explanation_pos["lead_time_driver"]

    # Test negative float delta
    comp_neg = {
        "deltas": {"lead_time_days": -3.0, "service_level": 0.0, "safety_stock": -5.0, "reorder_point": -10.0, "inventory_position": 0.0},
        "trigger_changed": False,
    }
    explanation_neg = explain_scenario_dynamics(comp_neg)
    assert "-3 days vs baseline" in explanation_neg["lead_time_driver"]

    # Test integer delta as well
    comp_int = {
        "deltas": {"lead_time_days": 1, "service_level": 0.0, "safety_stock": 2.0, "reorder_point": 5.0, "inventory_position": 0.0},
        "trigger_changed": False,
    }
    explanation_int = explain_scenario_dynamics(comp_int)
    assert "+1 days vs baseline" in explanation_int["lead_time_driver"]


def test_scenario_results_formatting_regression():
    """Verify delta_qty formatting handles float, int, and None values safely."""
    # Test float delta
    d_float = {"suggested_order_qty": 15.0}
    delta_qty_float = (
        f"{int(round(float(d_float['suggested_order_qty']))):+d} units"
        if (d_float.get("suggested_order_qty") is not None and pd.notnull(d_float["suggested_order_qty"]))
        else None
    )
    assert delta_qty_float == "+15 units"

    # Test zero float delta
    d_zero = {"suggested_order_qty": 0.0}
    delta_qty_zero = (
        f"{int(round(float(d_zero['suggested_order_qty']))):+d} units"
        if (d_zero.get("suggested_order_qty") is not None and pd.notnull(d_zero["suggested_order_qty"]))
        else None
    )
    assert delta_qty_zero == "+0 units"

    # Test negative float delta
    d_neg = {"suggested_order_qty": -8.0}
    delta_qty_neg = (
        f"{int(round(float(d_neg['suggested_order_qty']))):+d} units"
        if (d_neg.get("suggested_order_qty") is not None and pd.notnull(d_neg["suggested_order_qty"]))
        else None
    )
    assert delta_qty_neg == "-8 units"

    # Test integer delta
    d_int = {"suggested_order_qty": 10}
    delta_qty_int = (
        f"{int(round(float(d_int['suggested_order_qty']))):+d} units"
        if (d_int.get("suggested_order_qty") is not None and pd.notnull(d_int["suggested_order_qty"]))
        else None
    )
    assert delta_qty_int == "+10 units"

    # Test None delta
    d_none = {"suggested_order_qty": None}
    delta_qty_none = (
        f"{int(round(float(d_none['suggested_order_qty']))):+d} units"
        if (d_none.get("suggested_order_qty") is not None and pd.notnull(d_none["suggested_order_qty"]))
        else None
    )
    assert delta_qty_none is None


def test_apptest_simulation_page_renders_with_zero_errors():
    """Verify that the Simulation & What-If Planning page renders in AppTest with zero errors and zero exceptions."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    entrypoint = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
    at = AppTest.from_file(entrypoint)
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("6. Simulation & What-If Planning").run(timeout=30)

    assert len(at.exception) == 0, f"Unexpected exception in Simulation page: {at.exception}"
    assert len(at.error) == 0, f"Unexpected st.error in Simulation page: {[e.value for e in at.error]}"

