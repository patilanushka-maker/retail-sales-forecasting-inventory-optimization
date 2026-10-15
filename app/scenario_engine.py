"""Simulation and What-If scenario calculation engine.

Computes alternative periodic review (R, s, S) inventory policies under user-configured assumptions:
- Cycle Service Level (CSL) and dynamic standard normal z-score.
- Supplier replenishment lead time (L) and periodic review interval (R).
- Simulated operational inventory positions (On-Hand, On-Order, Backorders).

Strict Horizon & Governance Invariants:
1. Validates that available forecast horizon (H) covers the complete protection period (P = L + R).
2. Does NOT zero-fill, extrapolate, or reuse forecasts when H < P.
3. Preserves legitimate nulls and reports explicit unavailable states.
4. Uses genuine out-of-sample forecast-error uncertainty; does NOT substitute zero for missing uncertainty.
5. Implements pure functions that can be tested independently of Streamlit.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# Ensure project root and src directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from retail_forecasting.inventory.calculate_stock import (
        calculate_safety_stock,
        calculate_lead_time_demand,
        calculate_protection_period_demand,
        calculate_reorder_point,
        calculate_target_stock,
        calculate_order_quantity,
    )
    from retail_forecasting.inventory.reorder_recommendations import (
        inventory_position,
    )
except ImportError:
    from src.retail_forecasting.inventory.calculate_stock import (
        calculate_safety_stock,
        calculate_lead_time_demand,
        calculate_protection_period_demand,
        calculate_reorder_point,
        calculate_target_stock,
        calculate_order_quantity,
    )
    from src.retail_forecasting.inventory.reorder_recommendations import (
        inventory_position,
    )
from app.data_loader import (
    get_single_product_inventory_details,
    load_product_forecast_comparison,
)


def get_z_score_from_service_level(service_level: float) -> float:
    """Calculate the standard normal quantile (z-score) for a target service level.

    Parameters
    ----------
    service_level : float
        Target Cycle Service Level (CSL) strictly between 0.0 and 1.0 exclusive.

    Returns
    -------
    float
        Standard normal z-score.

    Raises
    ------
    ValueError
        If service_level is not strictly between 0 and 1, or is non-finite.
    TypeError
        If service_level is non-numeric.
    """
    if not isinstance(service_level, (int, float, np.number)) or isinstance(service_level, bool):
        raise TypeError(f"service_level must be numeric, got {type(service_level).__name__}")
    if not np.isfinite(service_level):
        raise ValueError(f"service_level must be finite, got {service_level}")
    if service_level <= 0.0 or service_level >= 1.0:
        raise ValueError(f"service_level must be strictly between 0.0 and 1.0 exclusive, got {service_level}")

    return float(norm.ppf(service_level))


def validate_scenario_parameters(
    service_level: float,
    lead_time_days: int,
    review_period_days: int,
    on_hand: float,
    on_order: float,
    backorders: float,
) -> None:
    """Validate scenario control inputs against physical and probabilistic bounds.

    Raises
    ------
    ValueError
        If inputs violate non-negativity, horizon, or probability constraints.
    TypeError
        If input types are invalid.
    """
    # Service level check
    get_z_score_from_service_level(service_level)

    # Lead time check
    if not isinstance(lead_time_days, (int, np.integer)) or isinstance(lead_time_days, bool):
        if isinstance(lead_time_days, float) and lead_time_days.is_integer():
            lead_time_days = int(lead_time_days)
        else:
            raise TypeError(f"lead_time_days must be an integer, got {type(lead_time_days).__name__}")
    if lead_time_days <= 0:
        raise ValueError(f"lead_time_days must be strictly positive, got {lead_time_days}")

    # Review period check
    if not isinstance(review_period_days, (int, np.integer)) or isinstance(review_period_days, bool):
        if isinstance(review_period_days, float) and review_period_days.is_integer():
            review_period_days = int(review_period_days)
        else:
            raise TypeError(f"review_period_days must be an integer, got {type(review_period_days).__name__}")
    if review_period_days <= 0:
        raise ValueError(f"review_period_days must be strictly positive, got {review_period_days}")

    # Stock quantities checks
    for val, name in [
        (on_hand, "on_hand"),
        (on_order, "on_order"),
        (backorders, "backorders"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"'{name}' must be numeric, got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"'{name}' must be finite, got {val}")
        if val < 0.0:
            raise ValueError(f"Stock quantity '{name}' cannot be negative, got {val}")


def calculate_single_sku_scenario(
    product_id: str,
    service_level: float = 0.95,
    lead_time_days: int = 7,
    review_period_days: int = 1,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    forecast_df: Optional[pd.DataFrame] = None,
    baseline_record: Optional[Dict[str, Any]] = None,
    forecast_error_std: Optional[float] = None,
    rounding_mode: str = "ceil",
) -> Dict[str, Any]:
    """Calculate alternative inventory replenishment policy under scenario assumptions.

    Parameters
    ----------
    product_id : str
        Target product SKU identifier.
    service_level : float, default 0.95
        Target Cycle Service Level (CSL) strictly between 0 and 1.
    lead_time_days : int, default 7
        Supplier replenishment lead time in days (L >= 1).
    review_period_days : int, default 1
        Review cycle frequency in days (R >= 1).
    on_hand : float, default 0.0
        Simulated physical stock available (>= 0).
    on_order : float, default 0.0
        Simulated stock in transit (>= 0).
    backorders : float, default 0.0
        Simulated unfulfilled demand (>= 0).
    forecast_df : Optional[pd.DataFrame], optional
        Forecast comparison dataset for this product. If None, loaded from disk.
    baseline_record : Optional[Dict[str, Any]], optional
        Pre-extracted baseline recommendation record for uncertainty extraction.
    forecast_error_std : Optional[float], optional
        Direct forecast error standard deviation override.
    rounding_mode : str, default 'ceil'
        Rounding policy for suggested order quantities ('ceil', 'round', 'floor').

    Returns
    -------
    Dict[str, Any]
        Dictionary of scenario outputs, policy thresholds, and validity status.
    """
    # 1. Parameter Validation (raises ValueError on invalid user inputs)
    validate_scenario_parameters(
        service_level=service_level,
        lead_time_days=lead_time_days,
        review_period_days=review_period_days,
        on_hand=on_hand,
        on_order=on_order,
        backorders=backorders,
    )

    lead_time_days = int(lead_time_days)
    review_period_days = int(review_period_days)
    protection_period_days = lead_time_days + review_period_days
    z = get_z_score_from_service_level(service_level)
    ip = inventory_position(on_hand=on_hand, on_order=on_order, backorders=backorders)

    # Base payload structure
    result: Dict[str, Any] = {
        "product_id": str(product_id).strip(),
        "service_level": service_level,
        "z_score": round(z, 4),
        "lead_time_days": lead_time_days,
        "review_period_days": review_period_days,
        "protection_period_days": protection_period_days,
        "on_hand": on_hand,
        "on_order": on_order,
        "backorders": backorders,
        "inventory_position": round(ip, 2),
        "expected_lead_time_demand": float("nan"),
        "expected_protection_period_demand": float("nan"),
        "forecast_error_std": float("nan"),
        "safety_stock": float("nan"),
        "safety_stock_protection": float("nan"),
        "reorder_point": float("nan"),
        "target_stock": float("nan"),
        "suggested_order_qty": 0,
        "reorder_triggered": False,
        "recommendation_status": "UNKNOWN",
        "is_valid": False,
        "rejection_reason": "",
        "available_horizon": 0,
        "required_horizon": protection_period_days,
    }

    # 2. Resolve Forecast Predictions
    if forecast_df is None:
        try:
            prod_fc = load_product_forecast_comparison(product_id)
        except Exception as err:
            result.update({
                "recommendation_status": "UNAVAILABLE_NO_FORECAST",
                "rejection_reason": f"Failed to load forecast data: {err}",
            })
            return result
    else:
        if "product_id" in forecast_df.columns:
            prod_fc = forecast_df[forecast_df["product_id"].fillna("").astype(str).str.strip() == str(product_id).strip()].copy()
        else:
            prod_fc = forecast_df.copy()

    if prod_fc is None or prod_fc.empty:
        result.update({
            "recommendation_status": "UNAVAILABLE_NO_FORECAST",
            "rejection_reason": f"No forecast records found for product SKU '{product_id}'.",
        })
        return result

    # Sort chronologically
    if "date" in prod_fc.columns:
        prod_fc = prod_fc.sort_values("date").reset_index(drop=True)

    available_h = len(prod_fc)
    result["available_horizon"] = available_h

    # 3. Horizon Validation (CRITICAL REQUIREMENT)
    if available_h < protection_period_days:
        result.update({
            "recommendation_status": "UNAVAILABLE_INSUFFICIENT_HORIZON",
            "rejection_reason": (
                f"Requested protection period ({protection_period_days} days = {lead_time_days}d lead time + {review_period_days}d review) "
                f"exceeds available forecast horizon ({available_h} days). Forecasts cannot be fabricated or extrapolated."
            ),
        })
        return result

    # Extract required prediction window
    pred_col = "random_forest_pred" if "random_forest_pred" in prod_fc.columns else prod_fc.columns[-1]
    window_preds = prod_fc[pred_col].iloc[:protection_period_days].values

    # Check for NaN / non-finite / negative forecasts
    if not np.isfinite(window_preds).all() or (window_preds < 0.0).any():
        result.update({
            "recommendation_status": "UNAVAILABLE_INVALID_FORECAST",
            "rejection_reason": "Forecast records contain missing, non-finite, or negative predictions in the required window.",
        })
        return result

    # 4. Resolve Forecast-Error Uncertainty (sigma_e)
    sigma_e = forecast_error_std
    if sigma_e is None:
        if baseline_record is not None and "forecast_error_std" in baseline_record:
            sigma_e = baseline_record["forecast_error_std"]
        else:
            baseline_details = get_single_product_inventory_details(product_id)
            if baseline_details is not None:
                sigma_e = baseline_details.get("forecast_error_std")

    if sigma_e is None or not isinstance(sigma_e, (int, float, np.number)) or not np.isfinite(sigma_e) or sigma_e < 0.0:
        result.update({
            "recommendation_status": "UNAVAILABLE_MISSING_UNCERTAINTY",
            "rejection_reason": "Product has missing, non-finite, or undefined forecast-error uncertainty in evaluation artifacts.",
        })
        return result

    result["forecast_error_std"] = round(float(sigma_e), 4)

    # 5. Core Inventory Policy Calculations
    try:
        ltd = calculate_lead_time_demand(window_preds, lead_time_days=lead_time_days)
        ppd = calculate_protection_period_demand(window_preds, protection_period_days=protection_period_days)

        ss_lt = calculate_safety_stock(
            service_level=service_level,
            daily_forecast_error_std=float(sigma_e),
            lead_time_days=lead_time_days,
        )
        ss_pp = calculate_safety_stock(
            service_level=service_level,
            daily_forecast_error_std=float(sigma_e),
            lead_time_days=protection_period_days,
        )

        rop = calculate_reorder_point(lead_time_demand=ltd, safety_stock=ss_lt)
        target_stock = calculate_target_stock(protection_period_demand=ppd, safety_stock=ss_pp)

        # Order Trigger: IP <= ROP
        reorder_triggered = bool(ip <= rop)
        if reorder_triggered:
            order_qty = calculate_order_quantity(
                target_stock=target_stock,
                inventory_position=ip,
                rounding_mode=rounding_mode,
            )
            rec_status = "REORDER_NOW" if order_qty > 0 else "STOCK_ADEQUATE"
        else:
            order_qty = 0
            rec_status = "STOCK_ADEQUATE"

        result.update({
            "expected_lead_time_demand": round(ltd, 2),
            "expected_protection_period_demand": round(ppd, 2),
            "safety_stock": round(ss_lt, 2),
            "safety_stock_protection": round(ss_pp, 2),
            "reorder_point": round(rop, 2),
            "target_stock": round(target_stock, 2),
            "suggested_order_qty": int(order_qty),
            "reorder_triggered": reorder_triggered,
            "recommendation_status": rec_status,
            "is_valid": True,
            "rejection_reason": "",
        })

    except Exception as err:
        result.update({
            "recommendation_status": "CALCULATION_ERROR",
            "rejection_reason": f"Calculation error during policy evaluation: {err}",
            "is_valid": False,
        })

    return result


def compute_scenario_comparison(
    baseline_record: Optional[Dict[str, Any]],
    scenario_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute absolute differences between baseline recommendation and scenario output.

    Parameters
    ----------
    baseline_record : Optional[Dict[str, Any]]
        Saved baseline recommendation record (not mutated).
    scenario_result : Dict[str, Any]
        Scenario calculation output dictionary.

    Returns
    -------
    Dict[str, Any]
        Comparison dictionary containing baseline values, scenario values, and deltas.
    """
    if baseline_record is None or not isinstance(baseline_record, dict):
        base_clean = {}
    else:
        # Preserve immutability: do not mutate baseline_record
        base_clean = baseline_record.copy()

    fields_to_compare = [
        ("service_level", "Service Level"),
        ("lead_time_days", "Lead Time (Days)"),
        ("review_period_days", "Review Period (Days)"),
        ("safety_stock", "Safety Stock (Units)"),
        ("reorder_point", "Reorder Point (ROP)"),
        ("target_stock", "Target Stock Level (S)"),
        ("inventory_position", "Inventory Position"),
        ("suggested_order_qty", "Suggested Order Qty"),
    ]

    comparison_rows = []
    deltas = {}

    for field_key, display_name in fields_to_compare:
        base_val = base_clean.get(field_key)
        scen_val = scenario_result.get(field_key)

        # Check if both values are valid numbers
        is_num_base = isinstance(base_val, (int, float, np.number)) and not np.isnan(base_val)
        is_num_scen = isinstance(scen_val, (int, float, np.number)) and not np.isnan(scen_val)

        if is_num_base and is_num_scen:
            delta = float(scen_val) - float(base_val)
            deltas[field_key] = delta
        else:
            delta = None
            deltas[field_key] = None

        comparison_rows.append({
            "metric": display_name,
            "field_key": field_key,
            "baseline": base_val,
            "scenario": scen_val,
            "delta": delta,
        })

    base_triggered = base_clean.get("reorder_triggered")
    scen_triggered = scenario_result.get("reorder_triggered")
    trigger_changed = (base_triggered != scen_triggered) if (base_triggered is not None and scen_triggered is not None) else None

    return {
        "comparison_rows": comparison_rows,
        "deltas": deltas,
        "baseline_status": base_clean.get("recommendation_status", "UNKNOWN"),
        "scenario_status": scenario_result.get("recommendation_status", "UNKNOWN"),
        "baseline_triggered": base_triggered,
        "scenario_triggered": scen_triggered,
        "trigger_changed": trigger_changed,
        "is_baseline_valid": base_clean.get("recommendation_status") in ["REORDER_NOW", "STOCK_ADEQUATE"],
        "is_scenario_valid": scenario_result.get("is_valid", False),
    }


def run_service_level_sensitivity(
    product_id: str,
    service_levels: Optional[List[float]] = None,
    lead_time_days: int = 7,
    review_period_days: int = 1,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    forecast_df: Optional[pd.DataFrame] = None,
    baseline_record: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate inventory policy thresholds across multiple service-level targets.

    Parameters
    ----------
    product_id : str
        Target product SKU.
    service_levels : Optional[List[float]], optional
        List of service-level targets (default [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]).
    lead_time_days : int, default 7
        Fixed lead time.
    review_period_days : int, default 1
        Fixed review period.
    on_hand : float, default 0.0
        Simulated stock.
    on_order : float, default 0.0
        Simulated in transit.
    backorders : float, default 0.0
        Simulated backorders.
    forecast_df : Optional[pd.DataFrame], optional
        Preloaded forecast DataFrame.
    baseline_record : Optional[Dict[str, Any]], optional
        Pre-extracted baseline record.

    Returns
    -------
    pd.DataFrame
        Sensitivity results table with columns for SL, z-score, SS, ROP, S, Q, and status.
    """
    if service_levels is None:
        service_levels = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]

    rows = []
    for sl in service_levels:
        res = calculate_single_sku_scenario(
            product_id=product_id,
            service_level=sl,
            lead_time_days=lead_time_days,
            review_period_days=review_period_days,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            forecast_df=forecast_df,
            baseline_record=baseline_record,
        )
        rows.append({
            "service_level": sl,
            "z_score": res.get("z_score"),
            "safety_stock": res.get("safety_stock"),
            "reorder_point": res.get("reorder_point"),
            "target_stock": res.get("target_stock"),
            "suggested_order_qty": res.get("suggested_order_qty"),
            "reorder_triggered": res.get("reorder_triggered"),
            "is_valid": res.get("is_valid", False),
            "status": res.get("recommendation_status", "UNKNOWN"),
        })

    return pd.DataFrame(rows)


def run_lead_time_sensitivity(
    product_id: str,
    lead_times: Optional[List[int]] = None,
    service_level: float = 0.95,
    review_period_days: int = 1,
    on_hand: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    forecast_df: Optional[pd.DataFrame] = None,
    baseline_record: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate inventory policy thresholds across varying supplier lead times.

    Scenarios where L + R exceeds available forecast horizon are explicitly marked
    unavailable and do NOT fabricate zero values.

    Parameters
    ----------
    product_id : str
        Target product SKU.
    lead_times : Optional[List[int]], optional
        List of lead times to evaluate (default [3, 5, 7, 10, 14, 21, 28]).
    service_level : float, default 0.95
        Fixed service level.
    review_period_days : int, default 1
        Fixed review period.
    on_hand : float, default 0.0
        Simulated on-hand.
    on_order : float, default 0.0
        Simulated on-order.
    backorders : float, default 0.0
        Simulated backorders.
    forecast_df : Optional[pd.DataFrame], optional
        Preloaded forecast DataFrame.
    baseline_record : Optional[Dict[str, Any]], optional
        Pre-extracted baseline record.

    Returns
    -------
    pd.DataFrame
        Sensitivity results table with columns for L, protection period, LTD, SS, ROP, and status.
    """
    if lead_times is None:
        lead_times = [3, 5, 7, 10, 14, 21, 28]

    rows = []
    for lt in lead_times:
        res = calculate_single_sku_scenario(
            product_id=product_id,
            service_level=service_level,
            lead_time_days=lt,
            review_period_days=review_period_days,
            on_hand=on_hand,
            on_order=on_order,
            backorders=backorders,
            forecast_df=forecast_df,
            baseline_record=baseline_record,
        )
        rows.append({
            "lead_time_days": lt,
            "protection_period_days": lt + review_period_days,
            "expected_lead_time_demand": res.get("expected_lead_time_demand"),
            "safety_stock": res.get("safety_stock"),
            "reorder_point": res.get("reorder_point"),
            "target_stock": res.get("target_stock"),
            "suggested_order_qty": res.get("suggested_order_qty"),
            "reorder_triggered": res.get("reorder_triggered"),
            "is_valid": res.get("is_valid", False),
            "status": res.get("recommendation_status", "UNKNOWN"),
        })

    return pd.DataFrame(rows)


def explain_scenario_dynamics(comparison: Dict[str, Any]) -> Dict[str, str]:
    """Generate beginner-friendly explanations of key drivers causing scenario changes.

    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison structure from compute_scenario_comparison.

    Returns
    -------
    Dict[str, str]
        Explanatory commentary cards for service level, lead time, inventory position, and trigger.
    """
    deltas = comparison.get("deltas", {})
    d_sl = deltas.get("service_level")
    d_lt = deltas.get("lead_time_days")
    d_ss = deltas.get("safety_stock")
    d_rop = deltas.get("reorder_point")
    d_ip = deltas.get("inventory_position")
    d_qty = deltas.get("suggested_order_qty")
    trigger_changed = comparison.get("trigger_changed")

    # 1. Service Level explanation
    if d_sl is not None and abs(d_sl) > 1e-4:
        direction = "increased" if d_sl > 0 else "decreased"
        ss_dir = "higher" if (d_ss is not None and d_ss > 0) else "lower"
        sl_text = (
            f"Target service level was **{direction}** by {abs(d_sl) * 100:.1f}%. "
            f"Under standard normal uncertainty, this changes the safety factor z-score, leading to a {ss_dir} "
            f"safety stock buffer ({d_ss:+.1f} units vs baseline)."
        )
    else:
        sl_text = "Target service level matches baseline (95% CSL, z = 1.6449)."

    # 2. Lead Time explanation
    if d_lt is not None and abs(d_lt) > 0:
        lt_dir = "longer" if d_lt > 0 else "shorter"
        rop_dir = "increases" if (d_rop is not None and d_rop > 0) else "decreases"
        lt_text = (
            f"Supplier lead time is configured **{lt_dir}** ({int(round(float(d_lt))):+d} days vs baseline). "
            f"A {lt_dir} lead time expands the exposure window, which typically {rop_dir} both cumulative lead-time demand "
            f"and the safety stock needed to absorb demand volatility during replenishment."
        )
    else:
        lt_text = "Replenishment lead time matches the baseline setting (7 days)."

    # 3. Inventory Position & Trigger explanation
    if d_ip is not None and abs(d_ip) > 1e-4:
        ip_dir = "higher" if d_ip > 0 else "lower"
        ip_text = (
            f"Simulated inventory position is **{ip_dir}** ({d_ip:+.0f} units vs baseline). "
            f"Replenishment is triggered strictly when net Inventory Position (On-Hand + On-Order - Backorders) "
            "falls to or below the Reorder Point (IP <= ROP)."
        )
    else:
        ip_text = "Simulated operational inventory quantities match baseline values."

    if trigger_changed is True:
        scen_trig = comparison.get("scenario_triggered")
        action = "now triggered (reorder required)" if scen_trig else "no longer triggered (sufficient stock)"
        trigger_text = f"🚨 **Reorder Status Changed**: Replenishment is {action} under this scenario."
    else:
        trigger_text = "Reorder trigger status is unchanged between baseline and scenario."

    return {
        "service_level_driver": sl_text,
        "lead_time_driver": lt_text,
        "inventory_driver": ip_text,
        "trigger_driver": trigger_text,
    }
