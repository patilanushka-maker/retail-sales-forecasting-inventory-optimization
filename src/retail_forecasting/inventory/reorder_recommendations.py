"""Reorder recommendation and inventory position evaluation module.

Evaluates current inventory positions against dynamic Reorder Points (ROP)
and produces actionable replenishment purchase order recommendations using:
1. Out-of-sample forecast error uncertainty estimation.
2. Protection-period order-up-to target stocking policy (P = L + R).
3. Strict separation between empirical retail transactions and simulated operational inputs.
4. Explicit labeling of retrospective demonstration backtests vs live operational decisions.

Operational Boundary Disclaimer:
All stock levels (on-hand, on-order, backorders), lead times, and operational costs
are SIMULATED parameters. The underlying UCI dataset does not record real warehouse
inventory or supplier contracts.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

from retail_forecasting.inventory.calculate_stock import (
    calculate_safety_stock,
    calculate_lead_time_demand,
    calculate_protection_period_demand,
    calculate_reorder_point,
    calculate_target_stock,
    calculate_order_quantity,
    calculate_economic_order_quantity,
)


def inventory_position(
    on_hand: float,
    on_order: float,
    backorders: float = 0.0,
) -> float:
    """Calculate net inventory position.

    Formula:
        IP = On-Hand + On-Order - Backorders

    Parameters
    ----------
    on_hand : float
        Physical stock currently available in warehouse (>= 0).
    on_order : float
        Stock already purchased and in transit from supplier (>= 0).
    backorders : float
        Committed customer orders currently unfulfilled due to prior stockout (>= 0).

    Returns
    -------
    float
        Net inventory position (can be negative if backorders exceed available stock).

    Raises
    ------
    ValueError
        If on_hand, on_order, or backorders are negative or non-finite.
    TypeError
        If inputs are non-numeric.
    """
    for val, name in [
        (on_hand, "on_hand"),
        (on_order, "on_order"),
        (backorders, "backorders"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}")
        if val < 0.0:
            raise ValueError(f"Stock quantity '{name}' cannot be negative, got {val}")

    return float(on_hand + on_order - backorders)


# Backwards-compatible alias for existing scaffold callers
evaluate_inventory_position = inventory_position


def estimate_forecast_uncertainty(
    forecast_df: pd.DataFrame,
    actual_col: str = "actual_demand",
    pred_col: str = "random_forest_pred",
    product_col: str = "product_id",
    min_observations: int = 7,
    fallback_policy: str = "catalog_pooled",
) -> pd.DataFrame:
    """Estimate daily forecast-error standard deviation per product from out-of-sample errors.

    Methodology:
    1. For each observation t, forecast error is defined as: e_t = actual_demand_t - pred_t.
    2. Per-product error variance is estimated with sample standard deviation (ddof=1).
    3. Products with insufficient evaluation history (< min_observations) or single observations
       are transparently assigned the catalog-wide pooled forecast error standard deviation,
       with an explicit fallback flag.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Table containing out-of-sample actuals and predictions.
    actual_col : str
        Column name for ground-truth demand.
    pred_col : str
        Column name for forecasted demand.
    product_col : str
        Product SKU identifier column name.
    min_observations : int
        Minimum out-of-sample days required for product-specific variance estimation (default 7).
    fallback_policy : str
        Uncertainty fallback strategy for sparse products ('catalog_pooled' or 'none').

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - product_id
        - error_history_count (int)
        - error_mean (float)
        - forecast_error_std (float)
        - is_uncertainty_fallback (bool)
        - uncertainty_status (str)

    Raises
    ------
    KeyError
        If required columns are missing from forecast_df.
    ValueError
        If min_observations <= 0.
    """
    if min_observations <= 0:
        raise ValueError(f"min_observations must be strictly positive, got {min_observations}")

    required = [actual_col, pred_col, product_col]
    for c in required:
        if c not in forecast_df.columns:
            raise KeyError(f"Required column '{c}' not found in forecast dataframe.")

    valid_df = forecast_df.dropna(subset=[actual_col, pred_col, product_col]).copy()
    valid_df["forecast_error"] = valid_df[actual_col].astype(float) - valid_df[pred_col].astype(float)

    # Compute catalog-wide pooled error standard deviation
    if len(valid_df) > 1:
        catalog_pooled_std = float(valid_df["forecast_error"].std(ddof=1))
    else:
        catalog_pooled_std = 0.0

    records = []
    for pid, group in valid_df.groupby(product_col):
        cnt = len(group)
        mean_err = float(group["forecast_error"].mean())

        if cnt >= 2:
            prod_std = float(group["forecast_error"].std(ddof=1))
        else:
            prod_std = float("nan")

        # Fallback decision
        if cnt >= min_observations and np.isfinite(prod_std):
            final_std = prod_std
            is_fallback = False
            status = "PRODUCT_SPECIFIC_ESTIMATE"
        else:
            if fallback_policy == "catalog_pooled":
                final_std = catalog_pooled_std
                is_fallback = True
                status = "CATALOG_POOLED_FALLBACK"
            else:
                final_std = float("nan")
                is_fallback = True
                status = "INSUFFICIENT_HISTORY"

        records.append({
            product_col: pid,
            "error_history_count": cnt,
            "error_mean": round(mean_err, 4),
            "forecast_error_std": round(final_std, 4) if np.isfinite(final_std) else float("nan"),
            "is_uncertainty_fallback": is_fallback,
            "uncertainty_status": status,
        })

    uncertainty_df = pd.DataFrame(records)
    return uncertainty_df


def generate_inventory_recommendations(
    forecast_df: pd.DataFrame,
    inventory_state_df: Optional[pd.DataFrame] = None,
    uncertainty_df: Optional[pd.DataFrame] = None,
    service_level: float = 0.95,
    lead_time_days: int = 7,
    review_period_days: int = 1,
    simulated_order_cost: float = 50.0,
    simulated_holding_cost: float = 2.0,
    pred_col: str = "random_forest_pred",
    actual_col: str = "actual_demand",
    product_col: str = "product_id",
    date_col: str = "date",
    rounding_mode: str = "ceil",
    min_observations_uncertainty: int = 7,
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    is_retrospective_demonstration: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate inventory replenishment recommendations for products under periodic review.

    Integrates:
    - Daily forecast horizon per product.
    - Out-of-sample forecast error uncertainty.
    - Protection-period order-up-to stocking policy (P = L + R).
    - Clear separation between actual historical series and simulated inventory parameters.
    - Explicit date semantics and retrospective demonstration labeling.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Daily product forecasts.
    inventory_state_df : Optional[pd.DataFrame]
        Table containing simulated stock state per product: [product_id, on_hand, on_order, backorders].
        If None, default simulated states are deterministically assigned.
    uncertainty_df : Optional[pd.DataFrame]
        Pre-calculated forecast error uncertainty table. If None, estimated from forecast_df.
    service_level : float
        Target cycle service level (default 0.95).
    lead_time_days : int
        Replenishment lead time in days (default 7).
    review_period_days : int
        Review period interval in days (default 1 for daily review).
    simulated_order_cost : float
        Simulated fixed cost per purchase order in monetary units (default 50.0).
    simulated_holding_cost : float
        Simulated annual holding cost per unit (default 2.0).
    pred_col : str
        Column name for daily predictions.
    actual_col : str
        Column name for ground-truth actuals (used strictly for uncertainty estimation).
    product_col : str
        Product SKU column name.
    date_col : str
        Date column name.
    rounding_mode : str
        Rounding rule for order quantities ('ceil', 'round', or 'floor').
    min_observations_uncertainty : int
        Minimum error sample size for product-specific uncertainty estimation.
    as_of_date : Optional[Union[str, pd.Timestamp]]
        Effective planning cutoff date. Inferred from min(date) - 1 day if not supplied.
    is_retrospective_demonstration : bool
        Flag indicating that inputs stem from historical evaluation backtest (default True).
    random_state : int
        Seed for reproducible simulated state generation when inventory_state_df is not supplied.

    Returns
    -------
    pd.DataFrame
        Complete inventory recommendations table with operational indicators.

    Raises
    ------
    ValueError
        If pred_col equals actual_col, or parameters are invalid.
    """
    if pred_col == actual_col:
        raise ValueError(
            f"Invalid configuration: 'pred_col' ('{pred_col}') cannot be identical to 'actual_col' ('{actual_col}'). "
            "The inventory engine must not use ground-truth actual demand as the future forecast."
        )

    if lead_time_days <= 0:
        raise ValueError(f"lead_time_days must be strictly positive, got {lead_time_days}")
    if review_period_days <= 0:
        raise ValueError(f"review_period_days must be strictly positive, got {review_period_days}")
    if service_level <= 0.0 or service_level >= 1.0:
        raise ValueError(f"service_level must be strictly between 0 and 1 exclusive, got {service_level}")

    protection_period_days = lead_time_days + review_period_days
    z = float(norm.ppf(service_level))

    # 1. Resolve uncertainty estimates
    if uncertainty_df is None:
        if actual_col in forecast_df.columns:
            uncertainty_df = estimate_forecast_uncertainty(
                forecast_df,
                actual_col=actual_col,
                pred_col=pred_col,
                product_col=product_col,
                min_observations=min_observations_uncertainty,
                fallback_policy="catalog_pooled",
            )
        else:
            raise ValueError(
                f"Cannot estimate forecast uncertainty: column '{actual_col}' not in forecast_df, "
                "and no uncertainty_df was provided."
            )

    unc_map = uncertainty_df.set_index(product_col).to_dict(orient="index")

    # 2. Resolve simulated inventory states
    if inventory_state_df is not None:
        inv_map = inventory_state_df.set_index(product_col).to_dict(orient="index")
    else:
        inv_map = {}

    rng = np.random.RandomState(random_state)
    records = []

    rec_type_label = "RETROSPECTIVE_DEMONSTRATION" if is_retrospective_demonstration else "OPERATIONAL_RECOMMENDATION"

    # Process each product entity
    for pid, group in forecast_df.groupby(product_col):
        sorted_grp = group.sort_values(by=date_col).reset_index(drop=True)
        horizon_len = len(sorted_grp)
        daily_preds = sorted_grp[pred_col].values

        # Date boundaries
        if not sorted_grp.empty and date_col in sorted_grp.columns:
            min_dt = pd.to_datetime(sorted_grp[date_col].min())
            max_dt = pd.to_datetime(sorted_grp[date_col].max())
            f_start = str(min_dt.date())
            f_end = str(max_dt.date())
            if as_of_date is not None:
                eff_as_of = str(pd.to_datetime(as_of_date).date())
            else:
                eff_as_of = str((min_dt - pd.Timedelta(days=1)).date())
        else:
            f_start = "N/A"
            f_end = "N/A"
            eff_as_of = str(as_of_date) if as_of_date is not None else "N/A"

        # Retrieve uncertainty metrics
        unc_entry = unc_map.get(pid, {
            "forecast_error_std": float("nan"),
            "error_history_count": 0,
            "is_uncertainty_fallback": True,
            "uncertainty_status": "NOT_EVALUATED",
        })
        sigma_err = unc_entry.get("forecast_error_std", float("nan"))
        err_count = unc_entry.get("error_history_count", 0)
        is_fallback = unc_entry.get("is_uncertainty_fallback", True)

        # Base metadata payload
        base_record = {
            product_col: pid,
            "as_of_date": eff_as_of,
            "forecast_start_date": f_start,
            "forecast_end_date": f_end,
            "is_retrospective_demonstration": is_retrospective_demonstration,
            "recommendation_type": rec_type_label,
            "forecast_model": pred_col,
            "forecast_horizon_days": horizon_len,
            "service_level": service_level,
            "z_score": round(z, 4),
            "lead_time_days": lead_time_days,
            "review_period_days": review_period_days,
            "protection_period_days": protection_period_days,
            "is_simulated_operational_input": True,
        }

        # Check horizon sufficiency for protection period
        if horizon_len < protection_period_days:
            base_record.update({
                "expected_daily_demand": round(float(np.mean(daily_preds)), 4) if horizon_len > 0 else 0.0,
                "expected_lead_time_demand": float("nan"),
                "expected_protection_period_demand": float("nan"),
                "forecast_error_std": sigma_err,
                "error_history_count": err_count,
                "is_uncertainty_fallback": is_fallback,
                "safety_stock": float("nan"),
                "reorder_point": float("nan"),
                "simulated_on_hand": float("nan"),
                "simulated_on_order": float("nan"),
                "simulated_backorders": float("nan"),
                "inventory_position": float("nan"),
                "target_stock": float("nan"),
                "suggested_order_qty": 0,
                "reorder_triggered": False,
                "economic_order_quantity": float("nan"),
                "recommendation_status": "UNAVAILABLE_INSUFFICIENT_HORIZON",
            })
            records.append(base_record)
            continue

        # Validate daily forecasts in the evaluation window
        try:
            ltd = calculate_lead_time_demand(daily_preds, lead_time_days)
            pp_demand = calculate_protection_period_demand(daily_preds, protection_period_days)
        except ValueError:
            # Forecast contains non-finite or negative values
            base_record.update({
                "expected_daily_demand": float("nan"),
                "expected_lead_time_demand": float("nan"),
                "expected_protection_period_demand": float("nan"),
                "forecast_error_std": sigma_err,
                "error_history_count": err_count,
                "is_uncertainty_fallback": is_fallback,
                "safety_stock": float("nan"),
                "reorder_point": float("nan"),
                "simulated_on_hand": float("nan"),
                "simulated_on_order": float("nan"),
                "simulated_backorders": float("nan"),
                "inventory_position": float("nan"),
                "target_stock": float("nan"),
                "suggested_order_qty": 0,
                "reorder_triggered": False,
                "economic_order_quantity": float("nan"),
                "recommendation_status": "UNAVAILABLE_INVALID_FORECAST",
            })
            records.append(base_record)
            continue

        # Check uncertainty validity
        if not np.isfinite(sigma_err) or sigma_err < 0.0:
            # Missing or non-finite uncertainty must not silently become zero safety stock
            base_record.update({
                "expected_daily_demand": round(float(np.mean(daily_preds)), 4),
                "expected_lead_time_demand": round(ltd, 4),
                "expected_protection_period_demand": round(pp_demand, 4),
                "forecast_error_std": float("nan"),
                "error_history_count": err_count,
                "is_uncertainty_fallback": is_fallback,
                "safety_stock": float("nan"),
                "reorder_point": float("nan"),
                "simulated_on_hand": float("nan"),
                "simulated_on_order": float("nan"),
                "simulated_backorders": float("nan"),
                "inventory_position": float("nan"),
                "target_stock": float("nan"),
                "suggested_order_qty": 0,
                "reorder_triggered": False,
                "economic_order_quantity": float("nan"),
                "recommendation_status": "UNAVAILABLE_MISSING_UNCERTAINTY",
            })
            records.append(base_record)
            continue

        # Sizing safety stocks
        ss_lt = calculate_safety_stock(
            service_level=service_level,
            daily_forecast_error_std=sigma_err,
            lead_time_days=lead_time_days,
        )
        ss_pp = calculate_safety_stock(
            service_level=service_level,
            daily_forecast_error_std=sigma_err,
            lead_time_days=protection_period_days,
        )

        # ROP and Target Stock (S)
        rop = calculate_reorder_point(lead_time_demand=ltd, safety_stock=ss_lt)
        target_stock = calculate_target_stock(
            protection_period_demand=pp_demand,
            safety_stock=ss_pp,
        )

        # Retrieve or simulate current stock position
        if pid in inv_map:
            p_inv = inv_map[pid]
            on_hand = float(p_inv.get("on_hand", 0.0))
            on_order = float(p_inv.get("on_order", 0.0))
            backorders = float(p_inv.get("backorders", 0.0))
        else:
            # Deterministic simulation based on ROP fraction
            # E.g., simulate on-hand stock between 0.4 * ROP and 1.4 * ROP
            stock_factor = float(rng.uniform(0.4, 1.4))
            on_hand = round(max(0.0, rop * stock_factor), 1)
            on_order = 0.0
            backorders = 0.0

        ip = inventory_position(on_hand=on_hand, on_order=on_order, backorders=backorders)

        # Order Trigger and Quantity
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

        # Annualized demand for EOQ
        mean_daily = float(np.mean(daily_preds))
        annual_demand = max(0.0, mean_daily * 365.0)
        eoq = calculate_economic_order_quantity(
            annual_demand=annual_demand,
            order_cost=simulated_order_cost,
            annual_holding_cost_per_unit=simulated_holding_cost,
        )

        base_record.update({
            "expected_daily_demand": round(mean_daily, 4),
            "expected_lead_time_demand": round(ltd, 4),
            "expected_protection_period_demand": round(pp_demand, 4),
            "forecast_error_std": round(sigma_err, 4),
            "error_history_count": err_count,
            "is_uncertainty_fallback": is_fallback,
            "safety_stock": round(ss_lt, 4),
            "reorder_point": round(rop, 4),
            "simulated_on_hand": on_hand,
            "simulated_on_order": on_order,
            "simulated_backorders": backorders,
            "inventory_position": round(ip, 4),
            "target_stock": round(target_stock, 4),
            "suggested_order_qty": order_qty,
            "reorder_triggered": reorder_triggered,
            "economic_order_quantity": round(eoq, 2),
            "recommendation_status": rec_status,
        })
        records.append(base_record)

    recommendations_df = pd.DataFrame(records)
    return recommendations_df


def generate_replenishment_orders(
    inventory_df: pd.DataFrame,
    reorder_point_col: str = "reorder_point",
    inventory_pos_col: str = "inventory_position",
    order_qty_col: str = "suggested_order_qty",
    product_col: str = "product_id",
) -> pd.DataFrame:
    """Filter inventory table to items requiring active replenishment purchase orders.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Table with inventory status and ROP values per product.
    reorder_point_col : str
        Column name for ROP threshold.
    inventory_pos_col : str
        Column name for current net inventory position.
    order_qty_col : str
        Column name for recommended order quantity.
    product_col : str
        Product SKU identifier column name.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe of items where inventory_position <= reorder_point
        and suggested_order_qty > 0.
    """
    required = [reorder_point_col, inventory_pos_col, order_qty_col]
    for c in required:
        if c not in inventory_df.columns:
            raise KeyError(f"Required column '{c}' not found in inventory dataframe.")

    # Items requiring reorder: IP <= ROP and order quantity > 0
    trigger_mask = (
        (inventory_df[inventory_pos_col] <= inventory_df[reorder_point_col]) &
        (inventory_df[order_qty_col] > 0)
    )
    orders_df = inventory_df[trigger_mask].copy().reset_index(drop=True)
    return orders_df
