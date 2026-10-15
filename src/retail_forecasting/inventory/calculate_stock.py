"""Inventory stock calculation engine.

Implements core supply chain inventory formulas:
1. Safety Stock (SS) considering demand forecast error uncertainty and lead time.
2. Lead-Time Demand (LTD) summed over the replenishment lead time horizon.
3. Reorder Point (ROP) = Lead-Time Demand + Safety Stock.
4. Protection-Period Demand and Target Stock (Order-Up-To Level) under periodic review.
5. Suggested Order Quantity (Q) constrained to non-negative whole units.
6. Economic Order Quantity (EOQ) balancing replenishment setup and inventory holding costs.

Operational Boundary Disclaimer:
All stock levels (on-hand, on-order, backorders), lead times, and operational costs
are SIMULATED parameters. The underlying UCI dataset does not record real warehouse
inventory or supplier contracts.
"""

from typing import List, Optional, Union
import math
import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_safety_stock(
    service_level: Optional[float] = None,
    daily_forecast_error_std: Optional[float] = None,
    lead_time_days: Optional[Union[int, float]] = None,
    *,
    z_score: Optional[float] = None,
    daily_demand_std: Optional[float] = None,
    lead_time_std_days: float = 0.0,
    mean_daily_demand: float = 0.0,
) -> float:
    """Calculate buffer safety stock required to absorb demand and lead-time volatility.

    Primary Formula (Fixed Lead Time):
        SS = z * sigma_error_daily * sqrt(L)

    Generalized Formula (Variable Lead Time):
        SS = z * sqrt( (L * sigma_d^2) + (d_bar^2 * sigma_L^2) )

    Parameters
    ----------
    service_level : Optional[float]
        Target Cycle Service Level (CSL) between 0 and 1 exclusive (e.g. 0.95 for 95%).
    daily_forecast_error_std : Optional[float]
        Standard deviation of daily out-of-sample forecast errors (sigma_d >= 0).
    lead_time_days : Optional[Union[int, float]]
        Supplier replenishment lead time in days (L > 0).
    z_score : Optional[float]
        Standard normal score. If provided without service_level, used directly.
    daily_demand_std : Optional[float]
        Backwards-compatible alias for daily_forecast_error_std.
    lead_time_std_days : float
        Standard deviation of supplier lead time (sigma_L >= 0). Default is 0.0 (fixed lead time).
    mean_daily_demand : float
        Average daily demand (d_bar >= 0), required if lead_time_std_days > 0.

    Returns
    -------
    float
        Recommended safety stock units (unrounded float buffer).

    Raises
    ------
    ValueError
        If service level is not strictly between 0 and 1, or standard deviations / lead time are invalid.
    TypeError
        If inputs are non-numeric.
    """
    # Resolve aliases
    sigma_d = daily_forecast_error_std if daily_forecast_error_std is not None else daily_demand_std
    if sigma_d is None:
        raise ValueError("Must provide 'daily_forecast_error_std' or 'daily_demand_std'.")

    if lead_time_days is None:
        raise ValueError("Must provide 'lead_time_days'.")

    # Validate numeric types and finiteness
    for val, name in [
        (sigma_d, "daily_forecast_error_std"),
        (lead_time_days, "lead_time_days"),
        (lead_time_std_days, "lead_time_std_days"),
        (mean_daily_demand, "mean_daily_demand"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}")

    if sigma_d < 0:
        raise ValueError(f"Forecast error standard deviation must be non-negative, got {sigma_d}")

    if lead_time_days <= 0:
        raise ValueError(f"Lead time days must be strictly positive, got {lead_time_days}")

    if lead_time_std_days < 0:
        raise ValueError(f"Lead time standard deviation must be non-negative, got {lead_time_std_days}")

    if mean_daily_demand < 0:
        raise ValueError(f"Mean daily demand must be non-negative, got {mean_daily_demand}")

    # Determine z_score
    if service_level is not None:
        if not isinstance(service_level, (int, float, np.number)) or isinstance(service_level, bool):
            raise TypeError(f"Expected numeric value for 'service_level', got {type(service_level).__name__}")
        if not np.isfinite(service_level):
            raise ValueError(f"Service level must be finite, got {service_level}")
        if service_level <= 0.0 or service_level >= 1.0:
            raise ValueError(f"Service level must be strictly between 0.0 and 1.0 exclusive, got {service_level}")
        computed_z = float(norm.ppf(service_level))
    elif z_score is not None:
        if not isinstance(z_score, (int, float, np.number)) or isinstance(z_score, bool):
            raise TypeError(f"Expected numeric value for 'z_score', got {type(z_score).__name__}")
        if not np.isfinite(z_score) or z_score <= 0.0:
            raise ValueError(f"z_score must be a finite positive number, got {z_score}")
        computed_z = float(z_score)
    else:
        # Default 95% service level
        computed_z = float(norm.ppf(0.95))

    # Zero variability yields zero safety stock
    if sigma_d == 0.0 and lead_time_std_days == 0.0:
        return 0.0

    # Combined variance
    if lead_time_std_days > 0.0:
        variance = (lead_time_days * (sigma_d ** 2)) + ((mean_daily_demand ** 2) * (lead_time_std_days ** 2))
    else:
        variance = lead_time_days * (sigma_d ** 2)

    safety_stock = computed_z * math.sqrt(variance)
    return float(safety_stock)


def calculate_lead_time_demand(
    daily_forecasts: Union[List[float], np.ndarray, pd.Series],
    lead_time_days: int,
) -> float:
    """Calculate expected cumulative demand across the supplier replenishment lead time.

    Formula:
        LTD = sum(forecasts[0 : lead_time_days])

    Parameters
    ----------
    daily_forecasts : Union[List[float], np.ndarray, pd.Series]
        Ordered sequence of daily demand forecasts for upcoming days (t+1, t+2, ...).
    lead_time_days : int
        Replenishment lead time in days (L >= 1).

    Returns
    -------
    float
        Sum of demand forecasts across the first lead_time_days.

    Raises
    ------
    ValueError
        If lead_time_days <= 0, forecast horizon is shorter than lead_time_days,
        or any forecast value is negative or non-finite.
    TypeError
        If daily_forecasts is not an iterable sequence or lead_time_days is not integer-like.
    """
    if not isinstance(lead_time_days, (int, np.integer)) or isinstance(lead_time_days, bool):
        # Allow floats that are exact integers (e.g. 7.0)
        if isinstance(lead_time_days, float) and lead_time_days.is_integer():
            lead_time_days = int(lead_time_days)
        else:
            raise TypeError(f"lead_time_days must be an integer, got {type(lead_time_days).__name__}")

    if lead_time_days <= 0:
        raise ValueError(f"lead_time_days must be strictly positive, got {lead_time_days}")

    try:
        arr = np.asarray(daily_forecasts, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"daily_forecasts must contain numeric values: {e}")

    if arr.ndim != 1:
        arr = arr.ravel()

    if len(arr) < lead_time_days:
        raise ValueError(
            f"Available forecast horizon ({len(arr)} days) is shorter than required "
            f"lead time ({lead_time_days} days). Cannot calculate lead-time demand."
        )

    eval_window = arr[:lead_time_days]

    if not np.isfinite(eval_window).all():
        raise ValueError("daily_forecasts contains non-finite values (NaN or Inf) in the lead time window.")

    if (eval_window < 0.0).any():
        raise ValueError("daily_forecasts contains negative values, which are physically impossible for demand.")

    return float(np.sum(eval_window))


def calculate_protection_period_demand(
    daily_forecasts: Union[List[float], np.ndarray, pd.Series],
    protection_period_days: int,
) -> float:
    """Calculate expected cumulative demand across the full protection period (P = L + R).

    Formula:
        PPD = sum(forecasts[0 : protection_period_days])

    Parameters
    ----------
    daily_forecasts : Union[List[float], np.ndarray, pd.Series]
        Ordered sequence of daily demand forecasts.
    protection_period_days : int
        Combined protection horizon in days (P = L + R >= 1).

    Returns
    -------
    float
        Sum of demand forecasts across the protection period window.

    Raises
    ------
    ValueError
        If horizon is shorter than protection_period_days or contains non-finite/negative values.
    """
    return calculate_lead_time_demand(daily_forecasts, lead_time_days=protection_period_days)


def calculate_reorder_point(
    lead_time_demand: Optional[float] = None,
    safety_stock: Optional[float] = None,
    *,
    mean_daily_demand: Optional[float] = None,
    lead_time_days: Optional[float] = None,
) -> float:
    """Calculate the Reorder Point (ROP) threshold.

    When net inventory position drops to or below ROP, a replenishment order
    must be triggered.

    Formula:
        ROP = Expected Lead-Time Demand + Safety Stock

    Parameters
    ----------
    lead_time_demand : Optional[float]
        Expected demand over the supplier lead time.
    safety_stock : Optional[float]
        Calculated safety stock buffer.
    mean_daily_demand : Optional[float]
        Average daily demand (used if lead_time_demand is not directly provided).
    lead_time_days : Optional[float]
        Supplier lead time in days (used if lead_time_demand is not directly provided).

    Returns
    -------
    float
        Reorder point threshold.

    Raises
    ------
    ValueError
        If inputs are negative or non-finite.
    """
    if safety_stock is None:
        raise ValueError("Must provide 'safety_stock'.")

    if not isinstance(safety_stock, (int, float, np.number)) or isinstance(safety_stock, bool):
        raise TypeError(f"Expected numeric value for 'safety_stock', got {type(safety_stock).__name__}")

    if not np.isfinite(safety_stock) or safety_stock < 0.0:
        raise ValueError(f"safety_stock must be a finite non-negative number, got {safety_stock}")

    if lead_time_demand is None:
        if mean_daily_demand is None or lead_time_days is None:
            raise ValueError("Must provide 'lead_time_demand' or both 'mean_daily_demand' and 'lead_time_days'.")
        if not np.isfinite(mean_daily_demand) or mean_daily_demand < 0.0:
            raise ValueError(f"mean_daily_demand must be finite and non-negative, got {mean_daily_demand}")
        if not np.isfinite(lead_time_days) or lead_time_days <= 0.0:
            raise ValueError(f"lead_time_days must be finite and positive, got {lead_time_days}")
        lead_time_demand = mean_daily_demand * lead_time_days

    if not isinstance(lead_time_demand, (int, float, np.number)) or isinstance(lead_time_demand, bool):
        raise TypeError(f"Expected numeric value for 'lead_time_demand', got {type(lead_time_demand).__name__}")

    if not np.isfinite(lead_time_demand) or lead_time_demand < 0.0:
        raise ValueError(f"lead_time_demand must be a finite non-negative number, got {lead_time_demand}")

    return float(lead_time_demand + safety_stock)


def calculate_target_stock(
    protection_period_demand: float,
    safety_stock: float,
) -> float:
    """Calculate order-up-to target stock (S) for periodic review inventory policies.

    The protection period equals lead time plus review period (P = L + R).
    Target stock covers expected demand through the entire protection period
    plus the safety stock buffer sized for that protection window.

    Formula:
        Target Stock (S) = Expected Protection-Period Demand + Safety Stock

    Important Architectural Boundary:
    Lead-time demand is a subset of protection-period demand. They MUST NOT be added
    together, as doing so would double-count the lead-time portion.

    Parameters
    ----------
    protection_period_demand : float
        Sum of forecasted demand over the protection period (L + R days).
    safety_stock : float
        Safety stock buffer dimensioned for the protection period.

    Returns
    -------
    float
        Order-up-to target stock level.

    Raises
    ------
    ValueError
        If inputs are negative or non-finite.
    """
    for val, name in [
        (protection_period_demand, "protection_period_demand"),
        (safety_stock, "safety_stock"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"'{name}' must be a finite non-negative number, got {val}")

    return float(protection_period_demand + safety_stock)


def calculate_order_quantity(
    target_stock: float,
    inventory_position: float,
    rounding_mode: str = "ceil",
) -> int:
    """Calculate recommended replenishment order quantity under an order-up-to policy.

    Formula:
        Q = max(0, round(target_stock - inventory_position))

    Parameters
    ----------
    target_stock : float
        Desired order-up-to stock level (S).
    inventory_position : float
        Current net inventory position (On-Hand + On-Order - Backorders).
    rounding_mode : str
        Policy for rounding fractional physical units ('ceil', 'round', or 'floor').
        Default is 'ceil' to prevent safety-stock erosion.

    Returns
    -------
    int
        Whole-unit replenishment quantity (never negative).

    Raises
    ------
    ValueError
        If inputs are non-finite or rounding_mode is unsupported.
    """
    for val, name in [
        (target_stock, "target_stock"),
        (inventory_position, "inventory_position"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"'{name}' must be finite, got {val}")

    net_deficit = target_stock - inventory_position
    if net_deficit <= 0.0:
        return 0

    if rounding_mode == "ceil":
        rounded_q = math.ceil(net_deficit)
    elif rounding_mode == "round":
        rounded_q = round(net_deficit)
    elif rounding_mode == "floor":
        rounded_q = math.floor(net_deficit)
    else:
        raise ValueError(f"Unsupported rounding_mode '{rounding_mode}'. Choose 'ceil', 'round', or 'floor'.")

    return max(0, int(rounded_q))


def calculate_economic_order_quantity(
    annual_demand: float,
    order_cost: float,
    annual_holding_cost_per_unit: float,
) -> float:
    """Calculate Economic Order Quantity (EOQ) using the classical Wilson formula.

    Formula:
        EOQ = sqrt( (2 * D * S) / H )

    Model Assumptions & Real-World Caveats:
    - Classical EOQ assumes steady, continuous, deterministic demand.
    - It does not guarantee cost-optimal batch sizes under intermittent, lumpy,
      or seasonal retail demand. It is provided as an operational benchmark.

    Parameters
    ----------
    annual_demand : float
        Total annual demand in physical units (D >= 0).
    order_cost : float
        Fixed administrative and logistics cost per replenishment order (S > 0).
    annual_holding_cost_per_unit : float
        Cost to carry one unit of inventory for one full year (H > 0).

    Returns
    -------
    float
        Unrounded economic order batch quantity. Returns 0.0 if annual_demand == 0.

    Raises
    ------
    ValueError
        If annual_demand is negative, or order_cost / holding_cost are non-positive or non-finite.
    TypeError
        If inputs are non-numeric.
    """
    for val, name in [
        (annual_demand, "annual_demand"),
        (order_cost, "order_cost"),
        (annual_holding_cost_per_unit, "annual_holding_cost_per_unit"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"'{name}' must be finite, got {val}")

    if annual_demand < 0.0:
        raise ValueError(f"annual_demand must be non-negative, got {annual_demand}")

    if order_cost <= 0.0:
        raise ValueError(f"order_cost must be strictly positive, got {order_cost}")

    if annual_holding_cost_per_unit <= 0.0:
        raise ValueError(f"annual_holding_cost_per_unit must be strictly positive, got {annual_holding_cost_per_unit}")

    if annual_demand == 0.0:
        return 0.0

    eoq = math.sqrt((2.0 * annual_demand * order_cost) / annual_holding_cost_per_unit)
    return float(eoq)
