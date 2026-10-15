"""DataCo inventory optimization engine and decision-support module.

Implements inventory optimization policies tailored for DataCo retail demand:
1. Inventory position accounting: IP = on_hand + on_order - backorders
2. Safety stock sizing using normal quantile approximation and forecast error uncertainty:
   SS = z * sigma_error_daily * sqrt(lead_time_days)
3. Expected lead-time demand (LTD) and protection-period demand (PPD = L + R)
4. Reorder Point: ROP = LTD + SS
5. Target Stock (Order-Up-To Level): S = PPD + SS
6. Suggested Replenishment Quantity: Q = max(0, ceil(S - IP))
7. Optional Economic Order Quantity (EOQ) benchmarking

Operational Boundary & Data Invariants:
- DataCo logs historical customer transaction lines and outbound carrier shipping days.
- Inbound supplier replenishment lead time, physical warehouse stock-on-hand, dock receipts,
  purchase orders, and stockouts are NOT present in DataCo.
- All stock levels and supplier lead times in this module are SIMULATED operational parameters.
- No future actual demand is accessed or leaked when computing inventory recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_inventory_position(
    on_hand: float,
    on_order: float,
    backorders: float = 0.0
) -> float:
    """Calculate net inventory position.

    Formula:
        IP = on_hand + on_order - backorders

    Parameters
    ----------
    on_hand : float
        Current physical warehouse stock (must be >= 0).
    on_order : float
        Quantity currently in transit from supplier (must be >= 0).
    backorders : float
        Unfulfilled customer orders committed but delayed due to prior stockout (must be >= 0).

    Returns
    -------
    float
        Net inventory position (can be negative if backorders exceed stock).

    Raises
    ------
    ValueError
        If inputs are negative or non-finite.
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


def calculate_dataco_safety_stock(
    service_level: float,
    daily_forecast_error_std: float,
    lead_time_days: Union[int, float],
    lead_time_std_days: float = 0.0,
    mean_daily_demand: float = 0.0
) -> float:
    """Calculate safety stock buffer using normal quantile approximation.

    Formula (Fixed Lead Time):
        SS = z * sigma_error_daily * sqrt(L)
    Formula (Variable Lead Time):
        SS = z * sqrt( (L * sigma_error_daily^2) + (mean_daily_demand^2 * sigma_L^2) )

    Parameters
    ----------
    service_level : float
        Cycle Service Level target strictly between 0.0 and 1.0 (exclusive).
    daily_forecast_error_std : float
        Estimated standard deviation of daily out-of-sample forecast errors (sigma_e >= 0).
    lead_time_days : Union[int, float]
        Assumed replenishment lead time in days (L > 0).
    lead_time_std_days : float
        Standard deviation of lead time in days (default 0.0 for fixed lead time).
    mean_daily_demand : float
        Average daily demand (d_bar >= 0), required if lead_time_std_days > 0.

    Returns
    -------
    float
        Recommended safety stock buffer units.

    Raises
    ------
    ValueError
        If service_level is outside (0, 1), or parameters are negative / non-finite.
    TypeError
        If inputs are non-numeric.
    """
    for val, name in [
        (service_level, "service_level"),
        (daily_forecast_error_std, "daily_forecast_error_std"),
        (lead_time_days, "lead_time_days"),
        (lead_time_std_days, "lead_time_std_days"),
        (mean_daily_demand, "mean_daily_demand"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}")

    if not (0.0 < service_level < 1.0):
        raise ValueError(
            f"Service level must be strictly between 0.0 and 1.0 exclusive, got {service_level}"
        )

    if daily_forecast_error_std < 0.0:
        raise ValueError(
            f"Forecast error std must be non-negative, got {daily_forecast_error_std}"
        )

    if lead_time_days <= 0.0:
        raise ValueError(f"Lead time days must be strictly positive, got {lead_time_days}")

    if lead_time_std_days < 0.0:
        raise ValueError(f"Lead time std must be non-negative, got {lead_time_std_days}")

    if mean_daily_demand < 0.0:
        raise ValueError(f"Mean daily demand must be non-negative, got {mean_daily_demand}")

    z = float(norm.ppf(service_level))

    if lead_time_std_days > 0.0:
        combined_variance = (lead_time_days * (daily_forecast_error_std ** 2)) + (
            (mean_daily_demand ** 2) * (lead_time_std_days ** 2)
        )
        return float(z * math.sqrt(combined_variance))

    return float(z * daily_forecast_error_std * math.sqrt(lead_time_days))


def calculate_dataco_reorder_point(
    expected_lead_time_demand: float,
    safety_stock: float
) -> float:
    """Calculate Reorder Point (ROP).

    Formula:
        ROP = expected_lead_time_demand + safety_stock
    """
    for val, name in [
        (expected_lead_time_demand, "expected_lead_time_demand"),
        (safety_stock, "safety_stock"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}")
        if val < 0.0:
            raise ValueError(f"Parameter '{name}' cannot be negative, got {val}")

    return float(expected_lead_time_demand + safety_stock)


def calculate_dataco_target_stock(
    expected_protection_period_demand: float,
    safety_stock: float
) -> float:
    """Calculate Target Stock Level (Order-Up-To Level, S).

    Formula:
        S = expected_protection_period_demand + safety_stock
    """
    for val, name in [
        (expected_protection_period_demand, "expected_protection_period_demand"),
        (safety_stock, "safety_stock"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}")
        if val < 0.0:
            raise ValueError(f"Parameter '{name}' cannot be negative, got {val}")

    return float(expected_protection_period_demand + safety_stock)


def calculate_dataco_order_quantity(
    target_stock: float,
    inventory_position: float,
    reorder_point: Optional[float] = None,
    rounding_mode: str = "ceil"
) -> int:
    """Calculate suggested replenishment order quantity (Q).

    Under periodic review:
        If IP <= ROP (or if ROP not provided, whenever S > IP):
            Q = max(0, target_stock - inventory_position)
        Else:
            Q = 0
    """
    for val, name in [
        (target_stock, "target_stock"),
        (inventory_position, "inventory_position"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}")

    if target_stock < 0.0:
        raise ValueError(f"Target stock cannot be negative, got {target_stock}")

    # Check reorder trigger condition if ROP is supplied
    if reorder_point is not None:
        if not isinstance(reorder_point, (int, float, np.number)) or isinstance(reorder_point, bool):
            raise TypeError(f"Expected numeric value for 'reorder_point', got {type(reorder_point).__name__}")
        if not np.isfinite(reorder_point):
            raise ValueError(f"Parameter 'reorder_point' must be finite, got {reorder_point}")
        if inventory_position > reorder_point:
            return 0

    raw_diff = target_stock - inventory_position
    if raw_diff <= 0.0:
        return 0

    if rounding_mode == "ceil":
        return int(math.ceil(raw_diff))
    elif rounding_mode == "floor":
        return int(math.floor(raw_diff))
    elif rounding_mode == "round":
        return int(round(raw_diff))
    else:
        raise ValueError(f"Unsupported rounding_mode '{rounding_mode}'. Use 'ceil', 'floor', or 'round'.")


def calculate_dataco_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_annual_per_unit: float
) -> float:
    """Calculate Economic Order Quantity (EOQ).

    Formula:
        EOQ = sqrt( (2 * D * S) / H )
    """
    for val, name in [
        (annual_demand, "annual_demand"),
        (order_cost, "order_cost"),
        (holding_cost_annual_per_unit, "holding_cost_annual_per_unit"),
    ]:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise TypeError(f"Expected numeric value for '{name}', got {type(val).__name__}")
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' must be finite, got {val}")
        if val <= 0.0:
            raise ValueError(f"Parameter '{name}' must be strictly positive, got {val}")

    return float(math.sqrt((2.0 * annual_demand * order_cost) / holding_cost_annual_per_unit))


def estimate_dataco_forecast_uncertainty(
    historical_forecasts_df: pd.DataFrame,
    forecast_origin_date: Optional[str] = "2017-08-31",
    pred_col: str = "random_forest_pred",
    actual_col: str = "actual_demand",
    product_col: str = "product_id",
    date_col: str = "date",
    min_observations: int = 14,
    method: str = "sample_std",
) -> pd.DataFrame:
    """Estimate daily forecast error uncertainty per product using pre-decision residuals.

    Strict Leakage Prevention:
    - If forecast_origin_date is provided, only residuals strictly on or before
      forecast_origin_date are used.
    - Zero future actuals (e.g. post-decision test observations) are included.

    Statistical Method:
    - 'sample_std': Sample standard deviation of residuals (ddof=1), measuring error dispersion:
      s_e = sqrt( (1 / (N - 1)) * sum( (e_i - e_bar)^2 ) )
    - 'rmse': Root Mean Squared Error, combining residual dispersion and systematic bias:
      RMSE = sqrt( (1 / N) * sum( e_i^2 ) )

    Parameters
    ----------
    historical_forecasts_df : pd.DataFrame
        DataFrame containing historical actuals and model predictions.
    forecast_origin_date : Optional[str]
        Cutoff decision date (e.g. '2017-08-31'). Observations with date > forecast_origin_date
        are strictly excluded.
    pred_col : str
        Prediction column name (or 'prediction').
    actual_col : str
        Actual demand column name.
    product_col : str
        Product SKU identifier column name.
    date_col : str
        Date column name.
    min_observations : int
        Minimum error sample size required to estimate reliable product-specific uncertainty.
    method : str
        Uncertainty calculation method ('sample_std' or 'rmse').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        [product_id, sample_size, error_mean, error_std, error_rmse,
         uncertainty_value, uncertainty_status, is_fallback, min_date, max_date]
    """
    if method not in ("sample_std", "rmse"):
        raise ValueError(f"Unsupported method '{method}'. Choose 'sample_std' or 'rmse'.")

    df = historical_forecasts_df.copy()
    p_col = pred_col if pred_col in df.columns else "prediction"
    if p_col not in df.columns:
        raise ValueError(f"Neither '{pred_col}' nor 'prediction' found in DataFrame columns: {df.columns.tolist()}")

    if actual_col not in df.columns:
        raise ValueError(f"Actual demand column '{actual_col}' not found in DataFrame.")

    clean_df = df.dropna(subset=[p_col, actual_col]).copy()
    clean_df[product_col] = clean_df[product_col].astype(str)

    # Strictly filter by decision origin date to prevent lookahead leakage
    if forecast_origin_date is not None and date_col in clean_df.columns:
        origin_dt = pd.to_datetime(forecast_origin_date).floor("D")
        clean_df[date_col] = pd.to_datetime(clean_df[date_col]).dt.floor("D")
        clean_df = clean_df[clean_df[date_col] <= origin_dt]

    if clean_df.empty:
        # Return empty frame with expected structure
        return pd.DataFrame(columns=[
            product_col, "sample_size", "error_mean", "error_std", "error_rmse",
            "uncertainty_value", "uncertainty_status", "is_fallback", "min_date", "max_date"
        ])

    clean_df["forecast_error"] = clean_df[actual_col].astype(float) - clean_df[p_col].astype(float)

    # Compute catalog-pooled error statistics across pre-origin records as fallback
    n_total = len(clean_df)
    pooled_std = float(clean_df["forecast_error"].std(ddof=1)) if n_total > 1 else 0.0
    pooled_rmse = float(np.sqrt(np.mean(clean_df["forecast_error"] ** 2))) if n_total > 0 else 0.0

    records = []
    # Process each observed product in clean_df
    for pid, group in clean_df.groupby(product_col):
        n = len(group)
        errors = group["forecast_error"].astype(float)
        mean_err = float(errors.mean())
        std_err = float(errors.std(ddof=1)) if n > 1 else 0.0
        rmse_err = float(np.sqrt(np.mean(errors ** 2)))

        min_dt_str = group[date_col].min().strftime("%Y-%m-%d") if date_col in group.columns else None
        max_dt_str = group[date_col].max().strftime("%Y-%m-%d") if date_col in group.columns else None

        if n >= min_observations:
            chosen_uncert = std_err if method == "sample_std" else rmse_err
            status = "reliable_sample"
            is_fallback = False
        elif n > 0:
            chosen_uncert = pooled_std if method == "sample_std" else pooled_rmse
            status = "insufficient_sample_pooled_fallback"
            is_fallback = True
        else:
            chosen_uncert = np.nan
            status = "UNAVAILABLE"
            is_fallback = False

        records.append({
            product_col: str(pid),
            "sample_size": n,
            "error_mean": round(mean_err, 4),
            "error_std": round(std_err, 4),
            "error_rmse": round(rmse_err, 4),
            "uncertainty_value": round(chosen_uncert, 4) if np.isfinite(chosen_uncert) else np.nan,
            "uncertainty_status": status,
            "is_fallback": is_fallback,
            "min_date": min_dt_str,
            "max_date": max_dt_str,
        })

    return pd.DataFrame(records)


def generate_dataco_inventory_recommendations(
    forecast_comparison_df: pd.DataFrame,
    catalog_df: Optional[pd.DataFrame] = None,
    uncertainty_df: Optional[pd.DataFrame] = None,
    simulated_state_df: Optional[pd.DataFrame] = None,
    service_level_target: float = 0.95,
    lead_time_days: int = 7,
    review_period_days: int = 1,
    simulated_default_on_hand: float = 50.0,
    simulated_default_on_order: float = 0.0,
    simulated_default_backorders: float = 0.0,
    simulated_order_cost_fixed: float = 50.0,
    simulated_holding_cost_annual_per_unit: float = 2.0,
    rounding_mode: str = "ceil",
    forecast_origin_date: str = "2017-08-31",
    forecast_horizon_days: int = 8,
    pred_col: str = "predicted_demand",
    product_col: str = "product_id",
    date_col: str = "date",
    uncertainty_method: str = "sample_std"
) -> pd.DataFrame:
    """Generate reproducible replenishment recommendations for DataCo cohort.

    Guarantees:
    - Uses fixed-origin forecasts over the protection horizon (lead_time + review_period).
    - Uses pre-decision uncertainty estimates (no lookahead into test period).
    - Invalid or incomplete forecasts produce UNAVAILABLE recommendations instead of silent zeros.
    - All inventory states are clearly labelled as simulated.

    Parameters
    ----------
    forecast_comparison_df : pd.DataFrame
        Fixed-origin forecasts or forecast comparison table.
    catalog_df : Optional[pd.DataFrame]
        Product catalog metadata.
    uncertainty_df : Optional[pd.DataFrame]
        Pre-computed uncertainty table. If None, estimated strictly on data <= forecast_origin_date.
    simulated_state_df : Optional[pd.DataFrame]
        Optional simulated stock state per product.
    service_level_target : float
        Target Cycle Service Level (default 0.95).
    lead_time_days : int
        Replenishment lead time in days (default 7).
    review_period_days : int
        Review period interval in days (default 1).
    simulated_default_on_hand : float
        Default on-hand stock.
    simulated_default_on_order : float
        Default on-order stock.
    simulated_default_backorders : float
        Default committed backorders.
    simulated_order_cost_fixed : float
        Simulated fixed cost per order ($50.0).
    simulated_holding_cost_annual_per_unit : float
        Simulated holding cost per unit per year ($2.0).
    rounding_mode : str
        Rounding mode ('ceil', 'floor', 'round').
    forecast_origin_date : str
        Decision date origin for this recommendation scenario.
    forecast_horizon_days : int
        Required forward horizon days (must be >= lead_time_days + review_period_days).
    pred_col : str
        Forecast column name ('predicted_demand' or 'random_forest_pred').
    product_col : str
        Product SKU ID column.
    date_col : str
        Date column ('forecast_date' or 'date').
    uncertainty_method : str
        'sample_std' (dispersion) or 'rmse'.

    Returns
    -------
    pd.DataFrame
        Complete recommendation table with full input traceability and status indicators.
    """
    if forecast_comparison_df.empty:
        raise ValueError("forecast_comparison_df is empty.")

    origin_dt = pd.to_datetime(forecast_origin_date).floor("D")
    protection_period_days = lead_time_days + review_period_days

    # Identify date column header
    dt_col = "forecast_date" if "forecast_date" in forecast_comparison_df.columns else date_col
    # Identify prediction column header
    p_col = pred_col if pred_col in forecast_comparison_df.columns else (
        "predicted_demand" if "predicted_demand" in forecast_comparison_df.columns else "random_forest_pred"
    )

    fc_df = forecast_comparison_df.copy()
    fc_df[product_col] = fc_df[product_col].astype(str)
    if dt_col in fc_df.columns:
        fc_df[dt_col] = pd.to_datetime(fc_df[dt_col]).dt.floor("D")

    # If forecast_origin column exists, filter to records matching forecast_origin_date
    if "forecast_origin" in fc_df.columns:
        origin_str = origin_dt.strftime("%Y-%m-%d")
        fc_origin_filtered = fc_df[fc_df["forecast_origin"] == origin_str]
        if not fc_origin_filtered.empty:
            fc_df = fc_origin_filtered

    # 1. Uncertainty map
    if uncertainty_df is None:
        uncertainty_df = estimate_dataco_forecast_uncertainty(
            historical_forecasts_df=fc_df,
            forecast_origin_date=forecast_origin_date,
            pred_col=p_col,
            product_col=product_col,
            date_col=dt_col,
            method=uncertainty_method
        )

    uncertainty_map: Dict[str, Tuple[float, str]] = {}
    if not uncertainty_df.empty:
        for _, row in uncertainty_df.iterrows():
            pid = str(row[product_col])
            u_val = row.get("uncertainty_value", row.get("error_std", np.nan))
            status = row.get("uncertainty_status", "unknown")
            uncertainty_map[pid] = (float(u_val) if pd.notnull(u_val) else np.nan, str(status))

    # 2. Build simulated state map
    state_map: Dict[str, Tuple[float, float, float]] = {}
    if simulated_state_df is not None:
        for _, row in simulated_state_df.iterrows():
            pid = str(row[product_col])
            oh = float(row.get("on_hand", simulated_default_on_hand))
            oo = float(row.get("on_order", simulated_default_on_order))
            bo = float(row.get("backorders", simulated_default_backorders))
            state_map[pid] = (oh, oo, bo)

    # 3. Catalog map
    name_map = {}
    price_map = {}
    if catalog_df is not None:
        for _, row in catalog_df.iterrows():
            pid = str(row[product_col])
            name_map[pid] = str(row.get("product_name", f"SKU_{pid}"))
            price_map[pid] = float(row.get("product_price", 0.0))

    # 4. Generate recommendations per product
    pids = sorted(list(fc_df[product_col].unique()))
    forward_dates = pd.date_range(
        start=origin_dt + pd.Timedelta(days=1),
        periods=protection_period_days,
        freq="D"
    )

    recommendations = []
    simulated_note = (
        "SIMULATED_PARAMETERS: Inventory positions (on_hand, on_order, backorders), supplier lead time, "
        "and costs are simulated demonstration inputs. DataCo does not contain warehouse stock or supplier PO records."
    )

    for pid in pids:
        p_name = name_map.get(pid, f"SKU_{pid}")
        p_fc = fc_df[fc_df[product_col] == pid].copy()

        # Check forward forecast records strictly after forecast origin
        if "forecast_horizon_step" in p_fc.columns:
            p_fc = p_fc.sort_values(by="forecast_horizon_step")
            p_forward = p_fc[p_fc["forecast_horizon_step"] >= 1]
        elif dt_col in p_fc.columns:
            p_fc = p_fc.sort_values(by=dt_col)
            p_forward = p_fc[p_fc[dt_col].isin(forward_dates)]
        else:
            p_forward = p_fc

        # Validate that predictions for lead time and protection period exist and are valid
        has_complete_horizon = (
            len(p_forward) >= protection_period_days and
            not p_forward.iloc[:protection_period_days][p_col].isna().any() and
            np.all(np.isfinite(p_forward.iloc[:protection_period_days][p_col].values))
        )

        # Retrieve uncertainty
        sigma_e, uncert_status = uncertainty_map.get(pid, (np.nan, "UNAVAILABLE"))

        # Simulated Inventory State
        on_hand, on_order, backorders = state_map.get(
            pid,
            (simulated_default_on_hand, simulated_default_on_order, simulated_default_backorders)
        )
        try:
            ip = calculate_inventory_position(on_hand, on_order, backorders)
        except Exception:
            ip = np.nan

        # If forecast is incomplete or uncertainty is unavailable, return UNAVAILABLE recommendation
        if not has_complete_horizon or not np.isfinite(sigma_e) or uncert_status == "UNAVAILABLE":
            warning_msg = []
            if not has_complete_horizon:
                warning_msg.append(
                    f"INCOMPLETE_FORECAST_HORIZON: Expected {protection_period_days} forward days from origin {origin_dt.strftime('%Y-%m-%d')}, found {len(p_forward)}."
                )
            if not np.isfinite(sigma_e) or uncert_status == "UNAVAILABLE":
                warning_msg.append(
                    f"UNAVAILABLE_UNCERTAINTY: Insufficient historical residuals on or before {origin_dt.strftime('%Y-%m-%d')}."
                )

            recommendations.append({
                "product_id": pid,
                "product_name": p_name,
                "forecast_origin": str(origin_dt.date()),
                "forecast_horizon_days": protection_period_days,
                "expected_lead_time_demand": np.nan,
                "expected_protection_period_demand": np.nan,
                "safety_stock": np.nan,
                "reorder_point": np.nan,
                "target_stock_level": np.nan,
                "simulated_on_hand": round(on_hand, 2) if np.isfinite(on_hand) else np.nan,
                "simulated_on_order": round(on_order, 2) if np.isfinite(on_order) else np.nan,
                "simulated_backorders": round(backorders, 2) if np.isfinite(backorders) else np.nan,
                "inventory_position": round(ip, 2) if np.isfinite(ip) else np.nan,
                "reorder_triggered": False,
                "recommended_order_quantity": None,
                "optional_eoq_benchmark": np.nan,
                "service_level_target": service_level_target,
                "assumed_lead_time_days": lead_time_days,
                "review_period_days": review_period_days,
                "forecast_error_std": round(sigma_e, 4) if np.isfinite(sigma_e) else np.nan,
                "recommendation_status": "UNAVAILABLE",
                "warning_or_limitation": " | ".join(warning_msg) + " | " + simulated_note,
            })
            continue

        # Complete and valid forward forecasts
        ltd = float(p_forward.iloc[:lead_time_days][p_col].sum())
        ppd = float(p_forward.iloc[:protection_period_days][p_col].sum())

        # Safety Stock for lead time and protection period
        ss_lead = calculate_dataco_safety_stock(
            service_level=service_level_target,
            daily_forecast_error_std=sigma_e,
            lead_time_days=lead_time_days,
        )
        ss_protection = calculate_dataco_safety_stock(
            service_level=service_level_target,
            daily_forecast_error_std=sigma_e,
            lead_time_days=protection_period_days,
        )

        rop = calculate_dataco_reorder_point(ltd, ss_lead)
        target_stock = calculate_dataco_target_stock(ppd, ss_protection)

        q = calculate_dataco_order_quantity(
            target_stock=target_stock,
            inventory_position=ip,
            reorder_point=rop,
            rounding_mode=rounding_mode
        )

        reorder_triggered = bool(ip <= rop)
        status = "REORDER_TRIGGERED" if reorder_triggered else "ADEQUATELY_STOCKED"

        # Optional EOQ benchmark
        daily_rate = ltd / lead_time_days
        annual_demand = max(1.0, daily_rate * 365.0)
        try:
            eoq = round(calculate_dataco_eoq(
                annual_demand=annual_demand,
                order_cost=simulated_order_cost_fixed,
                holding_cost_annual_per_unit=simulated_holding_cost_annual_per_unit
            ), 1)
        except Exception:
            eoq = np.nan

        recommendations.append({
            "product_id": pid,
            "product_name": p_name,
            "forecast_origin": str(origin_dt.date()),
            "forecast_horizon_days": protection_period_days,
            "expected_lead_time_demand": round(ltd, 2),
            "expected_protection_period_demand": round(ppd, 2),
            "safety_stock": round(ss_lead, 2),
            "reorder_point": round(rop, 2),
            "target_stock_level": round(target_stock, 2),
            "simulated_on_hand": round(on_hand, 2),
            "simulated_on_order": round(on_order, 2),
            "simulated_backorders": round(backorders, 2),
            "inventory_position": round(ip, 2),
            "reorder_triggered": reorder_triggered,
            "recommended_order_quantity": q,
            "optional_eoq_benchmark": eoq,
            "service_level_target": service_level_target,
            "assumed_lead_time_days": lead_time_days,
            "review_period_days": review_period_days,
            "forecast_error_std": round(sigma_e, 4),
            "recommendation_status": status,
            "warning_or_limitation": simulated_note,
        })

    return pd.DataFrame(recommendations)

