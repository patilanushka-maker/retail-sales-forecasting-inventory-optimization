"""SKU Deep Dive & Explorer view for the Retail Sales Forecasting & Inventory Optimization System.

Unifies for an individually selected product:
1. Product selection & dataset coverage status across demand, forecast, and inventory artifacts.
2. Historical demand analysis preserving zero-demand calendar continuity.
3. Out-of-sample forecast backtest evaluation (Random Forest vs. Seasonal Naive baseline).
4. Retrospective periodic review inventory recommendations and simulated operational inputs.
5. Evidence-based cross-sectional interpretation and methodological limitations.
"""

from typing import Any, Dict, Optional, Tuple
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.data_loader import (
    check_product_data_coverage,
    compute_product_period_statistics,
    compute_single_product_forecast_metrics,
    generate_sku_synthesis,
    get_available_product_ids,
    get_single_product_inventory_details,
    load_product_daily_demand,
    load_product_forecast_comparison,
)
from app.components import (
    format_metric_value,
    get_recommendation_status_badge,
    render_section_heading,
)
from src.retail_forecasting.visualization.plots import (
    plot_forecast_vs_actual,
    plot_product_daily_demand,
)


def render_sku_selector_and_coverage(
    available_skus: list,
) -> Tuple[str, Dict[str, Any]]:
    """Render the SKU selection widget and product population coverage cards.

    Parameters
    ----------
    available_skus : list
        List of valid catalog product identifiers.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Selected product SKU and its data coverage status dictionary.
    """
    render_section_heading(
        title="Product Selection & Population Coverage",
        subtitle="Catalog populations: 3,804 demand SKUs | 2,813 forecast evaluation SKUs | 2,813 inventory recommendation SKUs.",
        icon="🔍",
    )

    if not available_skus:
        st.warning("No product identifiers available in the demand dataset.")
        return "", check_product_data_coverage(None)

    # Sensible default selection: '85123A' (top-selling SKU) or first available
    default_sku = "85123A" if "85123A" in available_skus else available_skus[0]
    default_idx = available_skus.index(default_sku)

    col_select, col_info = st.columns([1.2, 1.8])
    with col_select:
        selected_sku = st.selectbox(
            "Select Catalog Product SKU:",
            options=available_skus,
            index=default_idx,
            help="Select any valid product identifier to inspect its empirical trajectory and backtest results.",
            key="sku_explorer_select",
        )

    coverage = check_product_data_coverage(selected_sku)

    with col_info:
        st.caption(
            f"**Selected Product**: `{selected_sku}`  \n"
            "Products can be present in historical demand without having usable forecast evaluation or inventory recommendation records "
            "(e.g. products with sparse history prior to the November 11, 2011 cutoff date)."
        )

    # Render 3 coverage status cards
    col_c1, col_c2, col_c3 = st.columns(3)

    # 1. Historical Demand Coverage
    if coverage.get("has_demand"):
        col_c1.success(
            f"**Historical Demand**: Available  \n"
            f"• {coverage['demand_count']:,} calendar days  \n"
            f"• {coverage['demand_start_date']} to {coverage['demand_end_date']}",
            icon="📅",
        )
    else:
        col_c1.error("**Historical Demand**: Unavailable", icon="❌")

    # 2. Forecast Evaluation Coverage
    if coverage.get("has_forecast"):
        col_c2.success(
            f"**Forecast Evaluation**: Available  \n"
            f"• {coverage['forecast_count']} backtest days  \n"
            f"• {coverage['forecast_start_date']} to {coverage['forecast_end_date']}",
            icon="🎯",
        )
    else:
        col_c2.warning(
            "**Forecast Evaluation**: Unavailable  \n"
            "• Insufficient history at Nov 11 cutoff  \n"
            "• Excluded from out-of-sample backtest",
            icon="⚠️",
        )

    # 3. Inventory Recommendation Coverage
    if coverage.get("has_inventory"):
        if coverage.get("inventory_has_duplicates"):
            col_c3.error(
                f"**Inventory Policy**: Data Issue  \n"
                f"• {coverage['inventory_duplicate_count']} duplicate records detected  \n"
                "• Selection withheld for integrity",
                icon="🚨",
            )
        else:
            status_text = coverage.get("inventory_status", "UNKNOWN")
            badge = get_recommendation_status_badge(status_text)
            col_c3.success(
                f"**Inventory Policy**: Available  \n"
                f"• Status: {badge}  \n"
                "• Policy: Periodic Review (R, s, S)",
                icon="📦",
            )
    else:
        col_c3.warning(
            "**Inventory Policy**: Unavailable  \n"
            "• No recommendation record  \n"
            "• Forecast or horizon ineligible",
            icon="⚠️",
        )

    return selected_sku, coverage


def render_historical_demand_section(
    selected_sku: str,
) -> Optional[Dict[str, Any]]:
    """Render historical demand trajectory and statistical metrics for selected SKU.

    Parameters
    ----------
    selected_sku : str
        Target product SKU.

    Returns
    -------
    Optional[Dict[str, Any]]
        Demand statistics dictionary, or None if no demand data exists.
    """
    render_section_heading(
        title="1. Historical Demand Trajectory",
        subtitle="Empirical daily recorded demand units preserving zero-demand calendar continuity from first purchase transaction.",
        icon="📊",
    )

    product_df = load_product_daily_demand(selected_sku)
    if product_df.empty:
        st.info(f"No historical demand records found for Product `{selected_sku}`.")
        return None

    min_date = pd.to_datetime(product_df["date"].min()).date()
    max_date = pd.to_datetime(product_df["date"].max()).date()

    col_date_filter, col_date_info = st.columns([1.2, 1.8])
    with col_date_filter:
        date_selection = st.date_input(
            "Filter Historical Demand Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Select a custom window to compute empirical demand statistics strictly over that date range.",
            key=f"sku_date_filter_{selected_sku}",
        )

    # Safe parsing of date selection tuple
    if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
        start_date, end_date = date_selection
    elif isinstance(date_selection, (tuple, list)) and len(date_selection) == 1:
        start_date = end_date = date_selection[0]
    else:
        start_date, end_date = min_date, max_date

    # Compute statistics strictly from the filtered window
    stats = compute_product_period_statistics(
        df_product=product_df,
        start_date=start_date,
        end_date=end_date,
    )
    filtered_df = stats["filtered_df"]

    with col_date_info:
        st.caption(
            f"**Active Date Window**: {start_date} to {end_date}  \n"
            f"Calculations are strictly computed from the {stats['total_days']:,} calendar days within this window."
        )

    if filtered_df.empty:
        st.warning("No records match the selected date range.")
        return stats

    # Render 5 compact KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        label="Total Demand",
        value=format_metric_value(stats["total_demand"], format_type="integer"),
        help="Total recorded units demanded over the selected date range.",
    )
    col2.metric(
        label="Average Daily Demand",
        value=f"{stats['mean_demand']:.2f} units",
        help="Average daily recorded demand across all calendar days (including zero-demand days).",
    )
    col3.metric(
        label="Calendar Days",
        value=f"{stats['total_days']:,} days",
        help="Number of consecutive calendar days represented in the selected window.",
    )
    col4.metric(
        label="Active Sales Days",
        value=f"{stats['positive_demand_days']:,} days",
        delta=f"{(stats['positive_demand_days'] / max(1, stats['total_days'])) * 100:.1f}%",
        delta_color="normal",
        help="Number and percentage of calendar days on which at least 1 unit was recorded sold.",
    )
    col5.metric(
        label="Zero-Demand Days",
        value=f"{stats['zero_demand_days']:,} days",
        delta=f"{(stats['zero_demand_days'] / max(1, stats['total_days'])) * 100:.1f}%",
        delta_color="off",
        help="Number and percentage of calendar days with 0 recorded demand (preserved from daily calendar expansion).",
    )

    # Plot time series
    fig = plot_product_daily_demand(filtered_df, product_id=selected_sku)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "⚠️ **Methodological Notice**: The available dataset contains **date, product ID, and demand units**. "
        "Revenue, prices, margins, and customer counts are not present in this artifact and are not invented. "
        "Zero-demand days represent daily calendar continuity established from the product's first recorded transaction; "
        "they do not prove that stock was physically available in the warehouse on that day (unobserved stockouts)."
    )

    return stats


def render_forecast_evaluation_section(
    selected_sku: str,
    coverage: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Render out-of-sample forecast evaluation comparison for selected SKU.

    Parameters
    ----------
    selected_sku : str
        Target product SKU.
    coverage : Dict[str, Any]
        Coverage dictionary for selected SKU.

    Returns
    -------
    Optional[Dict[str, Any]]
        Forecast metrics dictionary, or None if forecast data is unavailable.
    """
    render_section_heading(
        title="2. Out-of-Sample Forecast Evaluation (Nov 12 – Dec 9, 2011)",
        subtitle="Historical rolling one-day-ahead predictions vs. actual demand across the 28-day backtest window.",
        icon="🎯",
    )

    if not coverage.get("has_forecast"):
        st.info(
            f"ℹ️ **Forecast Evaluation Unavailable for SKU `{selected_sku}`**  \n"
            "This product was excluded from the out-of-sample evaluation partition because it had insufficient transaction "
            "history prior to the November 11, 2011 cutoff date to construct the required feature lags (e.g. 14-day rolling statistics). "
            "No evaluation records were fabricated.",
            icon="ℹ️",
        )
        return None

    fc_df = load_product_forecast_comparison(selected_sku)
    if fc_df.empty:
        st.info(f"No saved evaluation records found for SKU `{selected_sku}`.")
        return None

    # Compute single-product forecast metrics (MAE, RMSE, WAPE)
    metrics = compute_single_product_forecast_metrics(fc_df)

    # Identify missing predictions if any
    if metrics["missing_count"] > 0:
        st.warning(
            f"⚠️ **Incomplete Predictions Detected**: {metrics['missing_count']} evaluation observations contained missing "
            "prediction or actual values. These rows were excluded from metric calculation without silent substitution.",
            icon="⚠️",
        )

    rf_m = metrics["random_forest"]
    sn_m = metrics["seasonal_naive"]

    # Display KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Evaluation Observations",
        value=f"{metrics['sample_size']} days",
        help="Number of valid out-of-sample evaluation observations (Nov 12 – Dec 9, 2011).",
    )
    col2.metric(
        label="Total Actual Demand",
        value=format_metric_value(metrics["total_actual"], format_type="integer"),
        help="Sum of actual observed demand units across the evaluation window.",
    )

    # RF Metrics
    rf_wape_display = f"{rf_m['wape']:.1f}%" if rf_m.get("wape_defined") else "Unavailable (Zero Actual)"
    sn_wape_display = f"{sn_m['wape']:.1f}%" if sn_m.get("wape_defined") else "Unavailable (Zero Actual)"

    col3.metric(
        label="Random Forest MAE",
        value=f"{rf_m['mae']:.2f} units",
        delta=f"WAPE: {rf_wape_display}",
        delta_color="normal",
        help="Mean Absolute Error of the chronological Random Forest forecaster on this SKU.",
    )
    col4.metric(
        label="Seasonal Naive MAE",
        value=f"{sn_m['mae']:.2f} units",
        delta=f"WAPE: {sn_wape_display}",
        delta_color="off",
        help="Mean Absolute Error of the 7-day Seasonal Naive baseline on this SKU.",
    )

    # Secondary metrics row: RMSE
    col_rmse1, col_rmse2, col_note1, col_note2 = st.columns(4)
    col_rmse1.metric(
        label="Random Forest RMSE",
        value=f"{rf_m['rmse']:.2f} units",
        help="Root Mean Squared Error for Random Forest on this SKU.",
    )
    col_rmse2.metric(
        label="Seasonal Naive RMSE",
        value=f"{sn_m['rmse']:.2f} units",
        help="Root Mean Squared Error for Seasonal Naive baseline on this SKU.",
    )
    with col_note1:
        st.caption(
            f"**Evaluation Window**: Nov 12, 2011 – Dec 09, 2011  \n"
            f"**Observations**: {metrics['sample_size']} days  \n"
            f"**Missing Rows**: {metrics['missing_count']} excluded"
        )
    with col_note2:
        st.caption(
            "**WAPE Formula**:  \n"
            r"$\frac{\sum |y - \hat{y}|}{\sum |y|} \times 100$  \n"
            r"Reported unavailable if $\sum |y| = 0$."
        )

    # Plot actual vs predicted
    fig = plot_forecast_vs_actual(
        dates=fc_df["date"],
        y_actual=fc_df["actual_demand"],
        y_pred=fc_df["random_forest_pred"],
        baseline_pred=fc_df["seasonal_naive_pred"],
        title=f"Out-of-Sample Forecast vs. Actual Demand: SKU {selected_sku}",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "ℹ️ **Methodological Context**: Predictions shown are **retrospective rolling one-day-ahead evaluation backtests** "
        "generated strictly from historical training data available up to each step (Nov 12 – Dec 9, 2011). "
        "These are not current live forecasts for future replenishment periods."
    )

    return metrics


def render_inventory_recommendation_section(
    selected_sku: str,
    coverage: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Render retrospective inventory policy recommendation details for selected SKU.

    Parameters
    ----------
    selected_sku : str
        Target product SKU.
    coverage : Dict[str, Any]
        Coverage dictionary for selected SKU.

    Returns
    -------
    Optional[Dict[str, Any]]
        Recommendation details dictionary, or None if unavailable.
    """
    render_section_heading(
        title="3. Retrospective Inventory Policy Recommendation",
        subtitle="Simulated periodic review (R, s, S) replenishment parameters, inventory position, and order triggers.",
        icon="📦",
    )

    inv_details = get_single_product_inventory_details(selected_sku)

    if inv_details is None or not coverage.get("has_inventory"):
        st.info(
            f"ℹ️ **Inventory Recommendation Unavailable for SKU `{selected_sku}`**  \n"
            "No recommendation record exists for this product in the saved inventory recommendations artifact. "
            "Products without eligible forecast predictions are excluded from replenishment calculations.",
            icon="ℹ️",
        )
        return None

    # Handle duplicate records
    if inv_details.get("_has_duplicates"):
        st.error(
            f"🚨 **Data Integrity Safeguard**: Multiple recommendation records ({inv_details.get('_duplicate_count')} rows) "
            f"were detected for SKU `{selected_sku}` in the inventory recommendations artifact. "
            "In accordance with rigorous data integrity rules, recommendations are not arbitrarily selected from duplicate rows. "
            "Please resolve upstream data generation before using recommendations for this SKU.",
            icon="🚨",
        )
        return inv_details

    status = inv_details.get("recommendation_status", "UNKNOWN")
    badge = get_recommendation_status_badge(status)

    st.markdown(f"#### Recommendation Status: {badge}")

    # If unavailable due to insufficient horizon
    if str(status).startswith("UNAVAILABLE"):
        st.warning(
            f"**Recommendation Status: `{status}`**  \n"
            "Periodic review inventory optimization requires a forecast horizon covering at least Lead Time + Review Period "
            r"($H \ge L + R$). For items where the replenishment cycle exceeds the available forecast horizon, "
            "safety stock, reorder point, and suggested order quantity are deliberately undefined (null) "
            "to prevent underestimating buffer requirements and risking stockouts.",
            icon="⚠️",
        )
        return inv_details

    # Render simulated operational inputs and policy parameters
    st.markdown("##### Simulated Operational State")
    col_oh, col_oo, col_bo, col_ip = st.columns(4)
    col_oh.metric(
        label="Simulated On-Hand Stock",
        value=format_metric_value(inv_details.get("simulated_on_hand"), format_type="integer"),
        help="Simulated physically available units in warehouse.",
    )
    col_oo.metric(
        label="Simulated On-Order",
        value=format_metric_value(inv_details.get("simulated_on_order"), format_type="integer"),
        help="Simulated units currently in transit from supplier.",
    )
    col_bo.metric(
        label="Simulated Backorders",
        value=format_metric_value(inv_details.get("simulated_backorders"), format_type="integer"),
        help="Simulated unfulfilled customer demand awaiting replenishment.",
    )
    col_ip.metric(
        label="Inventory Position (IP)",
        value=format_metric_value(inv_details.get("inventory_position"), format_type="float", decimals=1),
        help="Effective stock position: IP = On-Hand + On-Order - Backorders.",
    )

    st.markdown("##### Policy Thresholds & Suggested Replenishment Order")
    col_ltd, col_std, col_ss, col_rop, col_target, col_qty = st.columns(6)
    col_ltd.metric(
        label="Lead-Time Demand",
        value=format_metric_value(inv_details.get("expected_lead_time_demand"), format_type="float", decimals=1),
        help="Expected demand over lead time (L = 7 days).",
    )
    col_std.metric(
        label="Forecast Error Std (σ)",
        value=format_metric_value(inv_details.get("forecast_error_std"), format_type="float", decimals=2),
        help="Standard deviation of one-step forecast errors over evaluation period.",
    )
    col_ss.metric(
        label="Safety Stock (SS)",
        value=format_metric_value(inv_details.get("safety_stock"), format_type="float", decimals=1),
        help="Buffer stock to absorb demand uncertainty: SS = z * sigma_L.",
    )
    col_rop.metric(
        label="Reorder Point (ROP)",
        value=format_metric_value(inv_details.get("reorder_point"), format_type="float", decimals=1),
        help="Threshold level: ROP = Expected LTD + SS. Replenishment triggered when IP <= ROP.",
    )
    col_target.metric(
        label="Target Stock (S)",
        value=format_metric_value(inv_details.get("target_stock"), format_type="float", decimals=1),
        help="Order-up-to target level under periodic review: S = Expected Protection Period Demand + SS.",
    )
    col_qty.metric(
        label="Suggested Order Qty",
        value=format_metric_value(inv_details.get("suggested_order_qty"), format_type="integer"),
        delta="Order Triggered" if inv_details.get("reorder_triggered") else "No Order Needed",
        delta_color="normal" if inv_details.get("reorder_triggered") else "off",
        help="Suggested replenishment quantity: Q = max(0, ceil(Target Stock - IP)).",
    )

    # Optional EOQ display if present
    eoq_val = inv_details.get("economic_order_quantity")
    if eoq_val is not None and not pd.isna(eoq_val):
        st.caption(
            f"**Economic Order Quantity (EOQ)**: {format_metric_value(eoq_val, format_type='integer')} units "
            "(based on simulated holding cost £1.50/unit/year and ordering cost £25.00/order)."
        )

    # Explicit policy assumptions and simulated inputs disclaimer
    st.caption(
        "ℹ️ **Operational Policy Assumptions**: Policy parameters evaluate a periodic review $(R, s, S)$ framework with "
        "**Service Level = 95%** ($z = 1.6449$), **Lead Time $L = 7$ days**, **Review Period $R = 1$ day**, and "
        "**Protection Period $P = L + R = 8$ days**. "
        "On-hand inventory levels and purchase orders reflect **simulated demonstration scenarios**; "
        "they do not constitute live enterprise purchasing instructions."
    )

    return inv_details


def render_synthesis_section(
    coverage: Dict[str, Any],
    demand_stats: Optional[Dict[str, Any]],
    forecast_metrics: Optional[Dict[str, Any]],
    inventory_details: Optional[Dict[str, Any]],
) -> None:
    """Render objective, evidence-based synthesis connecting demand, forecast, and inventory.

    Parameters
    ----------
    coverage : Dict[str, Any]
        Data coverage status.
    demand_stats : Optional[Dict[str, Any]]
        Historical demand statistics.
    forecast_metrics : Optional[Dict[str, Any]]
        Forecast evaluation metrics.
    inventory_details : Optional[Dict[str, Any]]
        Inventory recommendation details.
    """
    render_section_heading(
        title="4. Cross-Sectional Synthesis & Methodological Limitations",
        subtitle="Objective synthesis connecting historical demand velocity, backtest accuracy, and inventory policy.",
        icon="💡",
    )

    synthesis = generate_sku_synthesis(
        coverage=coverage,
        demand_stats=demand_stats,
        forecast_metrics=forecast_metrics,
        inventory_details=inventory_details,
    )

    with st.container():
        st.markdown(synthesis["overall_synthesis"])
        st.write("")

        col_d, col_f, col_i = st.columns(3)
        with col_d:
            st.markdown("##### 📅 Demand Characteristics")
            st.write(synthesis["demand_commentary"])

        with col_f:
            st.markdown("##### 🎯 Forecast Comparison")
            st.write(synthesis["forecast_commentary"])

        with col_i:
            st.markdown("##### 📦 Inventory Recommendation")
            st.write(synthesis["inventory_commentary"])

    st.divider()
    st.markdown("##### ⚠️ Methodological Safeguards & Governance")
    st.caption(
        "1. **No Universal Superiority Claims**: Forecast performance on an individual product during a 28-day historical window "
        "does not establish that one model is universally superior across the catalog or in future operating periods.  \n"
        "2. **Simulated Inventory Inputs**: On-hand stock, lead times, and replenishment orders are simulated demonstration scenarios "
        "designed to evaluate inventory mathematics, not real physical warehouse stock.  \n"
        "3. **Zero-Demand vs Stockout Ambiguity**: The underlying data does not record store stockouts. Zero recorded demand represents "
        "days without purchase transactions, which may reflect either zero consumer demand or unobserved inventory stockouts."
    )


def render_sku_explorer_page() -> None:
    """Render the full SKU Deep Dive & Explorer dashboard page."""
    render_section_heading(
        title="SKU Deep Dive & Explorer",
        subtitle="Unified product-level analysis connecting empirical historical demand, rolling one-day-ahead forecast backtest results, and periodic review inventory recommendations.",
        icon="🔍",
    )

    try:
        available_skus = get_available_product_ids()
        if not available_skus:
            st.warning("Daily demand dataset contains no valid product identifiers.")
            return

        # 1. Product selector and population coverage
        selected_sku, coverage = render_sku_selector_and_coverage(available_skus)
        if not selected_sku:
            return

        st.divider()

        # 2. Historical demand analysis
        demand_stats = render_historical_demand_section(selected_sku)
        st.divider()

        # 3. Forecast evaluation
        forecast_metrics = render_forecast_evaluation_section(selected_sku, coverage)
        st.divider()

        # 4. Inventory recommendation
        inventory_details = render_inventory_recommendation_section(selected_sku, coverage)
        st.divider()

        # 5. Cross-sectional synthesis and limitations
        render_synthesis_section(
            coverage=coverage,
            demand_stats=demand_stats,
            forecast_metrics=forecast_metrics,
            inventory_details=inventory_details,
        )

    except Exception as err:
        st.error(f"Failed to render SKU Deep Dive & Explorer: {err}")
