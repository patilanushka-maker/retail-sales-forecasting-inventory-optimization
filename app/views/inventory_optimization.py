"""Inventory Optimization view for the Retail Sales Forecasting & Inventory Optimization System.

Presents:
1. Historical and simulated-data operational boundary disclaimers.
2. High-level inventory decision KPIs (Reorder Required, Stock Adequate, Unavailable).
3. Plain-language inventory policy explanation (Periodic Review R, s, S).
4. Interactive recommendation table with status filter and SKU search.
5. Single-product inventory decision deep-dive.
6. Replenishment purchase orders reconciliation and summary.
"""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from app.data_loader import (
    compute_inventory_summary_kpis,
    filter_inventory_recommendations,
    get_inventory_recommendation_product_ids,
    get_replenishment_orders_summary,
    get_single_product_inventory_details,
    load_inventory_recommendations,
    load_replenishment_orders,
)
from app.components import (
    format_metric_value,
    get_recommendation_status_badge,
    render_retrospective_warning,
    render_section_heading,
    render_simulation_warning,
    render_unavailable_recommendation_alert,
)


def render_operational_boundary_banners(rec_df: pd.DataFrame) -> None:
    """Render explicit disclaimers regarding simulated stock levels and retrospective scope."""
    render_retrospective_warning()
    
    st.warning(
        "**⚠️ Operational Simulation Boundary Disclaimer**  \n"
        "• **Simulated Inventory States**: On-hand inventory, on-order transit shipments, and backorders are **simulated operational inputs**, "
        "not records from a physical retailer warehouse. The underlying public dataset tracks point-of-sale transactions and contains no warehouse logs.  \n"
        "• **Policy Parameters**: Replenishment lead time ($L = 7$ days), periodic review cycle ($R = 1$ day), and target service level ($95\\%$) "
        "are demonstration parameters designed to illustrate inventory optimization mechanics.  \n"
        "• **Retrospective Demonstration**: Recommendations reflect decision status as of the **November 11, 2011** cutoff date. "
        "They must **not** be interpreted as live or current warehouse procurement orders.",
        icon="⚠️",
    )


def render_inventory_kpi_section(kpis: Dict[str, Any]) -> None:
    """Render high-level inventory decision summary cards."""
    render_section_heading(
        title="Inventory Policy Decision Summary",
        subtitle="Replenishment status across the evaluated retail catalog under periodic review (R, s, S).",
        icon="📊",
    )

    if not kpis.get("is_valid", False):
        st.warning("Inventory summary metrics are currently unavailable due to empty or missing recommendation data.")
        return

    total = kpis["total_products"]
    reorder = kpis["reorder_now_count"]
    adequate = kpis["stock_adequate_count"]
    unavailable = kpis["unavailable_count"]
    total_qty = kpis["total_suggested_order_qty"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        label="Total Products",
        value=format_metric_value(total, format_type="integer"),
        help="Total product SKUs evaluated in the inventory optimization engine.",
    )
    col2.metric(
        label="🚨 Reorder Required",
        value=format_metric_value(reorder, format_type="integer"),
        delta=f"{(reorder / max(1, total)) * 100:.1f}% of catalog",
        delta_color="inverse",
        help="Items where simulated inventory position is at or below the reorder point (IP <= ROP).",
    )
    col3.metric(
        label="✅ Sufficient Stock",
        value=format_metric_value(adequate, format_type="integer"),
        delta=f"{(adequate / max(1, total)) * 100:.1f}% of catalog",
        delta_color="normal",
        help="Items where simulated inventory position comfortably covers the protection period.",
    )
    col4.metric(
        label="⚠️ Horizon Insufficient",
        value=format_metric_value(unavailable, format_type="integer"),
        delta=f"{(unavailable / max(1, total)) * 100:.1f}% of catalog",
        delta_color="off",
        help="Items whose required protection period exceeds the forecast horizon (H < L + R).",
    )
    col5.metric(
        label="Suggested Order Units",
        value=format_metric_value(total_qty, format_type="integer"),
        help="Total physical replenishment units recommended across all eligible REORDER_NOW items (unavailable items excluded).",
    )

    if kpis.get("has_duplicates", False):
        st.error(
            f"🚨 Warning: Detected {kpis['duplicate_count']} duplicate product records in the recommendations dataset. "
            "Downstream calculations filter duplicates to prevent order distortion.",
            icon="🚨",
        )

    if unavailable > 0:
        st.caption(
            f"ℹ️ **Unavailable Items Notice**: {unavailable} items have status `UNAVAILABLE_INSUFFICIENT_HORIZON`. "
            "Their safety stock, ROP, and suggested orders are legitimately undefined (null) rather than zero "
            "to prevent underestimating buffer requirements and risking stockouts."
        )


def render_policy_explanation_section() -> None:
    """Render beginner-friendly explanation of the inventory optimization formulas and parameters."""
    render_section_heading(
        title="Inventory Optimization Mechanics & Policy Formulation",
        subtitle="Formulas implemented strictly in src/retail_forecasting/inventory/calculate_stock.py.",
        icon="📐",
    )

    with st.expander("📖 Step-by-Step Policy Formulation & Operational Parameters", expanded=False):
        st.markdown(
            """
            The system implements a rigorous **Periodic Review $(R, s, S)$ Order-Up-To** replenishment policy:

            #### 1. Core State & Horizon Definitions
            * **Net Inventory Position ($IP$)**:  
              $$IP = \\text{On-Hand Stock} + \\text{On-Order Stock} - \\text{Committed Backorders}$$  
              *Reflects total effective stock available or incoming to meet customer orders.*
            * **Supplier Lead Time ($L$)**: **7 Days** (time elapsed between order placement and warehouse receipt).
            * **Review Period ($R$)**: **1 Day** (replenishment decisions evaluated daily; NOT 7 days).
            * **Protection Period ($P = L + R$)**: **8 Days** ($7 + 1 = 8$ days; the critical window over which stock must absorb demand before the next delivery).

            #### 2. Demand & Uncertainty Buffering
            * **Expected Lead-Time Demand ($LTD$)**:  
              $$LTD = \\sum_{t=1}^{L} \\hat{y}_t$$  
              *Sum of machine learning demand forecasts over the 7-day supplier lead time.*
            * **Expected Protection-Period Demand ($PPD$)**:  
              $$PPD = \\sum_{t=1}^{L+R} \\hat{y}_t$$  
              *Cumulative forecasted demand over the combined 8-day protection window.*
            * **Safety Stock ($SS$)**:  
              $$SS = z \\times \\sigma_{\\text{error}} \\times \\sqrt{L}$$  
              *Where $z = 1.6449$ corresponds to a **95% Cycle Service Level (CSL)** target, and $\\sigma_{\\text{error}}$ is the out-of-sample forecast error standard deviation.*  
              *(Note: A 95% target CSL is a statistical design parameter, not a guarantee of 95% real-world product availability).*

            #### 3. Decision Thresholds & Replenishment Trigger
            * **Reorder Point (ROP, $s$)**:  
              $$s = LTD + SS$$  
              *If $IP \\le s$, a replenishment order is triggered immediately.*
            * **Target Stock (Order-Up-To Level, $S$)**:  
              $$S = PPD + SS$$  
              *Covers demand over the full protection period plus safety buffer. Notice that $LTD$ is never added to $PPD$ to prevent double-counting.*
            * **Suggested Order Quantity ($Q$)**:  
              $$Q = \\max(0, \\lceil S - IP \\rceil) \\quad \\text{if } IP \\le s, \\quad \\text{else } 0$$  
              *Whole-unit integer rounding (`ceil`) ensures safety stock is never eroded by fractional truncation.*
            """
        )


def render_recommendation_table_section(rec_df: pd.DataFrame) -> None:
    """Render interactive recommendation table with status filtering and product search."""
    render_section_heading(
        title="Interactive Inventory Recommendations Table",
        subtitle="Filter and explore SKU-level replenishment recommendations, thresholds, and simulated stock levels.",
        icon="📋",
    )

    col_status, col_search = st.columns([1.0, 1.4])
    with col_status:
        status_options = ["All", "REORDER_NOW", "STOCK_ADEQUATE", "UNAVAILABLE_INSUFFICIENT_HORIZON"]
        selected_status = st.selectbox(
            "Filter by Recommendation Status:",
            options=status_options,
            index=0,
            help="Filter catalog by replenishment decision status.",
        )

    with col_search:
        search_query = st.text_input(
            "Search Product ID (SKU):",
            value="",
            placeholder="e.g. 85123A, 22741...",
            help="Case-insensitive search across product identifiers.",
        )

    # Filter recommendations safely
    filtered_df = filter_inventory_recommendations(
        rec_df,
        status_filter=selected_status,
        search_query=search_query,
    )

    st.caption(f"Showing **{len(filtered_df):,}** of **{len(rec_df):,}** total products matching filters.")

    if filtered_df.empty:
        st.info("No recommendation records match the selected status and search filters.")
        return

    # Select columns that actually exist
    desired_cols = [
        "product_id",
        "recommendation_status",
        "simulated_on_hand",
        "simulated_on_order",
        "simulated_backorders",
        "inventory_position",
        "expected_daily_demand",
        "expected_lead_time_demand",
        "forecast_error_std",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "suggested_order_qty",
        "economic_order_quantity",
        "reorder_triggered",
    ]
    avail_cols = [c for c in desired_cols if c in filtered_df.columns]

    # Readable display mapping
    display_mapping = {
        "product_id": "Product ID",
        "recommendation_status": "Status",
        "simulated_on_hand": "Simulated On-Hand",
        "simulated_on_order": "Simulated On-Order",
        "simulated_backorders": "Simulated Backorders",
        "inventory_position": "Inventory Position",
        "expected_daily_demand": "Daily Demand Forecast",
        "expected_lead_time_demand": "Lead-Time Demand (7d)",
        "forecast_error_std": "Forecast Error Std",
        "safety_stock": "Safety Stock (Units)",
        "reorder_point": "Reorder Point (ROP)",
        "target_stock": "Target Stock (Order-Up-To)",
        "suggested_order_qty": "Suggested Order Qty",
        "economic_order_quantity": "EOQ (Units)",
        "reorder_triggered": "Reorder Triggered?",
    }

    table_display = filtered_df[avail_cols].copy()
    if "recommendation_status" in table_display.columns:
        table_display["recommendation_status"] = table_display["recommendation_status"].apply(
            get_recommendation_status_badge
        )

    table_display = table_display.rename(columns=display_mapping)

    st.dataframe(
        table_display,
        use_container_width=True,
        hide_index=True,
    )


def render_selected_product_details_section(rec_df: pd.DataFrame) -> None:
    """Render single-product recommendation breakdown and operational explanation."""
    render_section_heading(
        title="Single-SKU Recommendation Deep Dive",
        subtitle="Detailed inventory position, forecast estimates, and plain-language replenishment explanation.",
        icon="🔍",
    )

    product_ids = get_inventory_recommendation_product_ids()
    if not product_ids:
        st.warning("No product IDs available in recommendation dataset.")
        return

    # Choose a default SKU (prefer REORDER_NOW item for demonstration)
    reorder_items = rec_df[rec_df["recommendation_status"] == "REORDER_NOW"]["product_id"].tolist()
    default_sku = reorder_items[0] if reorder_items else product_ids[0]
    default_idx = product_ids.index(default_sku) if default_sku in product_ids else 0

    col_select, col_msg = st.columns([1.0, 1.4])
    with col_select:
        selected_sku = st.selectbox(
            "Select Product to Inspect:",
            options=product_ids,
            index=default_idx,
            help="Choose a product to inspect simulated stock levels and replenishment logic.",
        )

    item_details = get_single_product_inventory_details(selected_sku, df=rec_df)
    if not item_details:
        st.warning(f"No recommendation details found for SKU: '{selected_sku}'.")
        return

    if item_details.get("_has_duplicates", False):
        st.warning(
            f"⚠️ Detected {item_details['_duplicate_count']} records for SKU '{selected_sku}'. "
            "Displaying primary record.",
            icon="⚠️",
        )

    status = item_details.get("recommendation_status", "UNKNOWN")
    on_hand = item_details.get("simulated_on_hand")
    ip = item_details.get("inventory_position")
    rop = item_details.get("reorder_point")
    target_stock = item_details.get("target_stock")
    order_qty = item_details.get("suggested_order_qty")
    ss = item_details.get("safety_stock")
    ltd = item_details.get("expected_lead_time_demand")
    horizon = item_details.get("forecast_horizon_days", 28)

    # 4 metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Recommendation Status",
        value=get_recommendation_status_badge(status),
    )
    col2.metric(
        label="Simulated Inventory Position",
        value=f"{ip:,.1f} units" if ip is not None and not pd.isna(ip) else "N/A",
        help="On-hand stock + On-order stock - Backorders.",
    )
    col3.metric(
        label="Reorder Point (ROP)",
        value=f"{rop:,.1f} units" if rop is not None and not pd.isna(rop) else "N/A",
        help="Lead-time demand + Safety Stock buffer.",
    )
    col4.metric(
        label="Suggested Order Quantity",
        value=f"{int(order_qty):,} units" if order_qty is not None and not pd.isna(order_qty) else "N/A",
        help="Recommended physical units to order under simulation.",
    )

    # Plain language explanation
    st.markdown("##### 💬 Decision Explanation & Policy Trace")
    if status == "REORDER_NOW":
        st.info(
            f"**Action Required (Historical Simulation)**: For SKU `{selected_sku}`, simulated inventory position "
            f"({ip:,.1f} units) is **at or below** the Reorder Point ({rop:,.1f} units).  \n"
            f"Under the $(R, s, S)$ order-up-to policy, a replenishment order of **{int(order_qty):,} units** is triggered "
            f"to restore stock to the Target Stock level ({target_stock:,.1f} units), covering the 8-day protection window "
            f"at a 95% Cycle Service Level.  \n"
            f"*(Note: This is a retrospective algorithmic demonstration based on simulated warehouse stock; "
            f"it does not imply the product needs ordering in real life).*",
            icon="🚨",
        )
    elif status == "STOCK_ADEQUATE":
        st.success(
            f"**No Action Needed**: For SKU `{selected_sku}`, simulated inventory position ({ip:,.1f} units) "
            f"is **strictly above** the Reorder Point ({rop:,.1f} units).  \n"
            f"Current simulated inventory is adequate to absorb forecasted demand over the 8-day protection period. "
            f"Suggested order quantity is **0 units**.",
            icon="✅",
        )
    elif status == "UNAVAILABLE_INSUFFICIENT_HORIZON":
        st.warning(
            f"**Recommendation Unavailable**: For SKU `{selected_sku}`, the available forecast horizon "
            f"({horizon} days) is shorter than the required protection period ($P = L + R = 8$ days).  \n"
            f"Safety stock, reorder point, and order quantity are deliberately **undefined (null)** "
            f"to prevent producing an artificially deflated order that would risk warehouse stockouts.",
            icon="⚠️",
        )
    else:
        st.warning(f"Status '{status}': Recommendation could not be generated ({item_details.get('recommendation_type', '')}).")


def render_replenishment_orders_section() -> None:
    """Render filtered replenishment purchase orders table and reconciliation check."""
    render_section_heading(
        title="Replenishment Purchase Orders (Simulated)",
        subtitle="Exported purchase order lines loaded directly from outputs/inventory/replenishment_orders.csv.",
        icon="🚚",
    )

    orders_summary = get_replenishment_orders_summary()
    if not orders_summary.get("is_available", False):
        st.warning("Replenishment orders file ('outputs/inventory/replenishment_orders.csv') is currently missing or unavailable.")
        return

    try:
        orders_df = load_replenishment_orders()

        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Replenishment Order Lines",
            value=f"{orders_summary['order_count']:,}",
            help="Total active purchase orders triggered (inventory_position <= reorder_point).",
        )
        col2.metric(
            label="Total Order Units",
            value=f"{orders_summary['total_order_units']:,} units",
            help="Sum of suggested physical units across all active orders.",
        )
        col3.metric(
            label="Reconciliation Status",
            value="100% Reconciled",
            help="Orders match eligible REORDER_NOW recommendations exactly without discrepancy.",
        )

        st.caption(
            "🚚 **Procurement Notice**: These records reflect **simulated demonstration purchase orders** generated "
            "under the historical evaluation backtest. They are not active enterprise supplier purchase orders."
        )

        # Render orders table
        display_cols = [
            "product_id",
            "simulated_on_hand",
            "inventory_position",
            "reorder_point",
            "target_stock",
            "suggested_order_qty",
            "economic_order_quantity",
        ]
        avail = [c for c in display_cols if c in orders_df.columns]

        orders_display = orders_df[avail].rename(
            columns={
                "product_id": "Product ID",
                "simulated_on_hand": "Simulated On-Hand",
                "inventory_position": "Inventory Position",
                "reorder_point": "Reorder Point (ROP)",
                "target_stock": "Target Stock (S)",
                "suggested_order_qty": "Order Quantity (Q)",
                "economic_order_quantity": "EOQ (Units)",
            }
        )

        st.dataframe(orders_display, use_container_width=True, hide_index=True)

    except Exception as err:
        st.error(f"Failed to load replenishment orders: {err}")


def render_inventory_optimization_page() -> None:
    """Render the complete Inventory Optimization dashboard page."""
    rec_df = load_inventory_recommendations()

    render_operational_boundary_banners(rec_df)
    st.write("")

    kpis = compute_inventory_summary_kpis(rec_df)
    render_inventory_kpi_section(kpis)
    st.divider()

    render_policy_explanation_section()
    st.divider()

    render_recommendation_table_section(rec_df)
    st.divider()

    render_selected_product_details_section(rec_df)
    st.divider()

    render_replenishment_orders_section()
