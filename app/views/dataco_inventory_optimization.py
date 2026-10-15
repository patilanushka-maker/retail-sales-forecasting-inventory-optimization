"""DataCo Smart Supply Chain: Inventory Optimization & ROP view for Streamlit.

Presents:
1. Prominent simulation disclaimer: warehouse inventory, supplier lead time, service level, and costs
   are simulated demonstration parameters, not real enterprise purchase records.
2. Complete Core 9 Inventory Replenishment Recommendation Table.
3. Summary KPIs (evaluated products, reorder orders triggered, total replenishment units).
4. Individual SKU Deep Dive:
   - Key inventory metrics cards (LTD, PPD, Safety Stock, ROP, Target Stock, Net IP, Order Qty, EOQ).
   - Fixed-Origin 8-Day Recursive Forecast Horizon chart and daily table (2017-09-01 to 2017-09-08).
   - Full input parameter traceability.
5. Actionable replenishment purchase orders export.
6. Explicit handling of unavailable recommendations without silent zero substitution.
"""

from typing import Any, Dict
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.data_loader import (
    load_dataco_inventory_recommendations,
    load_dataco_fixed_origin_forecasts,
    load_dataco_replenishment_orders,
    load_dataco_catalog,
)
from app.components import (
    format_metric_value,
    get_recommendation_status_badge,
    render_dataco_disclaimer_banner,
    render_section_heading,
)
from src.retail_forecasting.visualization.plots import (
    plot_dataco_fixed_horizon_forecast,
)


def render_dataco_inventory_optimization_page() -> None:
    """Render DataCo Inventory Optimization view."""
    render_dataco_disclaimer_banner()

    render_section_heading(
        title="DataCo Inventory Optimization & Replenishment Planning",
        subtitle="Periodic review (R, s, S) decision-support policies sizing safety stock buffers across fixed-origin horizons.",
        icon="📦",
    )

    try:
        recs_df = load_dataco_inventory_recommendations()
        fixed_fc_df = load_dataco_fixed_origin_forecasts()
        catalog_df = load_dataco_catalog()
    except Exception as err:
        st.error(f"Error loading DataCo inventory artifacts: {err}")
        return

    if recs_df.empty:
        st.warning("⚠️ No DataCo inventory recommendations available. Please generate recommendations artifact first.")
        return

    # Summary KPIs
    total_skus = len(recs_df)
    reorders_triggered = int(recs_df["reorder_triggered"].sum()) if "reorder_triggered" in recs_df.columns else 0
    total_order_qty = int(recs_df["recommended_order_quantity"].dropna().sum()) if "recommended_order_quantity" in recs_df.columns else 0
    total_ltd = float(recs_df["expected_lead_time_demand"].dropna().sum()) if "expected_lead_time_demand" in recs_df.columns else 0.0
    avg_ss = float(recs_df["safety_stock"].dropna().mean()) if "safety_stock" in recs_df.columns else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Evaluated Products", f"{total_skus} SKUs")
    kpi2.metric("Orders Triggered", f"{reorders_triggered} / {total_skus}")
    kpi3.metric("Total Replenishment Units", f"{total_order_qty:,}")
    kpi4.metric("Total Lead-Time Demand", f"{int(total_ltd):,} units")
    kpi5.metric("Avg Safety Stock Buffer", f"{avg_ss:.1f} units")

    st.write("")

    # 1. Complete Recommendations Table
    st.markdown("### 📋 Core 9 Replenishment Policy Recommendation Table")

    display_recs = recs_df.copy()
    display_recs["Status"] = display_recs["recommendation_status"].apply(get_recommendation_status_badge)

    cols_order = [
        "product_id", "product_name", "forecast_origin", "expected_lead_time_demand",
        "expected_protection_period_demand", "forecast_error_std", "safety_stock",
        "reorder_point", "target_stock_level", "inventory_position",
        "recommended_order_quantity", "optional_eoq_benchmark", "Status"
    ]
    cols_exist = [c for c in cols_order if c in display_recs.columns]

    rename_map = {
        "product_id": "Product ID",
        "product_name": "Product Name",
        "forecast_origin": "Origin Date",
        "expected_lead_time_demand": "LTD (7d)",
        "expected_protection_period_demand": "PPD (8d)",
        "forecast_error_std": "Daily σ_e",
        "safety_stock": "Safety Stock",
        "reorder_point": "Reorder Point",
        "target_stock_level": "Target Stock (S)",
        "inventory_position": "Current Net IP",
        "recommended_order_quantity": "Order Qty (Q)",
        "optional_eoq_benchmark": "EOQ Benchmark",
    }
    table_view = display_recs[cols_exist].rename(columns=rename_map)
    st.dataframe(table_view, use_container_width=True, hide_index=True)

    st.write("")

    # 2. SKU Deep Dive & Forecast Horizon Trajectory
    render_section_heading(
        title="SKU Inventory & Multi-Step Forecast Deep Dive",
        subtitle="Inspect fixed-origin multi-step recursive forecast rollout and detailed policy parameters for a selected SKU.",
        icon="🔍",
    )

    available_pids = sorted(list(recs_df["product_id"].unique()))
    name_lookup = dict(zip(recs_df["product_id"], recs_df["product_name"]))

    selected_pid = st.selectbox(
        "Select Core 9 SKU for Deep Dive:",
        options=available_pids,
        format_func=lambda pid: f"SKU {pid} — {name_lookup.get(pid, 'Unknown')}",
        key="dataco_inv_opt_sku_select"
    )

    matching_recs = recs_df[recs_df["product_id"] == selected_pid]
    if matching_recs.empty:
        st.error(f"SKU {selected_pid} not found in inventory recommendations artifact.")
        return

    p_rec = matching_recs.iloc[0]
    p_name = p_rec.get("product_name", f"SKU {selected_pid}")
    p_fc = fixed_fc_df[fixed_fc_df["product_id"] == selected_pid].sort_values("forecast_horizon_step") if not fixed_fc_df.empty else pd.DataFrame()

    # Extract artifact origin and horizon
    p_origin = str(p_rec.get("forecast_origin", "Unknown"))
    if not p_fc.empty and "forecast_origin" in p_fc.columns:
        fc_origin = str(p_fc["forecast_origin"].iloc[0])
        if fc_origin and fc_origin != "nan":
            p_origin = fc_origin

    horizon_days = int(p_rec.get("forecast_horizon_days", len(p_fc))) if pd.notnull(p_rec.get("forecast_horizon_days")) else len(p_fc)
    min_fc_date = str(p_fc["forecast_date"].min()) if not p_fc.empty else "N/A"
    max_fc_date = str(p_fc["forecast_date"].max()) if not p_fc.empty else "N/A"

    # Check unavailable status
    is_unavailable = (
        p_rec.get("recommendation_status") == "UNAVAILABLE"
        or p_fc.empty
        or pd.isna(p_rec.get("expected_lead_time_demand"))
    )

    if is_unavailable:
        st.error(
            f"⚠️ **Recommendation Unavailable for SKU {selected_pid}**: "
            f"{p_rec.get('warning_or_limitation', 'Incomplete forecast horizon or invalid uncertainty inputs.')}. "
            "Data integrity guardrail: Missing or invalid recommendations are never silently set to zero."
        )
    else:
        # SKU Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Lead-Time Demand (LTD)", f"{p_rec['expected_lead_time_demand']:.1f} units", help=f"Sum of 7 daily forecasts ({min_fc_date} to 7th day)")
        col_m2.metric("Protection Demand (PPD)", f"{p_rec['expected_protection_period_demand']:.1f} units", help=f"Sum of {horizon_days} daily forecasts ({min_fc_date} to {max_fc_date})")
        col_m3.metric("Safety Stock (SS)", f"{p_rec['safety_stock']:.1f} units", help=f"z * σ_e * sqrt(L); Daily error std σ_e = {p_rec['forecast_error_std']:.2f}")
        col_m4.metric("Reorder Point (ROP)", f"{p_rec['reorder_point']:.1f} units", help="LTD + Safety Stock")

        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        col_m5.metric("Target Stock (S)", f"{p_rec['target_stock_level']:.1f} units", help="PPD + Protection Safety Stock")
        col_m6.metric("Simulated Net IP", f"{p_rec['inventory_position']:.1f} units", help="On-Hand (50) + On-Order (0) - Backorders (0)")
        
        reorder_trig = p_rec["reorder_triggered"]
        order_qty = p_rec["recommended_order_quantity"]
        col_m7.metric(
            "Recommended Order (Q)",
            f"{int(order_qty):,} units" if pd.notnull(order_qty) else "N/A",
            delta="Reorder Triggered (IP ≤ ROP)" if reorder_trig else "Adequate Stock",
            delta_color="inverse" if reorder_trig else "normal"
        )
        col_m8.metric("Optional EOQ Benchmark", f"{p_rec['optional_eoq_benchmark']:.1f} units", help="Planning benchmark; S=$50/PO, H=$2/unit/yr")

        st.write("")

        # 3. Fixed-Origin Multi-Step Recursive Forecast Trajectory
        st.markdown(f"#### 📅 Fixed-Origin {horizon_days}-Day Recursive Forecast Rollout — {p_name}")
        st.caption(
            f"Generated strictly from decision origin date **{p_origin}** across a **{horizon_days}-day** forward horizon "
            f"({min_fc_date} through {max_fc_date}). Day 1 uses actual demand through {p_origin}. "
            f"Days 2 through {horizon_days} recursively consume previous model predictions as autoregressive inputs. "
            f"**Zero future actual demand** is accessed."
        )

        col_chart, col_tbl = st.columns([2, 1])

        with col_chart:
            fig_fc = plot_dataco_fixed_horizon_forecast(
                p_fc,
                ltd=p_rec["expected_lead_time_demand"],
                rop=p_rec["reorder_point"],
                ss=p_rec["safety_stock"],
                target_stock=p_rec["target_stock_level"],
                product_name=p_name,
            )
            st.pyplot(fig_fc)
            plt.close(fig_fc)

        with col_tbl:
            st.markdown("##### Forecast Step Breakdown")
            tbl_disp = p_fc[["forecast_horizon_step", "forecast_date", "predicted_demand"]].copy()
            tbl_disp["predicted_demand"] = tbl_disp["predicted_demand"].apply(lambda v: f"{v:.2f}")
            tbl_disp.columns = ["Step", "Date", "Forecast (Units)"]
            st.dataframe(tbl_disp, use_container_width=True, hide_index=True)

    st.write("")

    # 4. Actionable Replenishment Orders Table
    render_section_heading(
        title="Simulated Replenishment Purchase Orders",
        subtitle="Filtered to products where simulated inventory position is at or below the reorder point (IP ≤ ROP).",
        icon="🛒",
    )

    try:
        orders_df = load_dataco_replenishment_orders()
        st.dataframe(orders_df, use_container_width=True, hide_index=True)
        csv_bytes = orders_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download DataCo Replenishment Orders (CSV)",
            data=csv_bytes,
            file_name="dataco_replenishment_orders.csv",
            mime="text/csv",
            key="download_dataco_orders_csv"
        )
    except Exception as err:
        st.warning(f"Could not load replenishment orders: {err}")
