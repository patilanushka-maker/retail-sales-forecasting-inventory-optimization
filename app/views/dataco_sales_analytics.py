"""DataCo Smart Supply Chain: Sales Analytics view for Streamlit.

Presents:
1. Historical portfolio daily gross placed demand trends with rolling smoothing.
2. Interactive date range filtering and product filtering.
3. Top products demand comparison bar chart.
4. Department and category demand breakdowns.
5. Individual product daily demand time-series inspection.
"""

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.data_loader import (
    load_dataco_daily_demand,
    load_dataco_daily_demand_summary,
    load_dataco_top_products_demand,
    load_dataco_department_summary,
    load_dataco_catalog,
)
from app.components import (
    format_metric_value,
    render_dataco_disclaimer_banner,
    render_section_heading,
)
from src.retail_forecasting.visualization.plots import (
    plot_dataco_portfolio_daily_demand,
    plot_dataco_department_demand,
    plot_top_products_bar,
)


def render_dataco_sales_analytics_page() -> None:
    """Render DataCo Sales Analytics view."""
    render_dataco_disclaimer_banner()

    render_section_heading(
        title="DataCo Sales & Demand Analytics",
        subtitle="Exploration of gross placed retail demand across time, departments, categories, and products.",
        icon="📈",
    )

    try:
        summary_df = load_dataco_daily_demand_summary()
        catalog_df = load_dataco_catalog()
    except Exception as err:
        st.error(f"Error loading DataCo demand summary: {err}")
        return

    # Date Range Filter
    min_date = summary_df["date"].min().date()
    max_date = summary_df["date"].max().date()

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        start_date = st.date_input(
            "Start Date:",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="dataco_sales_start_date"
        )
    with col_filter2:
        end_date = st.date_input(
            "End Date:",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="dataco_sales_end_date"
        )

    if start_date > end_date:
        st.error("Error: Start Date must be earlier than or equal to End Date.")
        return

    # Filter summary
    filtered_summary = summary_df[
        (summary_df["date"].dt.date >= start_date) &
        (summary_df["date"].dt.date <= end_date)
    ].copy()

    total_period_demand = filtered_summary["total_demand"].sum()
    avg_daily_demand = filtered_summary["total_demand"].mean()
    peak_daily_demand = filtered_summary["total_demand"].max()
    peak_day_row = filtered_summary.loc[filtered_summary["total_demand"].idxmax()] if len(filtered_summary) > 0 else None

    # Summary KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Days", f"{len(filtered_summary):,} Days")
    col2.metric("Total Period Demand", f"{int(total_period_demand):,} Units")
    col3.metric("Avg Daily Demand", f"{avg_daily_demand:.1f} Units/Day")
    col4.metric(
        "Peak Daily Demand",
        f"{int(peak_daily_demand):,} Units" if peak_day_row is not None else "N/A",
        help=f"Peak occurred on {peak_day_row['date'].strftime('%Y-%m-%d')}" if peak_day_row is not None else ""
    )

    # 1. Daily Demand Trend Chart
    st.markdown("### 📊 Historical Daily Gross Placed Demand Trend")
    fig_portfolio = plot_dataco_portfolio_daily_demand(filtered_summary)
    st.pyplot(fig_portfolio)
    plt.close(fig_portfolio)

    st.write("")

    # 2. Department Breakdown & Top Products
    col_dept, col_top = st.columns(2)

    with col_dept:
        st.markdown("### 🏢 Demand by Department")
        dept_df = load_dataco_department_summary()
        fig_dept = plot_dataco_department_demand(dept_df)
        st.pyplot(fig_dept)
        plt.close(fig_dept)

    with col_top:
        st.markdown("### 🏆 Top 10 Products by Demand")
        top_df = load_dataco_top_products_demand(top_n=10)
        # Display top products table
        top_display = top_df[["product_id", "product_name", "department_name", "total_demand"]].copy()
        top_display["total_demand"] = top_display["total_demand"].apply(lambda v: f"{int(v):,}")
        top_display.columns = ["Product ID", "Product Name", "Department", "Total Units"]
        st.dataframe(top_display, use_container_width=True, hide_index=True)

    st.write("")

    # 3. Product-Level Time Series Explorer
    render_section_heading(
        title="Product-Level Daily Demand Explorer",
        subtitle="Select any product SKU from the catalog to inspect its individual daily demand history.",
        icon="🔍",
    )

    all_demand = load_dataco_daily_demand()
    all_pids = sorted(list(catalog_df["product_id"].unique()))
    name_lookup = dict(zip(catalog_df["product_id"], catalog_df["product_name"]))

    selected_pid = st.selectbox(
        "Choose a Product to Inspect:",
        options=all_pids,
        format_func=lambda pid: f"SKU {pid} — {name_lookup.get(pid, 'Unknown')}",
        key="dataco_analytics_product_select"
    )

    p_demand = all_demand[all_demand["product_id"] == selected_pid].sort_values("date")
    p_name = name_lookup.get(selected_pid, f"SKU {selected_pid}")

    p_total = float(p_demand["demand"].sum())
    p_active_days = int((p_demand["demand"] > 0).sum())
    p_zero_days = int((p_demand["demand"] == 0).sum())
    p_avg = float(p_demand["demand"].mean()) if len(p_demand) > 0 else 0.0

    pkpi1, pkpi2, pkpi3, pkpi4 = st.columns(4)
    pkpi1.metric("Lifetime Demand", f"{int(p_total):,} units")
    pkpi2.metric("Active Demand Days", f"{p_active_days:,} days")
    pkpi3.metric("Zero Demand Days", f"{p_zero_days:,} days")
    pkpi4.metric("Mean Daily Demand", f"{p_avg:.2f} units/day")

    # Product time series chart
    fig_prod, ax = plt.subplots(figsize=(11, 4))
    fig_prod.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")
    ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")

    ax.plot(p_demand["date"], p_demand["demand"], color="#38BDF8", linewidth=1.2, label=f"SKU {selected_pid} Daily Demand")
    rolling_p = p_demand["demand"].rolling(14, min_periods=3).mean()
    ax.plot(p_demand["date"], rolling_p, color="#F59E0B", linewidth=2.0, label="14-Day Rolling Mean")

    ax.set_title(f"Historical Daily Demand — {p_name} (SKU {selected_pid})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Demand Units", fontsize=10)
    ax.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#E2E8F0", loc="upper right")
    fig_prod.tight_layout()
    st.pyplot(fig_prod)
    plt.close(fig_prod)
