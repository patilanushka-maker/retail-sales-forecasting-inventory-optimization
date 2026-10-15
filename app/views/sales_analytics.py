"""Sales Analytics view for the Retail Sales Forecasting & Inventory Optimization System.

Features:
1. Aggregate daily portfolio demand trend over time.
2. Top 10 products ranked by total recorded demand units.
3. Interactive SKU-level demand explorer preserving zero-demand days.
"""

from typing import Optional, Tuple
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.data_loader import (
    compute_product_period_statistics,
    get_available_product_ids,
    load_daily_demand_summary,
    load_product_daily_demand,
    load_top_products_demand,
)
from app.components import (
    format_metric_value,
    render_section_heading,
)
from src.retail_forecasting.visualization.plots import (
    plot_portfolio_daily_demand,
    plot_product_daily_demand,
    plot_top_products_bar,
)


def render_aggregate_demand_section() -> None:
    """Render aggregate historical daily demand trend across all catalog products."""
    render_section_heading(
        title="Portfolio-Wide Daily Demand Trend",
        subtitle="Total recorded units demanded daily across all catalog products (Dec 2010 – Dec 2011).",
        icon="📈",
    )

    try:
        summary_df = load_daily_demand_summary()

        # Display trend summary KPIs
        total_units = summary_df["total_demand"].sum()
        avg_daily = summary_df["total_demand"].mean()
        peak_day = summary_df.loc[summary_df["total_demand"].idxmax()]
        trough_day = summary_df.loc[summary_df["total_demand"].idxmin()]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            label="Total Recorded Units",
            value=format_metric_value(total_units, format_type="integer"),
            help="Total recorded units demanded across the 374-day observation period.",
        )
        col2.metric(
            label="Mean Daily Portfolio Demand",
            value=f"{avg_daily:,.1f} units/day",
            help="Average units demanded per day across the entire catalog.",
        )
        col3.metric(
            label="Peak Demand Day",
            value=f"{peak_day['total_demand']:,.0f} units",
            delta=str(pd.to_datetime(peak_day['date']).strftime("%b %d, %Y")),
            delta_color="off",
            help="Single day with highest aggregate recorded demand.",
        )
        col4.metric(
            label="Lowest Demand Day",
            value=f"{trough_day['total_demand']:,.0f} units",
            delta=str(pd.to_datetime(trough_day['date']).strftime("%b %d, %Y")),
            delta_color="off",
            help="Single day with lowest aggregate recorded demand (typically Sundays or holidays).",
        )

        # Plot Matplotlib chart
        fig = plot_portfolio_daily_demand(summary_df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "ℹ️ **Historical Fact Note**: This chart illustrates **recorded historical demand**, "
            "not a future forecast. The prominent peaks in November and early December 2011 reflect seasonal holiday gift-buying volume."
        )

    except Exception as err:
        st.error(f"Failed to render aggregate demand trend: {err}")


def render_top_products_section() -> None:
    """Render Top 10 products ranked by total recorded demand units."""
    render_section_heading(
        title="Top 10 Products by Total Recorded Demand",
        subtitle="Ranked strictly by recorded demand units across the full processed demand dataset.",
        icon="🏆",
    )

    try:
        top10_df = load_top_products_demand(n=10)

        col_chart, col_table = st.columns([1.6, 1.0])

        with col_chart:
            fig = plot_top_products_bar(top10_df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_table:
            st.markdown("##### 📋 Top 10 Volume Leaders")
            table_display = top10_df.copy()
            table_display.columns = ["Product ID (SKU)", "Total Units Demanded"]
            table_display["Total Units Demanded"] = table_display["Total Units Demanded"].apply(
                lambda v: f"{v:,.0f}"
            )
            table_display.index = range(1, len(table_display) + 1)
            st.dataframe(table_display, use_container_width=True)

        st.caption(
            "📌 **Methodology Guardrail**: Rankings are computed directly from `daily_product_demand.parquet`. "
            "Products are not ranked using Random Forest predictions or filtered by the forecast evaluation subset."
        )

    except Exception as err:
        st.error(f"Failed to render top products ranking: {err}")


def render_product_explorer_section() -> None:
    """Render SKU-level demand explorer with adjustable date range and zero-demand day preservation."""
    render_section_heading(
        title="Product-Level Demand Explorer",
        subtitle="Inspect individual SKU daily demand history, seasonal patterns, and zero-demand prevalence.",
        icon="🔍",
    )

    try:
        available_products = get_available_product_ids()
        if not available_products:
            st.warning("No products found in the processed daily demand dataset.")
            return

        # Select SKU with sensible default
        default_sku = "85123A" if "85123A" in available_products else available_products[0]
        default_index = available_products.index(default_sku)

        col_sku, col_dates = st.columns([1.0, 1.2])
        with col_sku:
            selected_sku = st.selectbox(
                "Select Product SKU:",
                options=available_products,
                index=default_index,
                help="Search or select any of the 3,804 product SKUs in the catalog.",
            )

        # Load product daily data
        product_df = load_product_daily_demand(selected_sku)
        if product_df.empty:
            st.info(f"No demand records found for Product {selected_sku}.")
            return

        min_product_date = pd.to_datetime(product_df["date"].min()).date()
        max_product_date = pd.to_datetime(product_df["date"].max()).date()

        with col_dates:
            date_selection = st.date_input(
                "Adjust Date Range Filter:",
                value=(min_product_date, max_product_date),
                min_value=min_product_date,
                max_value=max_product_date,
                help="Filter the product demand series to a specific historical window.",
            )

        # Parse selected date tuple safely
        if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
            start_date, end_date = date_selection
        elif isinstance(date_selection, (tuple, list)) and len(date_selection) == 1:
            start_date = end_date = date_selection[0]
        else:
            start_date, end_date = min_product_date, max_product_date

        # Compute period stats with zero-demand preservation
        stats = compute_product_period_statistics(
            df_product=product_df,
            start_date=start_date,
            end_date=end_date,
        )
        filtered_df = stats["filtered_df"]

        if filtered_df.empty:
            st.warning("No records match the selected date filter range.")
            return

        # Render SKU KPI cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            label="Selected Window Demand",
            value=format_metric_value(stats["total_demand"], format_type="integer"),
            help="Total recorded demand units over the selected date range.",
        )
        col2.metric(
            label="Mean Daily Demand",
            value=f"{stats['mean_demand']:.2f} units",
            help="Average daily recorded demand (including zero-demand days).",
        )
        col3.metric(
            label="Timeline Days",
            value=f"{stats['total_days']:,} days",
            help="Number of consecutive calendar days in the selected window.",
        )
        col4.metric(
            label="Active Sales Days",
            value=f"{stats['positive_demand_days']:,} days",
            delta=f"{(stats['positive_demand_days'] / max(1, stats['total_days'])) * 100:.1f}%",
            delta_color="normal",
            help="Days where at least 1 unit was recorded sold.",
        )
        col5.metric(
            label="Zero-Demand Days",
            value=f"{stats['zero_demand_days']:,} days",
            delta=f"{(stats['zero_demand_days'] / max(1, stats['total_days'])) * 100:.1f}%",
            delta_color="off",
            help="Calendar days with 0 demand. These are preserved from preprocessing.",
        )

        # Plot SKU time series
        fig = plot_product_daily_demand(filtered_df, product_id=selected_sku)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "⚠️ **Zero-Demand Interpretation Safeguard**: Zero-demand days reflect the project's preprocessing assumptions "
            "(daily calendar continuity established from each SKU's first recorded transaction). Zero demand does **not** prove "
            "that the product was physically in stock and available for purchase on that date (unobserved stockouts)."
        )

    except Exception as err:
        st.error(f"Failed to render product explorer: {err}")


def render_sales_analytics_page() -> None:
    """Render the full Sales & Demand Analytics dashboard page."""
    render_aggregate_demand_section()
    st.divider()
    render_top_products_section()
    st.divider()
    render_product_explorer_section()
