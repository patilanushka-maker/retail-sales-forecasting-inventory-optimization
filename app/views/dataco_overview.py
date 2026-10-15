"""DataCo Smart Supply Chain: Project Overview view for Streamlit.

Presents:
1. Executive introduction and system scope for the DataCo dataset.
2. Verified dataset summary metrics (Timeline, Catalog SKUs, Core Cohort, Records, Total Demand).
3. Demand definition safeguards (Gross Placed Demand vs fulfilled sales).
4. Out-of-sample forecast benchmark evaluation scorecard (Seasonal Naive vs Rolling Mean vs Random Forest).
5. Simulated inventory optimization summary for Core 9 cohort.
6. End-to-end data science and inventory optimization pipeline architecture.
"""

from typing import Any, Dict
import streamlit as st

from app.data_loader import (
    get_dataco_portfolio_stats,
    load_dataco_forecast_metrics,
    load_dataco_inventory_recommendations,
)
from app.components import (
    format_metric_value,
    render_dataco_disclaimer_banner,
    render_section_heading,
)


def render_dataco_introduction() -> None:
    """Render executive introduction explaining DataCo system scope."""
    st.markdown(
        """
        ### 🎯 Executive Summary & DataCo System Scope
        This decision support system provides an end-to-end, leak-free data science and supply chain optimization workflow 
        tailored for the **DataCo Smart Supply Chain** dataset.
        
        The DataCo pipeline unites five core stages:
        1. **Demand Definition & Preprocessing**: Cleaned raw transactions into daily product demand series under an audited gross placed retail demand policy (excluding cancellations and suspected fraud).
        2. **Lifespan Continuity & Reindexing**: Established uniform daily calendar series per product with lifespan zero-filling, preserving true catalog lifespans.
        3. **Out-of-Sample Benchmark Forecasting**: Compared simple baselines (7-day Seasonal Naive, 7-day Rolling Mean) against a tabular autoregressive Random Forest regressor on identical test observations.
        4. **Fixed-Origin Multi-Step Recursive Forecasting**: Generated leak-free forward predictions from decision origin $\\tau = \\text{2017-08-31}$ across an 8-day horizon with zero test actuals consumed.
        5. **Periodic Review Inventory Optimization**: Sized safety stocks and reorder thresholds using pre-decision out-of-sample residual uncertainty and simulated operational parameters.
        """
    )


def render_dataco_dataset_summary() -> None:
    """Render verified catalog and transaction summary statistics from actual DataCo artifacts."""
    render_section_heading(
        title="Dataset Architecture & Scope",
        subtitle="Summary derived directly from processed DataCo demand and catalog artifacts.",
        icon="📊",
    )

    try:
        stats = get_dataco_portfolio_stats()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            label="Historical Timeline",
            value=f"{stats['total_days']:,} Days",
            help=f"Dataset coverage from {stats['start_date']} to {stats['end_date']}.",
        )
        col2.metric(
            label="Catalog Products",
            value=format_metric_value(stats["sku_count"], format_type="integer"),
            help="Total distinct product SKUs present in the processed daily demand dataset.",
        )
        col3.metric(
            label="Core Benchmark Cohort",
            value=format_metric_value(stats["core_cohort_count"], format_type="integer"),
            help="The 9 highest-volume products with continuous lifespan and active demand (2015-01-01 to 2017-10-02).",
        )
        col4.metric(
            label="Daily Demand Records",
            value=format_metric_value(stats["row_count"], format_type="integer"),
            help="Total (date, product_id) observations in processed daily demand (including verified zero-demand days).",
        )
        col5.metric(
            label="Total Gross Demand",
            value=format_metric_value(stats["total_demand"], format_type="integer"),
            help="Total recorded gross placed demand units across the catalog. Note: Physical units, NOT currency revenue.",
        )

        st.caption(
            "ℹ️ **Demand Definition Guardrail**: Demand is defined as **Gross Placed Retail Demand** "
            "(Order Item Quantity summed across placed customer orders, excluding `CANCELED` and `SUSPECTED_FRAUD`). "
            "Raw order transactions total **180,519 records**. Crucially, transaction logs reflect customer placed orders, "
            "not inventory fulfillment or stockout events. Unmet customer demand cannot be reconstructed."
        )

    except Exception as err:
        st.error(f"Could not load DataCo dataset summary metrics: {err}")


def render_dataco_forecast_performance_section() -> None:
    """Render out-of-sample forecast evaluation comparison for DataCo."""
    render_section_heading(
        title="Forecast Performance: Baseline vs. Random Forest Benchmark",
        subtitle="Evaluated out-of-sample across the 32-day test window (Sept 1 – Oct 2, 2017) on identical records.",
        icon="🎯",
    )

    try:
        metrics = load_dataco_forecast_metrics()
        pooled = metrics.get("pooled_metrics", {})

        col1, col2, col3 = st.columns(3)

        # Baseline A: Seasonal Naive
        sn = pooled.get("Seasonal Naive", {})
        with col1:
            st.markdown("#### 1. Seasonal Naive (7d)")
            st.caption("Baseline A: Persistence lag $\\hat{y}_t = y_{t-7}$")
            st.metric("MAE", f"{sn.get('mae', 0):.2f} units")
            st.metric("RMSE", f"{sn.get('rmse', 0):.2f} units")
            st.metric("WAPE", f"{sn.get('wape', 0):.2f}%")

        # Baseline B: Rolling Mean (7d)
        rm = pooled.get("Rolling Mean (7d)", {})
        with col2:
            st.markdown("#### 2. Rolling Mean (7d)")
            st.caption("Baseline B: 7-day trailing mean shifted by 1")
            st.metric("MAE", f"{rm.get('mae', 0):.2f} units", delta=f"{sn.get('mae', 0) - rm.get('mae', 0):+.2f} vs SN")
            st.metric("RMSE", f"{rm.get('rmse', 0):.2f} units", delta=f"{sn.get('rmse', 0) - rm.get('rmse', 0):+.2f} vs SN")
            st.metric("WAPE", f"{rm.get('wape', 0):.2f}%", delta=f"{sn.get('wape', 0) - rm.get('wape', 0):+.2f}%")

        # Benchmark C: Random Forest
        rf = pooled.get("Random Forest", {})
        with col3:
            st.markdown("#### 3. Random Forest")
            st.caption("Benchmark C: Autoregressive tabular model")
            st.metric("MAE", f"{rf.get('mae', 0):.2f} units", delta=f"{sn.get('mae', 0) - rf.get('mae', 0):+.2f} vs SN")
            st.metric("RMSE", f"{rf.get('rmse', 0):.2f} units", delta=f"{sn.get('rmse', 0) - rf.get('rmse', 0):+.2f} vs SN")
            st.metric("WAPE", f"{rf.get('wape', 0):.2f}%", delta=f"{sn.get('wape', 0) - rf.get('wape', 0):+.2f}%")

        st.info(
            "💡 **Methodological Protocol Note**: The metrics above evaluate a **rolling 1-day-ahead benchmark** across "
            "288 identical test observations. In contrast, inventory replenishment recommendations are driven by a "
            "**fixed-origin 8-day recursive rollout** starting from August 31, 2017. Rolling metrics do not establish "
            "multi-step recursive accuracy, but demonstrate the baseline model hierarchy.",
            icon="ℹ️",
        )

    except Exception as err:
        st.error(f"Could not load DataCo forecast metrics: {err}")


def render_dataco_inventory_summary_section() -> None:
    """Render simulated inventory optimization scorecard for DataCo Core 9."""
    render_section_heading(
        title="Simulated Inventory Replenishment Summary (Core 9 Cohort)",
        subtitle="Policy parameters: CSL = 95%, L = 7 days, R = 1 day, simulated starting stock = 50 units.",
        icon="📦",
    )

    try:
        recs_df = load_dataco_inventory_recommendations()
        total_skus = len(recs_df)
        reorders_triggered = int(recs_df["reorder_triggered"].sum())
        total_order_qty = int(recs_df["recommended_order_quantity"].dropna().sum())
        avg_ltd = float(recs_df["expected_lead_time_demand"].mean())
        avg_ss = float(recs_df["safety_stock"].mean())

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Core Cohort SKUs", f"{total_skus}")
        col2.metric("Reorder Triggered", f"{reorders_triggered} / {total_skus}")
        col3.metric("Total Replenishment Units", f"{total_order_qty:,}")
        col4.metric("Avg Lead-Time Demand", f"{avg_ltd:.1f} units")
        col5.metric("Avg Safety Stock Buffer", f"{avg_ss:.1f} units")

        st.caption(
            "⚠️ **Simulation Disclaimer**: DataCo records outbound courier delivery times (0-6 days), not inbound supplier "
            "replenishment lead times. Starting on-hand stock (50 units), supplier lead time (7 days), review period (1 day), "
            "and unit holding costs ($2.00/yr) are simulated demonstration parameters."
        )

    except Exception as err:
        st.error(f"Could not load DataCo inventory summary: {err}")


def render_dataco_workflow_architecture() -> None:
    """Render end-to-end DataCo data science workflow."""
    render_section_heading(
        title="DataCo System Workflow Architecture",
        subtitle="End-to-end data pipeline from raw customer transactions to inventory decision support.",
        icon="🔄",
    )

    st.markdown(
        """
        ```
        [Raw Transactions] (180,519 rows)
               │  Filter: Exclude CANCELED & SUSPECTED_FRAUD
               ▼
        [Daily Demand Series] (21,787 rows, 118 catalog products)
               │  Cohort Selection: Core 9 continuous products (2015-01 to 2017-10)
               ▼
        [Temporal Split Boundaries]
        ├── Train Window: 2015-01-01 to 2017-06-30 (912 days)
        ├── Validation Window: 2017-07-01 to 2017-08-31 (62 days)  ──► Residual Uncertainty (sigma_e)
        └── Test Window: 2017-09-01 to 2017-10-02 (32 days)        ──► Rolling Benchmark Evaluation
               │
               ▼
        [Fixed-Origin Multi-Step Forecasting] (tau = 2017-08-31)
               │  Recursive 8-Day Forward Rollout (2017-09-01 to 2017-09-08)
               │  Zero Future Actual Demand Consumed
               ▼
        [Inventory Optimization Engine]
        ├── Net Position: IP = On-Hand + On-Order - Backorders
        ├── Safety Stock: SS = z * sigma_e * sqrt(L)
        ├── Reorder Point: ROP = LTD + SS
        ├── Target Stock: S = PPD + SS_protection
        └── Order Quantity: Q = max(0, ceil(S - IP)) if IP <= ROP else 0
        ```
        """
    )


def render_dataco_overview_page() -> None:
    """Entry point for the DataCo Overview view."""
    render_dataco_disclaimer_banner()
    render_dataco_introduction()
    st.write("")
    render_dataco_dataset_summary()
    st.write("")
    render_dataco_forecast_performance_section()
    st.write("")
    render_dataco_inventory_summary_section()
    st.write("")
    render_dataco_workflow_architecture()
