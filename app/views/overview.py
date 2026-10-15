"""Project Overview view for the Retail Sales Forecasting & Inventory Optimization System.

Presents:
1. Executive project introduction and scope.
2. Verified dataset summary metrics (demand units, distinct SKUs, date ranges).
3. Forecast evaluation scorecard (Baseline vs. Random Forest) with honest interpretation.
4. Simulated inventory replenishment summary.
5. End-to-end data science and operational workflow.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from app.data_loader import (
    get_daily_demand_portfolio_stats,
    load_forecast_comparison,
    load_forecast_metrics,
    load_inventory_recommendations,
)
from app.components import (
    format_metric_value,
    render_kpi_metrics,
    render_section_heading,
)


def render_project_introduction() -> None:
    """Render executive introduction explaining the system's core capabilities."""
    st.markdown(
        """
        ### 🎯 Executive Summary & System Objectives
        This decision support system provides a production-grade data science workflow that connects 
        **historical retail demand patterns** with **statistically grounded inventory optimization**.
        
        The system unites five core stages:
        1. **Transaction Data Preparation**: Cleaning raw transactions, handling returns and credit cancellations, and establishing uniform daily calendar continuity.
        2. **Daily Product Demand Aggregation**: Transforming discrete transactional events into standardized daily demand time series across the retail catalog.
        3. **Machine Learning Forecasting**: Engineering lag, rolling statistical, and calendar features to train a chronologically constrained Random Forest regressor.
        4. **Rigorous Out-of-Sample Evaluation**: Benchmarking machine learning accuracy against a persistent 7-day Seasonal Naive baseline over a fixed 28-day historical evaluation window.
        5. **Inventory Policy Calculations**: Generating periodic review $(R, s, S)$ replenishment recommendations and safety stock levels using explicitly simulated operational inputs.
        """
    )


def render_dataset_summary() -> None:
    """Render verified catalog and transaction summary statistics from actual artifacts."""
    render_section_heading(
        title="Dataset Architecture & Scope",
        subtitle="Summary derived directly from the processed demand and evaluation artifacts.",
        icon="📊",
    )

    try:
        portfolio_stats = get_daily_demand_portfolio_stats()
        fc_df = load_forecast_comparison()
        test_products_count = int(fc_df["product_id"].nunique())

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            label="Historical Timeline",
            value=f"{portfolio_stats['total_days']} Days",
            help=f"Dataset coverage from {portfolio_stats['start_date']} to {portfolio_stats['end_date']}.",
        )
        col2.metric(
            label="Catalog Products",
            value=format_metric_value(portfolio_stats["sku_count"], format_type="integer"),
            help="Total distinct product SKUs present in the processed daily demand dataset.",
        )
        col3.metric(
            label="Evaluated Products",
            value=format_metric_value(test_products_count, format_type="integer"),
            help="Number of products meeting full historical requirements for out-of-sample forecast evaluation.",
        )
        col4.metric(
            label="Processed Records",
            value=format_metric_value(portfolio_stats["row_count"], format_type="integer"),
            help="Total (date, product_id) observations in processed daily demand (including verified zero-demand days).",
        )
        col5.metric(
            label="Total Demand Units",
            value=format_metric_value(portfolio_stats["total_demand"], format_type="integer"),
            help="Total recorded units demanded across the historical catalog. Note: This represents physical units demanded, NOT financial revenue.",
        )

        st.caption(
            "ℹ️ **Data Definition Note**: All totals reflect **recorded physical demand units**, not currency revenue. "
            "The processed daily-demand dataset (`date`, `product_id`, `demand`) tracks unit quantities sold without price multipliers."
        )

    except Exception as err:
        st.error(f"Could not load dataset summary metrics: {err}")


def render_forecast_performance_section() -> None:
    """Render out-of-sample forecast evaluation comparison with honest interpretation."""
    render_section_heading(
        title="Forecast Performance: Baseline vs. Random Forest",
        subtitle="Evaluated strictly out-of-sample across the 28-day historical evaluation window (Nov 12 – Dec 9, 2011).",
        icon="🎯",
    )

    try:
        metrics_dict = load_forecast_metrics()
        overall = metrics_dict.get("overall_metrics", {})
        macro = metrics_dict.get("macro_averaged_metrics", {})

        rf_overall = overall.get("random_forest", {})
        sn_overall = overall.get("seasonal_naive", {})
        rf_macro = macro.get("random_forest", {})
        sn_macro = macro.get("seasonal_naive", {})

        # Extract values safely
        rf_wape = rf_overall.get("wape")
        sn_wape = sn_overall.get("wape")
        rf_mae = rf_overall.get("mae")
        sn_mae = sn_overall.get("mae")
        rf_rmse = rf_overall.get("rmse")
        sn_rmse = sn_overall.get("rmse")
        rf_macro_wape = rf_macro.get("macro_wape")
        sn_macro_wape = sn_macro.get("macro_wape")

        # Metric Scorecard Table
        scorecard_data = [
            {
                "Evaluation Metric": "Weighted Absolute Percentage Error (Pooled WAPE)",
                "Seasonal Naive Baseline": f"{sn_wape:.2f}%" if sn_wape is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_wape:.2f}%" if rf_wape is not None else "Unavailable",
                "Relative Impact": f"{(rf_wape - sn_wape):+.2f}%" if (rf_wape is not None and sn_wape is not None) else "N/A",
                "Interpretation": "Random Forest improves pooled error across portfolio volume.",
            },
            {
                "Evaluation Metric": "Mean Absolute Error (MAE in Daily Units)",
                "Seasonal Naive Baseline": f"{sn_mae:.2f}" if sn_mae is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_mae:.2f}" if rf_mae is not None else "Unavailable",
                "Relative Impact": f"{(rf_mae - sn_mae):+.2f}" if (rf_mae is not None and sn_mae is not None) else "N/A",
                "Interpretation": "Random Forest reduces average daily unit prediction error by ~1.33 units.",
            },
            {
                "Evaluation Metric": "Root Mean Squared Error (RMSE)",
                "Seasonal Naive Baseline": f"{sn_rmse:.2f}" if sn_rmse is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_rmse:.2f}" if rf_rmse is not None else "Unavailable",
                "Relative Impact": f"{(rf_rmse - sn_rmse):+.2f}" if (rf_rmse is not None and sn_rmse is not None) else "N/A",
                "Interpretation": "Random Forest reduces large outliers and peak demand errors by 18.2%.",
            },
            {
                "Evaluation Metric": "Macro-Averaged WAPE (Equal Weight per Product)",
                "Seasonal Naive Baseline": f"{sn_macro_wape:.2f}%" if sn_macro_wape is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_macro_wape:.2f}%" if rf_macro_wape is not None else "Unavailable",
                "Relative Impact": f"{(rf_macro_wape - sn_macro_wape):+.2f}%" if (rf_macro_wape is not None and sn_macro_wape is not None) else "N/A",
                "Interpretation": "Seasonal Naive achieves better macro WAPE on low-volume/intermittent items.",
            },
        ]
        st.dataframe(pd.DataFrame(scorecard_data), use_container_width=True, hide_index=True)

        # Honest Technical Interpretation
        with st.expander("🔍 In-Depth Interpretation of Model Trade-offs", expanded=True):
            st.markdown(
                """
                - **Volume-Weighted Superiority**: Random Forest demonstrates clear superiority on **Pooled WAPE (97.20% vs. 110.88%)**, **MAE (9.44 vs. 10.77)**, and **RMSE (34.79 vs. 42.56)**. Because these metrics weight errors by demand volume, they reflect Random Forest's ability to capture calendar patterns and multi-week trend signals on high-velocity items.
                - **Macro WAPE Trade-off**: Random Forest exhibits higher **Macro WAPE (189.95% vs. 146.29%)**. Macro WAPE calculates percentage error for each product individually and averages them unweighted. On sporadic or intermittent products (which comprise over 70% of catalog SKUs), machine learning models tend to predict small continuous positive values, inflating relative percentage error on zero-demand days.
                - **Methodological Conclusion**: Machine learning forecasting does **not** establish uniformly superior performance across all SKUs. While highly effective for steady, fast-moving inventory, intermittent products benefit from specialized baseline treatment or pooled uncertainty fallbacks.
                """
            )

    except Exception as err:
        st.error(f"Could not load forecast metrics: {err}")


def render_inventory_summary_section() -> None:
    """Render summary of simulated inventory recommendations and replenishment orders."""
    render_section_heading(
        title="Inventory Optimization & Replenishment Summary",
        subtitle="Operational decisions generated using periodic review (R, s, S) logic as of November 11, 2011.",
        icon="📦",
    )

    try:
        rec_df = load_inventory_recommendations()

        total_recs = len(rec_df)
        reorder_now_count = int((rec_df["recommendation_status"] == "REORDER_NOW").sum())
        stock_adequate_count = int((rec_df["recommendation_status"] == "STOCK_ADEQUATE").sum())
        unavailable_count = int((rec_df["recommendation_status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON").sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            label="Total SKU Decisions",
            value=f"{total_recs:,}",
            help="Total products evaluated for replenishment recommendations.",
        )
        col2.metric(
            label="🚨 Reorder Required (Simulated)",
            value=f"{reorder_now_count:,}",
            delta=f"{(reorder_now_count / total_recs) * 100:.1f}% of catalog",
            delta_color="inverse",
            help="Items where simulated inventory position is at or below the calculated Reorder Point (IP <= ROP).",
        )
        col3.metric(
            label="✅ Sufficient Stock",
            value=f"{stock_adequate_count:,}",
            delta=f"{(stock_adequate_count / total_recs) * 100:.1f}% of catalog",
            delta_color="normal",
            help="Items with adequate simulated inventory position to cover the protection period.",
        )
        col4.metric(
            label="⚠️ Horizon Insufficient",
            value=f"{unavailable_count:,}",
            delta=f"{(unavailable_count / total_recs) * 100:.1f}% of catalog",
            delta_color="off",
            help="Items where required protection period exceeds the 28-day forecast horizon (H < L + R).",
        )

        st.caption(
            "⚠️ **Methodology Caveat**: Initial stock levels, supplier lead times (7 days), review cycles (7 days), "
            "and cost parameters (£1.50/unit/yr holding, £25/order) are **simulated operational inputs** designed to demonstrate "
            "optimization mechanics. Recommendations represent a retrospective demonstration as of the November 11, 2011 cutoff date."
        )

    except Exception as err:
        st.error(f"Could not load inventory recommendations: {err}")


def render_workflow_explanation() -> None:
    """Render readable end-to-end pipeline architecture workflow."""
    render_section_heading(
        title="End-to-End System Architecture",
        subtitle="Strict chronological progression from raw retail logs to purchase orders.",
        icon="🔄",
    )

    st.markdown(
        """
        ```mermaid
        flowchart LR
            subgraph S1 [Phase 1 & 2: Data Pipeline]
                A[Online Retail Transactions] --> B[Cleaning & Returns Filtering]
                B --> C[Daily Demand Matrix: 3,804 SKUs x 374 Days]
            end
            
            subgraph S2 [Phase 3 & 4: Feature Store]
                C --> D[Lags: 1, 7, 14, 28 Days]
                C --> E[Rolling Means & Stds]
                C --> F[Calendar Features: Day, Month, Weekend]
            end
            
            subgraph S3 [Phase 5: Forecasting Engine]
                D & E & F --> G[Chronological Cutoff: Nov 11, 2011]
                G --> H[Random Forest Regressor]
                G --> I[7-Day Seasonal Naive Baseline]
                H & I --> J[Out-of-Sample Backtest: 28 Days]
            end
            
            subgraph S4 [Phase 6 & 7: Inventory & UI]
                J --> K[Forecast Demand & Error Std: RMSE]
                K --> L[Periodic Review R, s, S Policy]
                L --> M[Replenishment Orders & Dashboard]
            end
        ```
        """
    )
    st.info(
        "📌 **Backtest Operational Scope**: This dashboard demonstrates historical backtest performance and decision support simulations. "
        "It does not poll live enterprise data or generate automated current-day purchase orders.",
        icon="ℹ️",
    )


def render_overview_page() -> None:
    """Render the full Project Overview dashboard page."""
    render_project_introduction()
    st.divider()
    render_dataset_summary()
    st.divider()
    render_forecast_performance_section()
    st.divider()
    render_inventory_summary_section()
    st.divider()
    render_workflow_explanation()
