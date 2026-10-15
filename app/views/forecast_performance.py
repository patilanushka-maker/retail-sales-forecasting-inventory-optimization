"""Forecast Performance and Model Comparison view for the Streamlit dashboard.

Displays:
1. Out-of-sample evaluation context, horizon, and chronological protocol.
2. Global performance scorecard (Seasonal Naive vs. Random Forest) with balanced interpretation.
3. Interactive product-level actual-versus-predicted evaluation chart.
4. Single-SKU evaluation metrics (MAE, RMSE, WAPE) with safe zero-denominator handling.
5. Technical methodology and evaluation safeguards.
"""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.data_loader import (
    compute_single_product_forecast_metrics,
    get_evaluated_product_ids,
    load_forecast_comparison,
    load_forecast_metrics,
    load_product_forecast_comparison,
)
from app.components import (
    format_metric_value,
    render_retrospective_warning,
    render_section_heading,
)
from src.retail_forecasting.visualization.plots import plot_forecast_vs_actual


def render_evaluation_context_section(fc_df: pd.DataFrame) -> None:
    """Render the evaluation context, chronological cutoffs, and observation totals."""
    render_section_heading(
        title="Out-of-Sample Evaluation Protocol & Horizon",
        subtitle="Chronologically isolated 28-day rolling evaluation strictly post-dating the training cutoff.",
        icon="⏱️",
    )

    total_obs = len(fc_df)
    total_skus = int(fc_df["product_id"].nunique())
    min_date = str(fc_df["date"].min().date())
    max_date = str(fc_df["date"].max().date())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        label="Planning Cutoff Date",
        value="Nov 11, 2011",
        help="All training features and model fitting strictly predate this cutoff.",
    )
    col2.metric(
        label="Evaluation Horizon",
        value="28 Days",
        help=f"Out-of-sample backtest window spanning {min_date} to {max_date}.",
    )
    col3.metric(
        label="Evaluation Protocol",
        value="Rolling 1-Day",
        help="One-day-ahead rolling forecasts where realized demand is updated only after each date elapses.",
    )
    col4.metric(
        label="Evaluated Observations",
        value=format_metric_value(total_obs, format_type="integer"),
        help="Total aligned (product, date) pairs evaluated across the test window.",
    )
    col5.metric(
        label="Evaluated Products",
        value=format_metric_value(total_skus, format_type="integer"),
        help="Number of catalog products with complete historical features for out-of-sample evaluation.",
    )

    st.caption(
        "📌 **Historical Out-of-Sample Notice**: All predictions displayed on this page evaluate historical observations "
        "from November 12 – December 9, 2011. They demonstrate retrospective algorithmic performance under simulated historical "
        "conditions and do **not** represent live or forward-looking retail forecasts."
    )


def render_model_comparison_scorecard() -> None:
    """Render comparative scorecard between Seasonal Naive and Random Forest with balanced interpretation."""
    render_section_heading(
        title="Model Comparison Scorecard: Baseline vs. Random Forest",
        subtitle="Saved evaluation metrics loaded directly from outputs/metrics/forecast_metrics.json.",
        icon="⚖️",
    )

    try:
        metrics_dict = load_forecast_metrics()
        overall = metrics_dict.get("overall_metrics", {})
        macro = metrics_dict.get("macro_averaged_metrics", {})

        rf_overall = overall.get("random_forest", {})
        sn_overall = overall.get("seasonal_naive", {})
        rf_macro = macro.get("random_forest", {})
        sn_macro = macro.get("seasonal_naive", {})

        # Extract values
        rf_wape = rf_overall.get("wape")
        sn_wape = sn_overall.get("wape")
        rf_mae = rf_overall.get("mae")
        sn_mae = sn_overall.get("mae")
        rf_rmse = rf_overall.get("rmse")
        sn_rmse = sn_overall.get("rmse")
        rf_macro_wape = rf_macro.get("macro_wape")
        sn_macro_wape = sn_macro.get("macro_wape")

        scorecard = [
            {
                "Evaluation Metric": "Pooled WAPE (%)",
                "Seasonal Naive Baseline": f"{sn_wape:.2f}%" if sn_wape is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_wape:.2f}%" if rf_wape is not None else "Unavailable",
                "Difference (RF - Baseline)": f"{(rf_wape - sn_wape):+.2f}%" if (rf_wape is not None and sn_wape is not None) else "N/A",
                "Better Model": "Random Forest" if (rf_wape is not None and sn_wape is not None and rf_wape < sn_wape) else "Baseline",
            },
            {
                "Evaluation Metric": "Mean Absolute Error (MAE Units)",
                "Seasonal Naive Baseline": f"{sn_mae:.2f}" if sn_mae is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_mae:.2f}" if rf_mae is not None else "Unavailable",
                "Difference (RF - Baseline)": f"{(rf_mae - sn_mae):+.2f}" if (rf_mae is not None and sn_mae is not None) else "N/A",
                "Better Model": "Random Forest" if (rf_mae is not None and sn_mae is not None and rf_mae < sn_mae) else "Baseline",
            },
            {
                "Evaluation Metric": "Root Mean Squared Error (RMSE)",
                "Seasonal Naive Baseline": f"{sn_rmse:.2f}" if sn_rmse is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_rmse:.2f}" if rf_rmse is not None else "Unavailable",
                "Difference (RF - Baseline)": f"{(rf_rmse - sn_rmse):+.2f}" if (rf_rmse is not None and sn_rmse is not None) else "N/A",
                "Better Model": "Random Forest" if (rf_rmse is not None and sn_rmse is not None and rf_rmse < sn_rmse) else "Baseline",
            },
            {
                "Evaluation Metric": "Macro-Averaged WAPE (%)",
                "Seasonal Naive Baseline": f"{sn_macro_wape:.2f}%" if sn_macro_wape is not None else "Unavailable",
                "Random Forest Forecaster": f"{rf_macro_wape:.2f}%" if rf_macro_wape is not None else "Unavailable",
                "Difference (RF - Baseline)": f"{(rf_macro_wape - sn_macro_wape):+.2f}%" if (rf_macro_wape is not None and sn_macro_wape is not None) else "N/A",
                "Better Model": "Seasonal Naive" if (rf_macro_wape is not None and sn_macro_wape is not None and sn_macro_wape < rf_macro_wape) else "Random Forest",
            },
        ]
        st.dataframe(pd.DataFrame(scorecard), use_container_width=True, hide_index=True)

        # Balanced technical interpretation
        st.markdown("#### ⚖️ Balanced Evaluation Assessment")
        st.info(
            "• **Volume-Weighted Superiority**: Random Forest achieves significantly lower **MAE (9.44 vs. 10.77)**, "
            "**RMSE (34.79 vs. 42.56)**, and **Pooled WAPE (97.20% vs. 110.88%)**. Because these metrics sum absolute errors across "
            "all units, they demonstrate that Random Forest effectively learns holiday trends, day-of-week seasonality, and lag patterns "
            "on high-volume products that generate the bulk of retail throughput.  \n"
            "• **Unweighted Macro Disadvantage**: Conversely, Random Forest yields higher **Macro WAPE (189.95% vs. 146.29%)**. "
            "Macro WAPE weights each of the 2,813 SKUs equally regardless of volume. On intermittent, sporadic items with sparse sales, "
            "tree models predict continuous fractional positive values (e.g. 0.3 units) on zero-demand days, generating large relative percentages.  \n"
            "• **Synthesis**: Model comparison is **mixed**. Random Forest is superior for portfolio aggregate volume and high-velocity SKUs, "
            "while the Seasonal Naive baseline retains competitive stability on low-velocity, intermittent demand.",
            icon="⚖️",
        )

    except Exception as err:
        st.error(f"Failed to load forecast metrics: {err}")


def render_product_level_forecast_section() -> None:
    """Render interactive product-level forecast comparison chart and item metrics."""
    render_section_heading(
        title="Product-Level Forecast Comparison",
        subtitle="Compare actual demand against Random Forest and Seasonal Naive baseline predictions for any evaluated SKU.",
        icon="🔍",
    )

    try:
        evaluated_skus = get_evaluated_product_ids()
        if not evaluated_skus:
            st.warning("No evaluated products found in the forecast comparison artifact.")
            return

        # Default to a high-volume SKU with 28 days of evaluation
        default_sku = "85123A" if "85123A" in evaluated_skus else evaluated_skus[0]
        default_idx = evaluated_skus.index(default_sku)

        col_select, col_info = st.columns([1.0, 1.4])
        with col_select:
            selected_sku = st.selectbox(
                "Select Evaluated Product SKU:",
                options=evaluated_skus,
                index=default_idx,
                help="Select from 2,813 product SKUs evaluated in the out-of-sample backtest.",
            )

        # Load product evaluation slice
        df_product = load_product_forecast_comparison(selected_sku)

        if df_product.empty:
            st.warning(f"No evaluation records available for product '{selected_sku}'.")
            return

        with col_info:
            st.caption(
                f"**Evaluation Horizon for SKU {selected_sku}**:  \n"
                f"Dates: {df_product['date'].min().date()} to {df_product['date'].max().date()} "
                f"({len(df_product)} rolling evaluation days)"
            )

        # Plot actual vs predictions
        fig = plot_forecast_vs_actual(
            dates=df_product["date"],
            y_actual=df_product["actual_demand"],
            y_pred=df_product["random_forest_pred"],
            baseline_pred=df_product["seasonal_naive_pred"],
            title=f"Out-of-Sample Demand & Predictions: SKU {selected_sku} (Nov 12 – Dec 9, 2011)",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Calculate product-level metrics safely
        product_metrics = compute_single_product_forecast_metrics(df_product)

        st.markdown(f"##### 📊 Accuracy Scorecard for SKU: `{selected_sku}`")
        if product_metrics["missing_count"] > 0:
            st.warning(
                f"⚠️ Note: {product_metrics['missing_count']} records contain missing prediction data and were omitted from metric calculation."
            )

        rf_m = product_metrics["random_forest"]
        sn_m = product_metrics["seasonal_naive"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            label="Evaluated Days",
            value=f"{product_metrics['sample_size']} Days",
            help="Total evaluation days evaluated for this SKU.",
        )
        col2.metric(
            label="Total Actual Demand",
            value=f"{product_metrics['total_actual']:,.0f} units",
            help="Ground truth demand demanded over the 28-day window.",
        )

        # Format WAPE with safe zero-denominator handling
        if rf_m["wape_defined"] and sn_m["wape_defined"]:
            rf_wape_str = f"{rf_m['wape']:.1f}%"
            wape_delta = f"{(rf_m['wape'] - sn_m['wape']):+.1f}% vs Baseline"
        elif not rf_m["wape_defined"]:
            rf_wape_str = "Unavailable (0 actuals)"
            wape_delta = None
        else:
            rf_wape_str = "Unavailable"
            wape_delta = None

        col3.metric(
            label="Random Forest WAPE",
            value=rf_wape_str,
            delta=wape_delta,
            delta_color="inverse",
            help="Weighted Absolute Percentage Error on this SKU over the evaluation window.",
        )

        mae_delta = (
            f"{(rf_m['mae'] - sn_m['mae']):+.2f} vs Baseline"
            if (not np.isnan(rf_m["mae"]) and not np.isnan(sn_m["mae"]))
            else None
        )
        col4.metric(
            label="Random Forest MAE",
            value=f"{rf_m['mae']:.2f} units",
            delta=mae_delta,
            delta_color="inverse",
            help="Mean Absolute Error in daily units.",
        )

        # Detailed product metrics comparison table
        sku_scorecard = [
            {
                "Model": "Random Forest Regressor",
                "MAE (Units)": f"{rf_m['mae']:.2f}" if not np.isnan(rf_m["mae"]) else "N/A",
                "RMSE (Units)": f"{rf_m['rmse']:.2f}" if not np.isnan(rf_m["rmse"]) else "N/A",
                "WAPE (%)": f"{rf_m['wape']:.2f}%" if rf_m["wape_defined"] else "Unavailable (Zero actuals)",
            },
            {
                "Model": "7-Day Seasonal Naive Baseline",
                "MAE (Units)": f"{sn_m['mae']:.2f}" if not np.isnan(sn_m["mae"]) else "N/A",
                "RMSE (Units)": f"{sn_m['rmse']:.2f}" if not np.isnan(sn_m["rmse"]) else "N/A",
                "WAPE (%)": f"{sn_m['wape']:.2f}%" if sn_m["wape_defined"] else "Unavailable (Zero actuals)",
            },
        ]
        st.dataframe(pd.DataFrame(sku_scorecard), use_container_width=True, hide_index=True)

    except Exception as err:
        st.error(f"Failed to render product forecast comparison: {err}")


def render_forecast_methodology_section() -> None:
    """Render technical explanation of rolling evaluation and leakage prevention."""
    render_section_heading(
        title="Forecasting Methodology & Leakage Prevention",
        subtitle="Rigorous engineering safeguards applied during Phase 5 evaluation.",
        icon="🛡️",
    )

    with st.expander("🔬 Technical Architecture & Evaluation Design", expanded=False):
        st.markdown(
            """
            1. **Strict Chronological Split**:
               - **Training Data**: Dec 1, 2010 to Nov 11, 2011 (983,489 observations).
               - **Cutoff Date**: November 11, 2011. No feature extraction, target statistics, or rolling windows use data across the cutoff.
               - **Out-of-Sample Window**: Nov 12, 2011 to Dec 9, 2011 (28 days).
            2. **Rolling One-Day-Ahead Protocol**:
               - Forecasts for evaluation date $t$ are constructed using information available strictly up to day $t-1$.
               - As each evaluation day elapses, realized demand updates lag and rolling statistics for subsequent days without future lookahead.
            3. **Baseline Definition**:
               - The benchmark is a 7-day Seasonal Naive model ($\hat{y}_{t} = y_{t-7}$), capturing weekly periodicity.
            4. **Metric Definitions**:
               - **MAE**: $\\frac{1}{N} \\sum |y_i - \\hat{y}_i|$ (linear penalty on unit error).
               - **RMSE**: $\\sqrt{\\frac{1}{N} \\sum (y_i - \\hat{y}_i)^2}$ (penalizes large peak misses).
               - **Pooled WAPE**: $\\frac{\\sum |y_i - \\hat{y}_i|}{\\sum |y_i|} \\times 100$ (portfolio volume-weighted percentage error).
               - **Macro WAPE**: Unweighted average of per-product WAPE values (sensitive to intermittent SKUs).
            5. **Operational Caution**:
               - Demonstration predictions reflect historical backtest performance. They do not constitute current operational forecasts.
            """
        )


def render_forecast_performance_page() -> None:
    """Render the complete Forecast Performance dashboard page."""
    render_retrospective_warning()
    st.write("")

    fc_df = load_forecast_comparison()
    render_evaluation_context_section(fc_df)
    st.divider()
    render_model_comparison_scorecard()
    st.divider()
    render_product_level_forecast_section()
    st.divider()
    render_forecast_methodology_section()
