"""DataCo Smart Supply Chain: Forecast Performance & Benchmark view for Streamlit.

Presents:
1. Out-of-sample rolling benchmark evaluation results on test period (Sept 1 – Oct 2, 2017).
2. Explicit methodological distinction between rolling 1-day benchmark and fixed-origin multi-step forecasting.
3. Model comparison scorecard (Seasonal Naive vs Rolling Mean vs Random Forest).
4. Mathematical metric definitions (MAE, RMSE, WAPE).
5. Per-product accuracy breakdown across the Core 9 cohort.
6. Actual vs. Forecast visualization across the test period.
"""

from typing import Any, Dict
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.data_loader import (
    load_dataco_forecast_comparison,
    load_dataco_forecast_metrics,
    load_dataco_catalog,
)
from app.components import (
    format_metric_value,
    render_dataco_disclaimer_banner,
    render_section_heading,
)
from src.retail_forecasting.visualization.plots import (
    plot_dataco_model_comparison,
    plot_dataco_product_forecast_vs_actual,
)


def render_dataco_forecast_performance_page() -> None:
    """Render DataCo Forecast Performance view."""
    render_dataco_disclaimer_banner()

    render_section_heading(
        title="DataCo Forecast Benchmark Evaluation",
        subtitle="Out-of-sample performance evaluation across identical test observations (Sept 1 – Oct 2, 2017).",
        icon="🎯",
    )

    try:
        metrics_dict = load_dataco_forecast_metrics()
        fc_df = load_dataco_forecast_comparison()
        catalog_df = load_dataco_catalog()
    except Exception as err:
        st.error(f"Error loading DataCo forecast artifacts: {err}")
        return

    meta = metrics_dict.get("metadata", {})
    pooled = metrics_dict.get("pooled_metrics", {})
    per_product = metrics_dict.get("per_product_metrics", {})

    # Protocol Notice Card
    st.info(
        f"📋 **Evaluation Protocol & Chronological Partitioning**  \n"
        f"• **Training Window**: {meta.get('train_window', ['2015-01-01', '2017-06-30'])[0]} to {meta.get('train_window', ['2015-01-01', '2017-06-30'])[1]} (912 calendar days)  \n"
        f"• **Validation Window**: {meta.get('validation_window', ['2017-07-01', '2017-08-31'])[0]} to {meta.get('validation_window', ['2017-07-01', '2017-08-31'])[1]} (62 days; used for pre-decision residual uncertainty)  \n"
        f"• **Test Evaluation Window**: {meta.get('test_window', ['2017-09-01', '2017-10-02'])[0]} to {meta.get('test_window', ['2017-09-01', '2017-10-02'])[1]} (32 days; 288 identical product-date observations)  \n"
        f"• **Benchmark Protocol**: **Rolling 1-day-ahead out-of-sample forecast**. On each test date $t$, the model predicts day $t+1$ using features constructed through day $t$.  \n"
        f"• **CRITICAL METHODOLOGICAL DISTINCTION**: The benchmark metrics below evaluate rolling 1-day-ahead predictions. They do **not** "
        f"represent fixed-origin 8-day recursive forecast accuracy, which is subject to autoregressive error accumulation across multi-day horizons. "
        f"Fixed-origin multi-step forecasting is used strictly as a forward decision-support scenario for inventory sizing.",
        icon="ℹ️",
    )

    # 1. Summary Scorecard & Metrics Comparison
    st.markdown("### 📊 Benchmark Model Comparison Scorecard")

    comparison_records = []
    for model_name, m in pooled.items():
        comparison_records.append({
            "Model Name": model_name,
            "Test Samples": m.get("sample_size", 0),
            "MAE (Units)": f"{m.get('mae', 0.0):.2f}",
            "RMSE (Units)": f"{m.get('rmse', 0.0):.2f}",
            "WAPE (%)": f"{m.get('wape', 0.0):.2f}%",
            "Macro MAE": f"{m.get('macro_mae', 0.0):.2f}",
            "Macro RMSE": f"{m.get('macro_rmse', 0.0):.2f}",
            "Macro WAPE (%)": f"{m.get('macro_wape', 0.0):.2f}%",
        })

    comp_df = pd.DataFrame(comparison_records)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # 2. Metric Definitions Card
    with st.expander("📖 Click to expand: Mathematical Error Metric Definitions", expanded=False):
        st.markdown(
            """
            - **Mean Absolute Error (MAE)**:
              $$\\text{MAE} = \\frac{1}{N} \\sum_{t=1}^{N} |y_t - \\hat{y}_t|$$
              Measures the average magnitude of absolute prediction errors in physical demand units. Robust to outliers.
            
            - **Root Mean Squared Error (RMSE)**:
              $$\\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{t=1}^{N} (y_t - \\hat{y}_t)^2}$$
              Penalizes large errors disproportionately due to squaring. Closely reflects the standard deviation of forecast residuals under zero bias.
            
            - **Weighted Absolute Percentage Error (WAPE)**:
              $$\\text{WAPE} = \\frac{\\sum_{t=1}^{N} |y_t - \\hat{y}_t|}{\\sum_{t=1}^{N} y_t} \\times 100\\%$$
              Volume-weighted percentage error. Unlike standard MAPE, WAPE handles zero-demand days safely without division-by-zero errors.
            """
        )

    # 3. Model Comparison Bar Chart
    fig_comp = plot_dataco_model_comparison(metrics_dict)
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    st.write("")

    # 4. Per-Product Performance Across Core 9
    st.markdown("### 📦 Per-Product Accuracy Across Core 9 Cohort")
    cohort_pids = meta.get("cohort", [])

    # Assemble per-product table
    name_lookup = dict(zip(catalog_df["product_id"].astype(str), catalog_df["product_name"]))
    prod_rows = []
    rf_prods = {r["product_id"]: r for r in per_product.get("Random Forest", [])}
    rm_prods = {r["product_id"]: r for r in per_product.get("Rolling Mean (7d)", [])}
    sn_prods = {r["product_id"]: r for r in per_product.get("Seasonal Naive", [])}

    for pid in cohort_pids:
        rf_m = rf_prods.get(pid, {})
        rm_m = rm_prods.get(pid, {})
        sn_m = sn_prods.get(pid, {})
        prod_rows.append({
            "Product ID": pid,
            "Product Name": name_lookup.get(pid, f"SKU {pid}"),
            "RF MAE": f"{rf_m.get('mae', 0.0):.2f}",
            "RF RMSE": f"{rf_m.get('rmse', 0.0):.2f}",
            "RF WAPE": f"{rf_m.get('wape', 0.0):.1f}%",
            "RM WAPE": f"{rm_m.get('wape', 0.0):.1f}%",
            "SN WAPE": f"{sn_m.get('wape', 0.0):.1f}%",
            "WAPE Improvement vs SN": f"{sn_m.get('wape', 0.0) - rf_m.get('wape', 0.0):+.1f}%",
        })

    p_table_df = pd.DataFrame(prod_rows)
    st.dataframe(p_table_df, use_container_width=True, hide_index=True)

    st.write("")

    # 5. Actual vs. Forecast Visualizer
    render_section_heading(
        title="Test Period Actual vs. Forecast Visualizer",
        subtitle="Compare ground-truth demand against rolling model forecasts for a selected Core 9 product.",
        icon="📈",
    )

    selected_pid = st.selectbox(
        "Select Core 9 Product to Visualize:",
        options=cohort_pids,
        format_func=lambda pid: f"SKU {pid} — {name_lookup.get(pid, 'Unknown')}",
        key="dataco_fc_eval_product_select"
    )

    p_fc_df = fc_df[fc_df["product_id"] == selected_pid].sort_values("date")
    p_name = name_lookup.get(selected_pid, f"SKU {selected_pid}")

    fig_actual = plot_dataco_product_forecast_vs_actual(p_fc_df, product_name=p_name)
    st.pyplot(fig_actual)
    plt.close(fig_actual)
