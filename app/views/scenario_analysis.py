"""Simulation & What-If Planning view for the Retail Sales Forecasting & Inventory Optimization System.

Enables interactive exploration of alternative supply chain policies:
1. Service-level targets and dynamic z-score calculation.
2. Replenishment lead time (L) and periodic review cycle (R).
3. Simulated operational inventory positions (On-Hand, On-Order, Backorders).
4. Strict forecast-horizon validation ensuring H >= L + R without zero-filling or data fabrication.
5. Side-by-side comparison against saved immutable baseline recommendations.
6. Empirical sensitivity analysis and beginner-friendly driver explanations.
"""

from typing import Any, Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.scenario_engine import (
    calculate_single_sku_scenario,
    compute_scenario_comparison,
    explain_scenario_dynamics,
    get_z_score_from_service_level,
    run_lead_time_sensitivity,
    run_service_level_sensitivity,
)
from app.data_loader import (
    get_inventory_recommendation_product_ids,
    get_single_product_inventory_details,
    load_product_forecast_comparison,
)
from app.components import (
    format_metric_value,
    get_recommendation_status_badge,
    render_section_heading,
)
from src.retail_forecasting.visualization.plots import (
    plot_scenario_sensitivity,
)


def render_methodological_notice() -> None:
    """Render prominent banner explaining baseline context and simulation boundaries."""
    st.info(
        "**Simulation & What-If Scenario Analysis Environment**  \n"
        "• **Planning Cutoff Date**: November 11, 2011  \n"
        "• **Saved Forecast Horizon**: November 12 – December 9, 2011 (28 days out-of-sample backtest)  \n"
        "• **Baseline Policy Parameters**: 95% Service Level ($z = 1.6449$), 7-Day Lead Time ($L$), 1-Day Review Period ($R$)  \n"
        "• **Operational Disclaimer**: Stock positions, holding costs, and order triggers reflect **simulated demonstration scenarios**, "
        "not real warehouse enterprise records. Scenario outputs are alternative calculations based on user-selected assumptions "
        "and do **not** constitute live enterprise purchasing instructions.",
        icon="ℹ️",
    )


def render_scenario_controls(
    available_skus: list,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Render product selector and interactive scenario parameter inputs.

    Parameters
    ----------
    available_skus : list
        List of product SKUs with inventory recommendations.

    Returns
    -------
    Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]
        Selected SKU, dictionary of configured scenario controls, and baseline record.
    """
    render_section_heading(
        title="1. Scenario Configuration Controls",
        subtitle="Select a product and configure alternative service-level targets, supplier lead times, and simulated stock levels.",
        icon="🎛️",
    )

    default_sku = "85123A" if "85123A" in available_skus else (available_skus[0] if available_skus else "")
    default_idx = available_skus.index(default_sku) if default_sku in available_skus else 0

    col_sku, col_desc = st.columns([1.2, 1.8])
    with col_sku:
        selected_sku = st.selectbox(
            "Select Product SKU for Scenario Analysis:",
            options=available_skus,
            index=default_idx,
            help="Choose any product evaluated in the inventory optimization module.",
            key="scenario_sku_select",
        )

    # Retrieve baseline recommendation for this product
    baseline_record = get_single_product_inventory_details(selected_sku) if selected_sku else None

    with col_desc:
        if baseline_record is not None and not baseline_record.get("_has_duplicates"):
            base_status = baseline_record.get("recommendation_status", "UNKNOWN")
            badge = get_recommendation_status_badge(base_status)
            st.caption(
                f"**Baseline Status for `{selected_sku}`**: {badge}  \n"
                f"Baseline Lead Time: {baseline_record.get('lead_time_days', 7)}d | "
                f"Review Period: {baseline_record.get('review_period_days', 1)}d | "
                f"Service Level: {baseline_record.get('service_level', 0.95) * 100:.0f}%  \n"
                f"Simulated On-Hand: {baseline_record.get('simulated_on_hand', 0.0):.0f} units | "
                f"Inventory Position: {baseline_record.get('inventory_position', 0.0):.0f} units"
            )
        else:
            st.caption(f"No valid non-duplicate baseline recommendation found for SKU `{selected_sku}`.")

    # Defaults from baseline if available
    def_sl = float(baseline_record.get("service_level", 0.95)) if baseline_record else 0.95
    def_lt = int(baseline_record.get("lead_time_days", 7)) if baseline_record else 7
    def_rp = int(baseline_record.get("review_period_days", 1)) if baseline_record else 1
    def_oh = float(baseline_record.get("simulated_on_hand", 0.0)) if baseline_record else 0.0
    def_oo = float(baseline_record.get("simulated_on_order", 0.0)) if baseline_record else 0.0
    def_bo = float(baseline_record.get("simulated_backorders", 0.0)) if baseline_record else 0.0

    st.markdown("##### Policy Assumptions & Simulated Stock Inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        service_level_pct = st.slider(
            "Target Cycle Service Level (%):",
            min_value=80.0,
            max_value=99.9,
            value=float(round(def_sl * 100.0, 1)),
            step=0.5,
            format="%.1f%%",
            help="Probability of not stocking out during a replenishment cycle. Dynamically calculates standard normal z-score.",
            key=f"scen_sl_{selected_sku}",
        )
        service_level = service_level_pct / 100.0
        z_curr = get_z_score_from_service_level(service_level)
        st.caption(f"Calculated standard normal score: **z = {z_curr:.4f}**")

    with col2:
        lead_time_days = st.number_input(
            "Supplier Lead Time L (Days):",
            min_value=1,
            max_value=28,
            value=def_lt,
            step=1,
            help="Days elapsed between placing a purchase order and receiving physical stock in warehouse.",
            key=f"scen_lt_{selected_sku}",
        )
        review_period_days = st.number_input(
            "Review Period R (Days):",
            min_value=1,
            max_value=14,
            value=def_rp,
            step=1,
            help="Periodic review interval in days between successive inventory evaluations.",
            key=f"scen_rp_{selected_sku}",
        )
        prot_period = lead_time_days + review_period_days
        st.caption(f"Protection Period: **P = L + R = {prot_period} days** (Must satisfy $P \\le 28$)")

    with col3:
        on_hand = st.number_input(
            "Simulated On-Hand Stock (Units):",
            min_value=0.0,
            value=def_oh,
            step=1.0,
            help="Simulated physically available stock in warehouse (non-negative).",
            key=f"scen_oh_{selected_sku}",
        )
        on_order = st.number_input(
            "Simulated On-Order In Transit (Units):",
            min_value=0.0,
            value=def_oo,
            step=1.0,
            help="Simulated purchase order units in transit from supplier (non-negative).",
            key=f"scen_oo_{selected_sku}",
        )
        backorders = st.number_input(
            "Simulated Backorders (Units):",
            min_value=0.0,
            value=def_bo,
            step=1.0,
            help="Simulated unfulfilled customer demand awaiting replenishment (non-negative).",
            key=f"scen_bo_{selected_sku}",
        )

    controls = {
        "service_level": service_level,
        "lead_time_days": int(lead_time_days),
        "review_period_days": int(review_period_days),
        "on_hand": float(on_hand),
        "on_order": float(on_order),
        "backorders": float(backorders),
    }

    return selected_sku, controls, baseline_record


def render_scenario_results(
    selected_sku: str,
    controls: Dict[str, Any],
    baseline_record: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute scenario calculation and render baseline-versus-scenario comparison.

    Parameters
    ----------
    selected_sku : str
        Target product SKU.
    controls : Dict[str, Any]
        Scenario control parameters.
    baseline_record : Optional[Dict[str, Any]]
        Saved baseline recommendation record.

    Returns
    -------
    Dict[str, Any]
        Scenario calculation output dictionary.
    """
    render_section_heading(
        title="2. Scenario Outputs & Baseline Comparison",
        subtitle="Direct side-by-side comparison between saved baseline policy and alternative scenario calculation.",
        icon="⚖️",
    )

    scenario_res = calculate_single_sku_scenario(
        product_id=selected_sku,
        service_level=controls["service_level"],
        lead_time_days=controls["lead_time_days"],
        review_period_days=controls["review_period_days"],
        on_hand=controls["on_hand"],
        on_order=controls["on_order"],
        backorders=controls["backorders"],
        baseline_record=baseline_record,
    )

    # Handle horizon and invalid scenario states
    if not scenario_res.get("is_valid"):
        status = scenario_res.get("recommendation_status", "UNKNOWN")
        reason = scenario_res.get("rejection_reason", "Invalid scenario.")
        st.warning(
            f"⚠️ **Scenario Cannot Be Calculated: `{status}`**  \n"
            f"{reason}  \n\n"
            "**Forecast Horizon Invariant**: Inventory optimization requires valid daily forecasts covering the full "
            "protection period ($P = L + R$). When the requested replenishment cycle exceeds the 28-day out-of-sample "
            "forecast horizon, calculations are safely marked unavailable without data extrapolation or zero-filling.",
            icon="⚠️",
        )
        return scenario_res

    # Compare against baseline
    comparison = compute_scenario_comparison(baseline_record, scenario_res)

    # Status Badges
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        base_badge = get_recommendation_status_badge(comparison["baseline_status"])
        st.markdown(f"**Baseline Recommendation**: {base_badge}")
    with col_stat2:
        scen_badge = get_recommendation_status_badge(comparison["scenario_status"])
        st.markdown(f"**Scenario Calculation**: {scen_badge}")

    # Primary KPI Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    d = comparison.get("deltas", {})

    delta_ss = f"{d['safety_stock']:+.1f} units" if (d.get("safety_stock") is not None and pd.notnull(d["safety_stock"])) else None
    delta_rop = f"{d['reorder_point']:+.1f} units" if (d.get("reorder_point") is not None and pd.notnull(d["reorder_point"])) else None
    delta_target = f"{d['target_stock']:+.1f} units" if (d.get("target_stock") is not None and pd.notnull(d["target_stock"])) else None
    delta_qty = (
        f"{int(round(float(d['suggested_order_qty']))):+d} units"
        if (d.get("suggested_order_qty") is not None and pd.notnull(d["suggested_order_qty"]))
        else None
    )

    c1.metric(
        label="Scenario Safety Stock (SS)",
        value=f"{scenario_res['safety_stock']:.1f} units",
        delta=delta_ss,
        help="Buffer stock to absorb demand uncertainty over lead time. Delta compares against baseline.",
    )
    c2.metric(
        label="Scenario Reorder Point (ROP)",
        value=f"{scenario_res['reorder_point']:.1f} units",
        delta=delta_rop,
        help="Threshold triggering replenishment: ROP = LTD + SS. Delta compares against baseline.",
    )
    c3.metric(
        label="Scenario Target Stock (S)",
        value=f"{scenario_res['target_stock']:.1f} units",
        delta=delta_target,
        help="Order-up-to target stocking level under periodic review: S = Protection Period Demand + SS.",
    )
    order_qty_val = scenario_res.get("suggested_order_qty")
    c4.metric(
        label="Suggested Order Qty (Q)",
        value=f"{int(round(float(order_qty_val))):,} units" if pd.notnull(order_qty_val) else "N/A",
        delta=delta_qty,
        delta_color="inverse" if (d.get("suggested_order_qty") is not None and d["suggested_order_qty"] > 0) else "normal",
        help="Recommended whole units to purchase under scenario: Q = max(0, ceil(S - IP)) if IP <= ROP else 0.",
    )

    # Detailed Side-by-Side Comparison Table
    st.markdown("##### Policy Metric Comparison Table")
    table_data = []
    for r in comparison["comparison_rows"]:
        b_val = format_metric_value(r["baseline"], format_type="float", decimals=2)
        s_val = format_metric_value(r["scenario"], format_type="float", decimals=2)
        if r["delta"] is not None:
            delta_str = f"{r['delta']:+.2f}"
        else:
            delta_str = "N/A"

        table_data.append({
            "Policy Metric": r["metric"],
            "Saved Baseline": b_val,
            "What-If Scenario": s_val,
            "Absolute Difference (Δ)": delta_str,
        })

    df_comp = pd.DataFrame(table_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.caption(
        "ℹ️ **Baseline Immutability Safeguard**: The saved baseline recommendation record is preserved without mutation. "
        "Absolute differences are calculated strictly when both baseline and scenario values are valid numbers."
    )

    return scenario_res


def render_sensitivity_section(
    selected_sku: str,
    controls: Dict[str, Any],
    baseline_record: Optional[Dict[str, Any]],
) -> None:
    """Render interactive sensitivity curves across varying service levels or lead times.

    Parameters
    ----------
    selected_sku : str
        Target product SKU.
    controls : Dict[str, Any]
        Scenario control parameters.
    baseline_record : Optional[Dict[str, Any]]
        Saved baseline recommendation record.
    """
    render_section_heading(
        title="3. Policy Parameter Sensitivity Analysis",
        subtitle="Empirical evaluation of how safety stock and reorder point respond across parameter sweeps.",
        icon="📈",
    )

    tab_sl, tab_lt = st.tabs(["🎯 Service Level Sensitivity", "⏱️ Lead Time Sensitivity"])

    with tab_sl:
        st.caption(
            "Evaluates how inventory buffers scale across target Cycle Service Levels (80% to 99%) "
            f"under the fixed scenario lead time ($L = {controls['lead_time_days']}$ days)."
        )
        df_sl = run_service_level_sensitivity(
            product_id=selected_sku,
            service_levels=[0.80, 0.85, 0.90, 0.95, 0.98, 0.99],
            lead_time_days=controls["lead_time_days"],
            review_period_days=controls["review_period_days"],
            on_hand=controls["on_hand"],
            on_order=controls["on_order"],
            backorders=controls["backorders"],
            baseline_record=baseline_record,
        )

        fig_sl = plot_scenario_sensitivity(df_sl, parameter_col="service_level", product_id=selected_sku)
        st.pyplot(fig_sl, use_container_width=True)
        plt.close(fig_sl)

        # Formatted summary table
        st.dataframe(
            df_sl[["service_level", "z_score", "safety_stock", "reorder_point", "target_stock", "suggested_order_qty", "status"]].rename(
                columns={
                    "service_level": "Service Level",
                    "z_score": "z-Score",
                    "safety_stock": "Safety Stock (SS)",
                    "reorder_point": "Reorder Point (ROP)",
                    "target_stock": "Target Stock (S)",
                    "suggested_order_qty": "Order Qty (Q)",
                    "status": "Recommendation Status",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_lt:
        st.caption(
            "Evaluates how inventory buffers scale across supplier lead times (3 to 28 days) "
            f"under the configured scenario service level ({controls['service_level'] * 100:.1f}% CSL).  \n"
            "Scenarios where $L + R > 28$ are explicitly marked unavailable rather than fabricated."
        )
        df_lt = run_lead_time_sensitivity(
            product_id=selected_sku,
            lead_times=[3, 5, 7, 10, 14, 21, 28],
            service_level=controls["service_level"],
            review_period_days=controls["review_period_days"],
            on_hand=controls["on_hand"],
            on_order=controls["on_order"],
            backorders=controls["backorders"],
            baseline_record=baseline_record,
        )

        fig_lt = plot_scenario_sensitivity(df_lt, parameter_col="lead_time_days", product_id=selected_sku)
        st.pyplot(fig_lt, use_container_width=True)
        plt.close(fig_lt)

        st.dataframe(
            df_lt[["lead_time_days", "protection_period_days", "expected_lead_time_demand", "safety_stock", "reorder_point", "suggested_order_qty", "status"]].rename(
                columns={
                    "lead_time_days": "Lead Time (Days)",
                    "protection_period_days": "Protection Period (P)",
                    "expected_lead_time_demand": "Lead-Time Demand",
                    "safety_stock": "Safety Stock (SS)",
                    "reorder_point": "Reorder Point (ROP)",
                    "suggested_order_qty": "Order Qty (Q)",
                    "status": "Status",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_dynamics_explanation(
    baseline_record: Optional[Dict[str, Any]],
    scenario_res: Dict[str, Any],
) -> None:
    """Render beginner-friendly explanations of scenario drivers and limitations.

    Parameters
    ----------
    baseline_record : Optional[Dict[str, Any]]
        Saved baseline recommendation record.
    scenario_res : Dict[str, Any]
        Scenario calculation output dictionary.
    """
    render_section_heading(
        title="4. Key Drivers & Methodological Governance",
        subtitle="Understanding how service-level choices, lead times, and inventory buffers interact.",
        icon="💡",
    )

    comparison = compute_scenario_comparison(baseline_record, scenario_res)
    explanations = explain_scenario_dynamics(comparison)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 🎯 Service-Level Target")
        st.write(explanations["service_level_driver"])
        st.caption(
            "Higher service-level targets increase the normal quantile $z$. When forecast error standard deviation "
            "is positive, safety stock increases non-linearly (steepening sharply above 95% service level)."
        )

    with col2:
        st.markdown("##### ⏱️ Replenishment Lead Time")
        st.write(explanations["lead_time_driver"])
        st.caption(
            "Lead time scales expected cumulative demand ($LTD = \sum_{t=1}^L \hat{d}_t$) and uncertainty exposure "
            "($SS = z \cdot \sigma \cdot \sqrt{L}$). Longer lead times necessitate larger buffer stocks."
        )

    with col3:
        st.markdown("##### 📦 Inventory Position & Reorder")
        st.write(explanations["inventory_driver"])
        st.write(explanations["trigger_driver"])
        st.caption(
            "The order quantity formula ($Q = \max(0, \lceil S - IP \\rceil)$) replenishes up to target stock $S$ "
            "strictly when net inventory position falls to or below ROP."
        )

    st.divider()
    st.markdown("##### ⚠️ Governance & Scope Limitations")
    st.caption(
        "1. **No Profit or Cost Optimization Claims**: This module evaluates physical stockout protection formulas; "
        "it does not solve an unconstrained enterprise profit optimization problem.  \n"
        "2. **Simulated Environment**: All on-hand, on-order, and backorder inputs reflect simulated demonstration states.  \n"
        "3. **Forecast Horizon Invariant**: Predictions are derived from the 28-day out-of-sample backtest window "
        "(November 12 – December 9, 2011). Scenarios exceeding this horizon are marked unavailable rather than extrapolated."
    )


def render_scenario_analysis_page() -> None:
    """Render the full Simulation & What-If Planning dashboard page."""
    render_section_heading(
        title="Simulation & What-If Planning",
        subtitle="Interactive decision support environment for evaluating alternative inventory policies under simulated supply chain conditions.",
        icon="🧪",
    )

    try:
        render_methodological_notice()
        st.write("")

        available_skus = get_inventory_recommendation_product_ids()
        if not available_skus:
            st.warning("No product recommendation records available in the inventory artifact.")
            return

        # 1. Scenario controls
        selected_sku, controls, baseline_record = render_scenario_controls(available_skus)
        if not selected_sku:
            return

        st.divider()

        # 2. Scenario results and baseline comparison
        scenario_res = render_scenario_results(selected_sku, controls, baseline_record)
        st.divider()

        # 3. Sensitivity analysis (only if scenario inputs are valid)
        if scenario_res.get("is_valid"):
            render_sensitivity_section(selected_sku, controls, baseline_record)
            st.divider()

        # 4. Key drivers and governance
        render_dynamics_explanation(baseline_record, scenario_res)

    except Exception as err:
        st.error(f"Failed to render Simulation & What-If Planning: {err}")
