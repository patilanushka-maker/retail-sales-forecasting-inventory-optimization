"""Reusable UI components and visual standards for the Streamlit dashboard application.

Provides modular, accessible widgets for KPI metric cards, explanatory alerts,
disclaimers, section headers, and recommendation tables.

All components adhere strictly to the repository's methodological safeguards:
1. Retrospective Demonstration Warning: Highlights that forecasts evaluate a
   historical 28-day backtest period (Nov 12 - Dec 9, 2011) with a Nov 11, 2011 cutoff.
2. Simulated Inventory Warning: Disclaims that on-hand stock levels and cost
   parameters are simulated operational inputs rather than real warehouse records.
3. Transparent Unavailable Indicators: Clearly explains why items with lead time +
   review period exceeding horizon (H < L + R) have undefined recommendations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Pure Helper Functions (Unit-testable without Streamlit context)
# =============================================================================

def format_metric_value(
    val: Union[int, float, str, None],
    format_type: str = "float",
    decimals: int = 2,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Format a metric value into a clean, display-ready string.

    Parameters
    ----------
    val : int, float, str, or None
        Raw metric value to format.
    format_type : str, default 'float'
        Format style: 'float', 'integer', 'percent', 'currency', or 'string'.
    decimals : int, default 2
        Decimal places for floating point values.
    prefix : str, default ''
        Prefix symbol (e.g., '£').
    suffix : str, default ''
        Suffix symbol (e.g., '%').

    Returns
    -------
    str
        Formatted string, or 'N/A' if value is None or NaN.
    """
    if val is None or pd.isna(val):
        return "N/A"

    try:
        num = float(val)
    except (ValueError, TypeError):
        return str(val)

    if format_type == "integer":
        return f"{prefix}{int(round(num)):,}{suffix}"
    elif format_type == "percent":
        # If value is between 0 and 1, convert to percentage e.g. 0.14 -> 14.0%
        # If value is already > 1 (e.g. 97.2), format directly
        pct = num * 100 if abs(num) <= 1.0 and num != 0 else num
        return f"{prefix}{pct:.{decimals}f}%{suffix}"
    elif format_type == "currency":
        curr_prefix = prefix or "£"
        return f"{curr_prefix}{num:,.{decimals}f}{suffix}"
    elif format_type == "float":
        return f"{prefix}{num:,.{decimals}f}{suffix}"
    else:
        return f"{prefix}{val}{suffix}"


def get_recommendation_status_badge(status: str) -> str:
    """Translate raw recommendation status codes into human-readable labels.

    Parameters
    ----------
    status : str
        Recommendation status code.

    Returns
    -------
    str
        Human-friendly label with descriptive context.
    """
    labels = {
        "REORDER_NOW": "🚨 Reorder Required (Simulation)",
        "REORDER_TRIGGERED": "🚨 Reorder Required (Simulation)",
        "STOCK_ADEQUATE": "✅ Sufficient Stock",
        "SUFFICIENT_STOCK": "✅ Sufficient Stock",
        "ADEQUATELY_STOCKED": "✅ Sufficient Stock (Simulation)",
        "UNAVAILABLE": "⚠️ Recommendation Unavailable",
        "UNAVAILABLE_INSUFFICIENT_HORIZON": "⚠️ Recommendation Unavailable (H < L + R)",
        "UNAVAILABLE_INVALID_FORECAST": "⚠️ Recommendation Unavailable (Invalid Forecast)",
        "UNAVAILABLE_MISSING_UNCERTAINTY": "⚠️ Recommendation Unavailable (Missing Uncertainty)",
        "ERROR": "❌ Error in Recommendation",
    }
    return labels.get(status, f"ℹ️ {status}")


def render_dataco_disclaimer_banner() -> None:
    """Render prominent methodological disclaimer banner for DataCo Smart Supply Chain."""
    st.warning(
        "⚠️ **DataCo Smart Supply Chain — Operational & Demand Scope**  \n"
        "• **Demand Definition**: Gross placed retail demand across customer orders (excluding cancellations and suspected fraud). "
        "Demand reflects customer orders placed, not fulfilled sales or historical stockout loss.  \n"
        "• **Simulated Inventory Inputs**: Physical warehouse stock-on-hand, inbound supplier replenishment lead time ($L=7\\text{ days}$), "
        "and procurement costs are **simulated demonstration parameters**. DataCo contains customer transaction lines and outbound shipping days, "
        "not real warehouse bin counts or supplier purchase orders.  \n"
        "• **Decision-Support Context**: All recommendations are policy simulation benchmarks, not live enterprise purchasing instructions.",
        icon="⚠️",
    )



def format_inventory_table(
    df: pd.DataFrame,
    columns_to_show: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Format an inventory recommendation dataframe for clean dashboard display.

    Parameters
    ----------
    df : pd.DataFrame
        Raw recommendations or replenishment orders dataframe.
    columns_to_show : Optional[List[str]], optional
        Subset of columns to display. If None, uses a curated set.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with readable column headers and rounded numbers.
    """
    display_df = df.copy()

    if "recommendation_status" in display_df.columns:
        display_df["status_display"] = display_df["recommendation_status"].apply(
            get_recommendation_status_badge
        )

    # Standard column mapping
    rename_map = {
        "product_id": "Product ID",
        "status_display": "Status",
        "simulated_on_hand": "Simulated On-Hand Stock",
        "current_stock": "Simulated On-Hand Stock",
        "safety_stock": "Safety Stock (Units)",
        "reorder_point": "Reorder Point (ROP)",
        "suggested_order_qty": "Suggested Order Qty (Simulation)",
        "order_quantity": "Suggested Order Qty (Simulation)",
        "expected_lead_time_demand": "Expected Demand (Lead Time)",
        "expected_protection_period_demand": "Expected Demand (Lead+Review)",
        "total_expected_demand": "Expected Demand (Lead+Review)",
        "inventory_position": "Inventory Position",
        "forecast_error_std": "Forecast Error Std (RMSE)",
        "is_uncertainty_fallback": "Pooled Fallback Used",
        "economic_order_quantity": "EOQ (Units)",
        "estimated_order_cost": "Estimated Order Cost (£)",
    }

    if columns_to_show:
        cols = [c for c in columns_to_show if c in display_df.columns]
        display_df = display_df[cols]

    display_df = display_df.rename(columns=rename_map)
    return display_df


# =============================================================================
# Streamlit UI Components
# =============================================================================

def render_retrospective_warning() -> None:
    """Render a persistent banner clarifying the historical backtest nature of forecasts."""
    st.info(
        "**Historical Backtest Demonstration (Retrospective Evaluation)**  \n"
        "Model forecasts displayed in this dashboard evaluate a historical 28-day evaluation window "
        "(November 12 – December 9, 2011) evaluated strictly out-of-sample as of November 11, 2011. "
        "These predictions demonstrate retrospective backtesting performance and are not live future forecasts.",
        icon="ℹ️",
    )


def render_simulation_warning() -> None:
    """Render a persistent banner clarifying that operational inputs are simulated scenarios."""
    st.caption(
        "**Operational Scenario Disclaimer**: On-hand stock levels, supplier lead times (e.g. 7 days), "
        "periodic review cycles (e.g. 7 days), and cost inputs (holding cost £1.50/unit/yr, order cost £25.00/order) "
        "are simulated operational parameters designed to demonstrate inventory optimization logic, "
        "not real warehouse enterprise records."
    )


def render_unavailable_recommendation_alert(count: int, horizon_days: int = 28) -> None:
    """Render an informational alert explaining legitimate null/unavailable recommendations.

    Parameters
    ----------
    count : int
        Number of items with unavailable recommendations.
    horizon_days : int, default 28
        Available forecast horizon in days.
    """
    st.warning(
        f"**{count:,} items have status: Recommendation Unavailable**  \n"
        f"Inventory optimization requires a forecast horizon covering at least Lead Time + Review Period "
        f"($H \\ge L + R$). For items whose replenishment cycle exceeds the {horizon_days}-day forecast horizon, "
        "safety stock, reorder points, and suggested order quantities are deliberately undefined (null) "
        "to prevent underestimating inventory requirements and risking stockouts.",
        icon="⚠️",
    )


def render_section_heading(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Render a consistent section header with optional subtitle caption.

    Parameters
    ----------
    title : str
        Section title.
    subtitle : Optional[str], optional
        Explanatory caption or context.
    icon : Optional[str], optional
        Optional leading emoji or icon.
    """
    heading = f"{icon} {title}" if icon else title
    st.subheader(heading)
    if subtitle:
        st.caption(subtitle)


def render_kpi_metrics(
    metrics: Dict[str, Dict[str, Any]],
    columns: int = 4,
) -> None:
    """Render summary metric cards across responsive columns.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, Any]]
        Dictionary where each entry specifies:
        - 'label': Human-friendly display label (e.g. 'Historical evaluation WAPE')
        - 'value': Raw or formatted metric value
        - 'delta': Optional delta comparison (e.g. '-2.4% vs Baseline')
        - 'help': Tooltip description explaining what the metric represents
        - 'format_type': Optional format type ('percent', 'integer', 'currency', 'float')
    columns : int, default 4
        Number of metric columns per row.
    """
    items = list(metrics.items())
    for i in range(0, len(items), columns):
        row_items = items[i : i + columns]
        cols = st.columns(len(row_items))
        for col, (metric_key, metric_info) in zip(cols, row_items):
            label = metric_info.get("label", metric_key)
            raw_val = metric_info.get("value")
            fmt = metric_info.get("format_type", "string")
            decimals = metric_info.get("decimals", 2)
            prefix = metric_info.get("prefix", "")
            suffix = metric_info.get("suffix", "")

            if fmt != "string":
                disp_val = format_metric_value(
                    raw_val,
                    format_type=fmt,
                    decimals=decimals,
                    prefix=prefix,
                    suffix=suffix,
                )
            else:
                disp_val = str(raw_val)

            delta = metric_info.get("delta")
            help_text = metric_info.get("help")

            col.metric(
                label=label,
                value=disp_val,
                delta=delta,
                help=help_text,
            )


def render_recommendation_table(
    df: pd.DataFrame,
    max_rows: int = 50,
) -> None:
    """Render a formatted inventory reorder recommendation table.

    Parameters
    ----------
    df : pd.DataFrame
        Table of items requiring replenishment or full recommendations.
    max_rows : int, default 50
        Maximum number of rows to render directly.
    """
    if df.empty:
        st.info("No records to display.")
        return

    formatted_df = format_inventory_table(df.head(max_rows))
    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
    )
    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows} of {len(df):,} total records.")
