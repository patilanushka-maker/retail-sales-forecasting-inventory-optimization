"""Matplotlib plotting utilities for retail sales forecasting and inventory.

Provides consistent styling and figures for:
- Historical portfolio daily demand trends.
- Top products ranked by recorded demand.
- Individual SKU daily demand time series preserving zero-demand days.
- Historical vs. forecasted sales time-series plots.
- Model comparison charts (Seasonal Naive vs. Random Forest).
- Inventory trajectory and stockout simulation plots.
"""

from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def _apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply consistent slate dark-theme styling matching Streamlit config."""
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.spines["bottom"].set_color("#475569")
    ax.spines["left"].set_color("#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")
    ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")


def plot_portfolio_daily_demand(summary_df: pd.DataFrame) -> plt.Figure:
    """Plot aggregate recorded daily demand trend across all products over time.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Aggregated daily demand DataFrame with 'date' and 'total_demand'.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _apply_dark_theme(fig, ax)

    dates = pd.to_datetime(summary_df["date"])
    demand = summary_df["total_demand"]

    ax.plot(
        dates,
        demand,
        color="#3B82F6",
        linewidth=1.8,
        label="Total Daily Recorded Demand",
    )

    # 7-day rolling average for trend smoothing
    rolling_7 = demand.rolling(7, min_periods=1).mean()
    ax.plot(
        dates,
        rolling_7,
        color="#F59E0B",
        linewidth=2.0,
        linestyle="--",
        label="7-Day Rolling Average",
    )

    ax.set_title("Historical Portfolio Daily Demand Trend (Dec 2010 – Dec 2011)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Recorded Demand (Units)", fontsize=10, labelpad=8)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper left",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_top_products_bar(top_df: pd.DataFrame) -> plt.Figure:
    """Plot horizontal bar chart of top products ranked by total recorded demand units.

    Parameters
    ----------
    top_df : pd.DataFrame
        DataFrame with 'product_id' and 'total_demand', sorted descending.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _apply_dark_theme(fig, ax)

    # Invert so highest product is at top
    df_sorted = top_df.iloc[::-1].copy()
    y_pos = np.arange(len(df_sorted))
    product_labels = [str(pid) for pid in df_sorted["product_id"]]
    values = df_sorted["total_demand"].values

    bars = ax.barh(
        y_pos,
        values,
        color="#6366F1",
        edgecolor="#818CF8",
        height=0.65,
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(product_labels, fontsize=9, fontweight="medium")
    ax.set_title("Top 10 Products by Total Recorded Demand Units", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Recorded Demand (Units)", fontsize=10, labelpad=8)
    ax.set_ylabel("Product ID (SKU)", fontsize=10, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # Value callouts on each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max(values) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}",
            va="center",
            ha="left",
            color="#E2E8F0",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    return fig


def plot_product_daily_demand(product_df: pd.DataFrame, product_id: str) -> plt.Figure:
    """Plot daily recorded demand time series for a single SKU preserving zero-demand days.

    Parameters
    ----------
    product_df : pd.DataFrame
        Daily demand series with 'date' and 'demand'.
    product_id : str
        Product identifier for plot title.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _apply_dark_theme(fig, ax)

    dates = pd.to_datetime(product_df["date"])
    demand = product_df["demand"]

    # Solid baseline connection
    ax.plot(
        dates,
        demand,
        color="#38BDF8",
        linewidth=1.2,
        alpha=0.8,
        label="Recorded Daily Demand",
    )

    # Highlight positive-demand days with scatter markers
    pos_mask = demand > 0
    if pos_mask.any():
        ax.scatter(
            dates[pos_mask],
            demand[pos_mask],
            color="#F43F5E",
            s=22,
            zorder=4,
            label="Transactions (Units > 0)",
        )

    ax.set_title(f"Recorded Daily Demand Series for SKU: {product_id}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Recorded Demand (Units)", fontsize=10, labelpad=8)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_forecast_vs_actual(
    dates: pd.Series,
    y_actual: pd.Series,
    y_pred: pd.Series,
    title: str = "Forecast vs. Actual Demand",
    baseline_pred: Optional[pd.Series] = None
) -> plt.Figure:
    """Generate time-series overlay of actual sales against forecasts.

    Parameters
    ----------
    dates : pd.Series
        Date index or column.
    y_actual : pd.Series
        Observed ground-truth demand.
    y_pred : pd.Series
        Model predicted demand.
    title : str, default 'Forecast vs. Actual Demand'
        Chart title.
    baseline_pred : Optional[pd.Series], optional
        Optional baseline forecast series for comparison.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _apply_dark_theme(fig, ax)

    dates_dt = pd.to_datetime(dates)

    # Actual demand series (solid blue line with points)
    ax.plot(
        dates_dt,
        y_actual,
        color="#38BDF8",
        linewidth=1.8,
        marker="o",
        markersize=4.5,
        label="Actual Recorded Demand",
        zorder=3,
    )

    # Random forest predictions (dashed emerald green line)
    ax.plot(
        dates_dt,
        y_pred,
        color="#10B981",
        linewidth=2.0,
        linestyle="--",
        label="Random Forest Prediction",
        zorder=4,
    )

    # Seasonal naive baseline predictions (dotted amber line)
    if baseline_pred is not None:
        ax.plot(
            dates_dt,
            baseline_pred,
            color="#F59E0B",
            linewidth=1.6,
            linestyle=":",
            label="Seasonal Naive Baseline",
            zorder=2,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Evaluation Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Demand (Units)", fontsize=10, labelpad=8)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_inventory_trajectory(
    dates: pd.Series,
    inventory_levels: pd.Series,
    reorder_points: pd.Series,
    safety_stocks: pd.Series,
    title: str = "Inventory Position & Reorder Thresholds"
) -> plt.Figure:
    """Plot simulated inventory position over time with ROP and safety stock thresholds.

    Parameters
    ----------
    dates : pd.Series
        Timeline.
    inventory_levels : pd.Series
        Daily simulated on-hand inventory.
    reorder_points : pd.Series
        Reorder point threshold level.
    safety_stocks : pd.Series
        Safety stock baseline level.
    title : str
        Chart title.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    raise NotImplementedError("TODO: Implement inventory trajectory plot in Page 5.")


def plot_scenario_sensitivity(
    df_sensitivity: pd.DataFrame,
    parameter_col: str = "service_level",
    product_id: Optional[str] = None,
) -> plt.Figure:
    """Plot inventory policy thresholds across sensitivity parameter values.

    Parameters
    ----------
    df_sensitivity : pd.DataFrame
        Table generated by run_service_level_sensitivity or run_lead_time_sensitivity.
    parameter_col : str, default 'service_level'
        Sensitivity parameter: 'service_level' or 'lead_time_days'.
    product_id : Optional[str], optional
        Product identifier for chart title.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _apply_dark_theme(fig, ax)

    sku_label = f"SKU: {product_id}" if product_id else "Selected SKU"
    valid_df = df_sensitivity[df_sensitivity["is_valid"] == True].copy() if "is_valid" in df_sensitivity.columns else df_sensitivity.copy()

    if valid_df.empty:
        ax.text(
            0.5,
            0.5,
            "No valid scenario outputs within available forecast horizon.",
            horizontalalignment="center",
            verticalalignment="center",
            color="#F87171",
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_title(f"Sensitivity Analysis ({sku_label})", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig

    if parameter_col == "service_level":
        x_labels = [f"{v * 100:.1f}%" if v < 1.0 else f"{v:.1f}%" for v in valid_df["service_level"]]
        x_indices = np.arange(len(x_labels))

        ax.plot(
            x_indices,
            valid_df["safety_stock"],
            marker="o",
            color="#38BDF8",
            linewidth=2.0,
            markersize=6,
            label="Safety Stock (SS)",
            zorder=3,
        )
        ax.plot(
            x_indices,
            valid_df["reorder_point"],
            marker="s",
            color="#F59E0B",
            linewidth=2.0,
            markersize=6,
            label="Reorder Point (ROP)",
            zorder=4,
        )
        if "target_stock" in valid_df.columns:
            ax.plot(
                x_indices,
                valid_df["target_stock"],
                marker="^",
                color="#10B981",
                linewidth=2.0,
                markersize=6,
                label="Target Stock (S)",
                zorder=2,
            )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Target Cycle Service Level (CSL)", fontsize=10, labelpad=8)
        ax.set_title(f"Sensitivity: Inventory Buffers vs. Target Service Level ({sku_label})", fontsize=12, fontweight="bold", pad=12)

    elif parameter_col == "lead_time_days":
        x_labels = [f"{int(v)}d" for v in valid_df["lead_time_days"]]
        x_indices = np.arange(len(x_labels))

        if "expected_lead_time_demand" in valid_df.columns:
            ax.plot(
                x_indices,
                valid_df["expected_lead_time_demand"],
                marker="d",
                color="#A855F7",
                linewidth=1.8,
                markersize=6,
                label="Lead-Time Demand (LTD)",
                zorder=2,
            )
        ax.plot(
            x_indices,
            valid_df["safety_stock"],
            marker="o",
            color="#38BDF8",
            linewidth=2.0,
            markersize=6,
            label="Safety Stock (SS)",
            zorder=3,
        )
        ax.plot(
            x_indices,
            valid_df["reorder_point"],
            marker="s",
            color="#F59E0B",
            linewidth=2.0,
            markersize=6,
            label="Reorder Point (ROP)",
            zorder=4,
        )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Replenishment Lead Time (Days)", fontsize=10, labelpad=8)
        ax.set_title(f"Sensitivity: Inventory Buffers vs. Supplier Lead Time ({sku_label})", fontsize=12, fontweight="bold", pad=12)
    else:
        x_indices = np.arange(len(valid_df))
        ax.plot(x_indices, valid_df["reorder_point"], marker="o", color="#F59E0B", label="Reorder Point")
        ax.set_xlabel(parameter_col, fontsize=10, labelpad=8)

    ax.set_ylabel("Quantity (Units)", fontsize=10, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper left",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


# =============================================================================
# DataCo Smart Supply Chain Plotting Utilities
# =============================================================================

def plot_dataco_portfolio_daily_demand(summary_df: pd.DataFrame) -> plt.Figure:
    """Plot aggregate daily gross placed demand trend across all DataCo products."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _apply_dark_theme(fig, ax)

    dates = pd.to_datetime(summary_df["date"])
    demand = summary_df["total_demand"]

    ax.plot(
        dates,
        demand,
        color="#38BDF8",
        linewidth=1.2,
        alpha=0.6,
        label="Daily Gross Placed Demand",
    )

    # 30-day rolling average for monthly trend smoothing
    rolling_30 = demand.rolling(30, min_periods=7).mean()
    ax.plot(
        dates,
        rolling_30,
        color="#F59E0B",
        linewidth=2.2,
        label="30-Day Rolling Mean Trend",
    )

    ax.set_title("DataCo Historical Daily Gross Placed Demand (2015 – 2018)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Order Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Gross Placed Demand (Units)", fontsize=10, labelpad=8)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_dataco_department_demand(dept_df: pd.DataFrame) -> plt.Figure:
    """Plot horizontal bar chart of gross demand by department in DataCo."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _apply_dark_theme(fig, ax)

    df_sorted = dept_df.sort_values(by="total_demand", ascending=True)
    y_pos = np.arange(len(df_sorted))

    bars = ax.barh(
        y_pos,
        df_sorted["total_demand"],
        color="#38BDF8",
        edgecolor="#0284C7",
        alpha=0.85,
        height=0.6,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["department_name"], fontsize=9)
    ax.set_xlabel("Total Gross Demand (Units)", fontsize=10, labelpad=8)
    ax.set_title("Gross Placed Demand by Department (Catalog Lifetime)", fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#E2E8F0",
        )

    fig.tight_layout()
    return fig


def plot_dataco_fixed_horizon_forecast(
    horizon_df: pd.DataFrame,
    ltd: Optional[float] = None,
    rop: Optional[float] = None,
    ss: Optional[float] = None,
    target_stock: Optional[float] = None,
    product_name: Optional[str] = None
) -> plt.Figure:
    """Plot the fixed-origin multi-step recursive forecast horizon with lead-time demarcations."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _apply_dark_theme(fig, ax)

    if horizon_df.empty:
        ax.text(0.5, 0.5, "No Forecast Data Available", ha="center", va="center", color="#94A3B8", transform=ax.transAxes)
        return fig

    df_sorted = horizon_df.sort_values(by="forecast_horizon_step").copy()
    steps = df_sorted["forecast_horizon_step"].values
    dates = df_sorted["forecast_date"].values
    preds = df_sorted["predicted_demand"].values

    x_labels = [f"Step {s}\n({str(d)[5:] if len(str(d)) >= 5 else d})" for s, d in zip(steps, dates)]

    # Plot bars for daily forecasts
    colors = ["#38BDF8" if s <= 7 else "#F59E0B" for s in steps]
    bars = ax.bar(
        steps,
        preds,
        color=colors,
        edgecolor="#0284C7",
        alpha=0.85,
        width=0.6,
        label="Predicted Daily Demand",
    )

    # Lead time boundary line between Step 7 and Step 8 (if >= 8 steps)
    if len(steps) >= 8:
        ax.axvline(x=7.5, color="#EF4444", linestyle="--", linewidth=1.5, label="Lead Time Boundary (L=7d | R=1d)")

    # Value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(1.0, height * 0.02),
            f"{height:.1f}",
            va="bottom",
            ha="center",
            fontsize=8.5,
            color="#E2E8F0",
        )

    origin_str = ""
    if "forecast_origin" in df_sorted.columns and not df_sorted["forecast_origin"].empty:
        origin_val = df_sorted["forecast_origin"].iloc[0]
        origin_str = f" (Origin: {origin_val})"

    title_name = f" — {product_name}" if product_name else ""
    ax.set_title(f"Fixed-Origin {len(steps)}-Day Recursive Forecast Horizon{origin_str}{title_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(steps)
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.set_ylabel("Predicted Demand (Units)", fontsize=10, labelpad=8)
    max_pred = max(preds) if len(preds) > 0 and max(preds) > 0 else 10.0
    ax.set_ylim(0, max_pred * 1.25)

    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper right",
        fontsize=8.5,
    )
    fig.tight_layout()
    return fig


def plot_dataco_model_comparison(metrics_dict: dict) -> plt.Figure:
    """Plot grouped bar chart comparing benchmark metrics across models."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _apply_dark_theme(fig, ax)

    pooled = metrics_dict.get("pooled_metrics", {})
    models = list(pooled.keys())
    if not models:
        return fig

    maes = [pooled[m].get("mae", 0.0) for m in models]
    rmses = [pooled[m].get("rmse", 0.0) for m in models]
    wapes = [pooled[m].get("wape", 0.0) for m in models]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, maes, width, label="MAE (Units)", color="#38BDF8", alpha=0.85)
    ax.bar(x, rmses, width, label="RMSE (Units)", color="#818CF8", alpha=0.85)
    ax.bar(x + width, wapes, width, label="WAPE (%)", color="#F59E0B", alpha=0.85)

    ax.set_ylabel("Metric Value", fontsize=10, labelpad=8)
    ax.set_title("DataCo Out-of-Sample Benchmark Error Comparison (Test Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, fontweight="semibold")

    # Add data labels
    for i in range(len(models)):
        ax.text(x[i] - width, maes[i] + 0.3, f"{maes[i]:.2f}", ha="center", va="bottom", fontsize=8, color="#E2E8F0")
        ax.text(x[i], rmses[i] + 0.3, f"{rmses[i]:.2f}", ha="center", va="bottom", fontsize=8, color="#E2E8F0")
        ax.text(x[i] + width, wapes[i] + 0.3, f"{wapes[i]:.1f}%", ha="center", va="bottom", fontsize=8, color="#E2E8F0")

    ax.set_ylim(0, max(wapes) * 1.2)
    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_dataco_product_forecast_vs_actual(
    fc_product_df: pd.DataFrame,
    product_name: Optional[str] = None
) -> plt.Figure:
    """Plot actual demand vs benchmark models for one DataCo product across the test period."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    _apply_dark_theme(fig, ax)

    if fc_product_df.empty:
        ax.text(0.5, 0.5, "No Test Period Forecast Data Available", ha="center", va="center", color="#94A3B8", transform=ax.transAxes)
        return fig

    df_sorted = fc_product_df.sort_values(by="date")
    dates = pd.to_datetime(df_sorted["date"])

    if "actual_demand" in df_sorted.columns:
        ax.plot(dates, df_sorted["actual_demand"], color="#E2E8F0", linewidth=1.5, marker="o", markersize=4, label="Actual Demand", zorder=3)
    if "random_forest_pred" in df_sorted.columns:
        ax.plot(dates, df_sorted["random_forest_pred"], color="#38BDF8", linewidth=2.0, linestyle="-", label="Random Forest (Benchmark C)", zorder=4)
    if "rolling_mean_pred" in df_sorted.columns:
        ax.plot(dates, df_sorted["rolling_mean_pred"], color="#F59E0B", linewidth=1.5, linestyle="--", label="Rolling Mean 7d (Baseline B)", zorder=2)
    if "seasonal_naive_pred" in df_sorted.columns:
        ax.plot(dates, df_sorted["seasonal_naive_pred"], color="#A855F7", linewidth=1.2, linestyle=":", label="Seasonal Naive (Baseline A)", zorder=1)

    title_name = f" — {product_name}" if product_name else ""
    ax.set_title(f"Out-of-Sample Rolling Test Forecast vs. Actual (Sept 1 – Oct 2, 2017){title_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Demand Units", fontsize=10, labelpad=8)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax.legend(
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#E2E8F0",
        loc="upper right",
        fontsize=8.5,
    )
    fig.tight_layout()
    return fig

