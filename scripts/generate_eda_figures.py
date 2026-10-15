"""EDA figure generation script for retail demand time-series.

Generates 7 standard exploratory figures and saves them to outputs/figures/:
1. Total daily demand across all products over time.
2. Distribution of product-level daily demand (linear and log scale).
3. Daily demand for representative products selected via reproducible percentile rules.
4. Proportion of zero-demand product-days over time.
5. Product history length and catalog sparsity distribution.
6. Day of week demand seasonality.
7. Monthly demand patterns with documented 1-year limitation note.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_all_eda_figures(
    processed_data_path: str = "data/processed/daily_product_demand.parquet",
    output_dir: str = "outputs/figures"
) -> None:
    """Generate and save all 7 EDA visualizations."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading processed daily demand from: {processed_data_path}")
    df = pd.read_parquet(processed_data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Configure plot styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10

    # -------------------------------------------------------------
    # Figure 1: Total daily demand across all products over time
    # -------------------------------------------------------------
    daily_total = df.groupby("date")["demand"].sum().reset_index()
    # 7-day rolling average for visual smoothing
    daily_total["demand_7d_ma"] = daily_total["demand"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_total["date"], daily_total["demand"], color="#94A3B8", alpha=0.6, label="Daily Aggregate Demand")
    ax.plot(daily_total["date"], daily_total["demand_7d_ma"], color="#2563EB", linewidth=2.0, label="7-Day Moving Average")
    ax.set_title("1. Total Daily Demand Across All Products (UCI Online Retail)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig1_path = out_path / "01_total_daily_demand.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Distribution of product-level daily demand
    # -------------------------------------------------------------
    pos_demand = df[df["demand"] > 0]["demand"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(pos_demand, bins=50, range=(0, 100), color="#3B82F6", edgecolor="white")
    ax1.set_title("Demand Distribution (Positive Sales, 0-100 units)")
    ax1.set_xlabel("Daily Units Sold per Product")
    ax1.set_ylabel("Frequency Count")

    ax2.hist(np.log1p(pos_demand), bins=50, color="#10B981", edgecolor="white")
    ax2.set_title("Log-Transformed Demand Distribution: log(1 + demand)")
    ax2.set_xlabel("log(1 + demand)")
    ax2.set_ylabel("Frequency Count")

    fig.suptitle("2. Product-Level Daily Demand Distribution (Heavy Right-Skew)", fontweight="bold")
    fig.tight_layout()
    fig2_path = out_path / "02_product_demand_distribution.png"
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Daily demand for representative products
    # Rule: Pick products at 95th, 50th, and 10th percentiles of total sales volume
    # -------------------------------------------------------------
    product_totals = df.groupby("product_id")["demand"].sum().sort_values()
    n_prods = len(product_totals)
    high_idx = int(n_prods * 0.95)
    med_idx = int(n_prods * 0.50)
    low_idx = int(n_prods * 0.10)

    p_high = product_totals.index[high_idx]
    p_med = product_totals.index[med_idx]
    p_low = product_totals.index[low_idx]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    reps = [
        (p_high, f"High-Volume SKU ({p_high}) - 95th Percentile (Total: {int(product_totals[p_high])} units)", "#1E40AF"),
        (p_med, f"Median-Volume SKU ({p_med}) - 50th Percentile (Total: {int(product_totals[p_med])} units)", "#0D9488"),
        (p_low, f"Low/Intermittent SKU ({p_low}) - 10th Percentile (Total: {int(product_totals[p_low])} units)", "#D97706"),
    ]

    for ax, (pid, label, color) in zip(axes, reps):
        sub = df[df["product_id"] == pid].sort_values("date")
        ax.plot(sub["date"], sub["demand"], color=color, linewidth=1.2, marker=".", markersize=3, label=label)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Units")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    fig.suptitle("3. Daily Demand Trajectories for Representative Products (Reproducible Rule)", fontweight="bold")
    fig.tight_layout()
    fig3_path = out_path / "03_representative_products_demand.png"
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Proportion of zero-demand product-days over time
    # -------------------------------------------------------------
    daily_zeros = (
        df.groupby("date")
        .apply(lambda g: (g["demand"] == 0).sum() / len(g))
        .reset_index(name="zero_ratio")
    )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily_zeros["date"], daily_zeros["zero_ratio"] * 100.0, color="#EF4444", alpha=0.5, label="Daily Zero %")
    ax.plot(
        daily_zeros["date"],
        daily_zeros["zero_ratio"].rolling(7, min_periods=1).mean() * 100.0,
        color="#B91C1C",
        linewidth=2.0,
        label="7-Day Moving Average"
    )
    ax.set_title("4. Proportion of Zero-Demand Product-Days Over Time (Active Catalog)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("% Zero-Demand Days")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig4_path = out_path / "04_zero_demand_proportion.png"
    fig.savefig(fig4_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig4_path}")

    # -------------------------------------------------------------
    # Figure 5: Product history length and catalog sparsity
    # -------------------------------------------------------------
    prod_history_len = df.groupby("product_id")["date"].count()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(prod_history_len, bins=40, color="#6366F1", edgecolor="white")
    ax1.set_title("Product Lifespan Distribution (Calendar Days)")
    ax1.set_xlabel("Active Calendar Days (First to Last Sale)")
    ax1.set_ylabel("Product SKU Count")

    tiers = {
        "<7 d": (prod_history_len < 7).sum(),
        "7-28 d": ((prod_history_len >= 7) & (prod_history_len <= 28)).sum(),
        "29-90 d": ((prod_history_len >= 29) & (prod_history_len <= 90)).sum(),
        "91-180 d": ((prod_history_len >= 91) & (prod_history_len <= 180)).sum(),
        ">180 d": (prod_history_len > 180).sum(),
    }
    bars = ax2.bar(tiers.keys(), tiers.values(), color="#8B5CF6", edgecolor="white")
    ax2.set_title("Product Catalog History Tiers")
    ax2.set_xlabel("History Tier")
    ax2.set_ylabel("Product SKU Count")
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 30, str(int(b.get_height())), ha="center", fontsize=9)

    fig.suptitle("5. Product Catalog History Length & Sparsity Analysis", fontweight="bold")
    fig.tight_layout()
    fig5_path = out_path / "05_product_history_length.png"
    fig.savefig(fig5_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig5_path}")

    # -------------------------------------------------------------
    # Figure 6: Demand patterns by day of week
    # -------------------------------------------------------------
    df["dow"] = df["date"].dt.dayofweek
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    dow_demand = df.groupby("dow")["demand"].agg(["mean", "sum"]).reset_index()
    dow_demand["day_name"] = dow_demand["dow"].map(dow_map)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(dow_demand["day_name"], dow_demand["mean"], color="#0284C7", edgecolor="white")
    ax.set_title("6. Average Daily Demand per Product by Day of Week", fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Units Sold per Product-Day")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig6_path = out_path / "06_demand_by_day_of_week.png"
    fig.savefig(fig6_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig6_path}")

    # -------------------------------------------------------------
    # Figure 7: Monthly demand patterns (with documented 1-year caveat)
    # -------------------------------------------------------------
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    month_totals = df.groupby("year_month")["demand"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(month_totals["year_month"], month_totals["demand"], color="#475569", edgecolor="white")
    ax.set_title("7. Monthly Total Demand (Note: Sample covers Dec 2010 to Dec 2011 ~1 Year)", fontweight="bold")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Aggregate Demand (Units)")
    plt.xticks(rotation=45)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 10000, f"{int(b.get_height()):,}", ha="center", fontsize=8, rotation=90)
    fig.tight_layout()
    fig7_path = out_path / "07_monthly_demand_patterns.png"
    fig.savefig(fig7_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig7_path}")

    print("All 7 EDA figures successfully generated and saved!")


if __name__ == "__main__":
    generate_all_eda_figures()
