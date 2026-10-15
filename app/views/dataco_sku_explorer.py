"""DataCo Smart Supply Chain: Product Catalog & Explorer view for Streamlit.

Presents:
1. Product catalog table across all 118 distinct SKUs (product ID, name, price, department, category).
2. Privacy safeguard: Zero customer PII displayed (no customer names, addresses, emails, or order PII).
3. Department & Category filters with live product search.
4. Individual SKU demand history deep-dive (total lifetime units, active demand days, zero-demand days).
5. Core 9 benchmark cohort membership indicator.
"""

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.data_loader import (
    load_dataco_catalog,
    load_dataco_daily_demand,
    load_dataco_forecast_metrics,
)
from app.components import (
    format_metric_value,
    render_dataco_disclaimer_banner,
    render_section_heading,
)


def render_dataco_sku_explorer_page() -> None:
    """Render DataCo Product Explorer view."""
    render_dataco_disclaimer_banner()

    render_section_heading(
        title="DataCo Product Catalog & Demand Explorer",
        subtitle="Catalog discovery and SKU-level demand histories across 118 verified physical retail products.",
        icon="🔍",
    )

    st.info(
        "🔒 **Privacy & Data Integrity Standard**  \n"
        "• **Product-Centric Analytics**: All tables, statistics, and charts display product attributes, catalog taxonomy, "
        "and aggregated demand volumes only.  \n"
        "• **Zero Customer PII**: Customer first names, last names, physical street addresses, email addresses, and order-level "
        "personal data are strictly excluded from all dashboard views and outputs.",
        icon="🔒",
    )

    try:
        catalog_df = load_dataco_catalog()
        demand_df = load_dataco_daily_demand()
        metrics = load_dataco_forecast_metrics()
        core_cohort = set(metrics.get("metadata", {}).get("cohort", []))
    except Exception as err:
        st.error(f"Error loading DataCo catalog or demand: {err}")
        return

    # 1. Catalog Explorer & Filters
    st.markdown("### 📦 Retail Product Catalog Explorer (118 SKUs)")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        departments = ["All Departments"] + sorted(list(catalog_df["department_name"].dropna().unique()))
        selected_dept = st.selectbox("Filter by Department:", options=departments, key="dataco_catalog_dept_filter")
    with col_f2:
        search_query = st.text_input("Search Product Name or ID:", placeholder="e.g. Cleat, Nike, 365...", key="dataco_catalog_search")

    filtered_cat = catalog_df.copy()
    if selected_dept != "All Departments":
        filtered_cat = filtered_cat[filtered_cat["department_name"] == selected_dept]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_cat = filtered_cat[
            filtered_cat["product_name"].str.lower().str.contains(q) |
            filtered_cat["product_id"].astype(str).str.contains(q)
        ]

    # Add core cohort indicator
    filtered_cat["Is Core 9 Cohort"] = filtered_cat["product_id"].apply(
        lambda pid: "⭐ Core 9 Benchmark" if pid in core_cohort else "Catalog SKU"
    )

    cat_display = filtered_cat[[
        "product_id", "product_name", "product_price", "department_name", "category_name", "Is Core 9 Cohort"
    ]].copy()
    cat_display["product_price"] = cat_display["product_price"].apply(lambda p: f"${p:.2f}")
    cat_display.columns = ["Product ID", "Product Name", "Unit Price", "Department", "Category", "Cohort Status"]

    st.dataframe(cat_display, use_container_width=True, hide_index=True)
    st.caption(f"Showing **{len(cat_display)}** matching products out of **{len(catalog_df)}** catalog SKUs.")

    st.write("")

    # 2. Individual SKU Demand History Deep Dive
    render_section_heading(
        title="SKU Demand Profile Deep Dive",
        subtitle="Select a product to inspect historical daily demand patterns and catalog attributes.",
        icon="📊",
    )

    available_pids = sorted(list(catalog_df["product_id"].unique()))
    name_lookup = dict(zip(catalog_df["product_id"], catalog_df["product_name"]))

    selected_pid = st.selectbox(
        "Choose SKU to Inspect:",
        options=available_pids,
        format_func=lambda pid: f"SKU {pid} — {name_lookup.get(pid, 'Unknown')}",
        key="dataco_sku_explorer_select"
    )

    prod_info = catalog_df[catalog_df["product_id"] == selected_pid].iloc[0]
    p_name = prod_info["product_name"]
    p_price = prod_info["product_price"]
    p_dept = prod_info["department_name"]
    p_cat = prod_info["category_name"]
    is_core = selected_pid in core_cohort

    # Product Metadata Cards
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Product Name", p_name)
    col_i2.metric("Catalog Price", f"${p_price:.2f}")
    col_i3.metric("Taxonomy", f"{p_dept} / {p_cat}")
    col_i4.metric("Benchmark Status", "⭐ Core 9 Cohort" if is_core else "Catalog SKU")

    # Historical demand series
    p_demand = demand_df[demand_df["product_id"] == selected_pid].sort_values("date")
    total_units = float(p_demand["demand"].sum())
    active_days = int((p_demand["demand"] > 0).sum())
    zero_days = int((p_demand["demand"] == 0).sum())
    mean_demand = float(p_demand["demand"].mean()) if len(p_demand) > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Lifetime Gross Demand", f"{int(total_units):,} units")
    kpi2.metric("Days with Recorded Demand", f"{active_days:,} days")
    kpi3.metric("Zero-Demand Days", f"{zero_days:,} days")
    kpi4.metric("Avg Daily Placed Demand", f"{mean_demand:.2f} units/day")

    # Time series chart
    fig_hist, ax = plt.subplots(figsize=(11, 4.2))
    fig_hist.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")
    ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")

    ax.plot(p_demand["date"], p_demand["demand"], color="#38BDF8", linewidth=1.2, alpha=0.7, label="Daily Placed Demand")
    rolling_14 = p_demand["demand"].rolling(14, min_periods=3).mean()
    ax.plot(p_demand["date"], rolling_14, color="#F59E0B", linewidth=2.0, label="14-Day Rolling Mean")

    ax.set_title(f"Historical Demand Profile — {p_name} (SKU {selected_pid})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Order Date", fontsize=10)
    ax.set_ylabel("Demand Units", fontsize=10)
    ax.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#E2E8F0", loc="upper right")
    fig_hist.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)
