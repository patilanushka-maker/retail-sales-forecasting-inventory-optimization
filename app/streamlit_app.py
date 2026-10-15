"""Main Streamlit dashboard entry point for retail sales forecasting and inventory optimization.

Coordinates dataset selection between:
1. UCI Online Retail (UK Gift Ware Catalog, 2010-2011)
2. DataCo Smart Supply Chain (Omnichannel Physical Goods, 2015-2018)

Maintains complete isolation between datasets with dedicated views, metrics, and models.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# UCI Online Retail Views
from app.components import (
    render_retrospective_warning,
    render_simulation_warning,
    render_section_heading,
)
from app.views.overview import render_overview_page
from app.views.sales_analytics import render_sales_analytics_page
from app.views.forecast_performance import render_forecast_performance_page
from app.views.sku_explorer import render_sku_explorer_page
from app.views.inventory_optimization import render_inventory_optimization_page
from app.views.scenario_analysis import render_scenario_analysis_page

# DataCo Smart Supply Chain Views
from app.views.dataco_overview import render_dataco_overview_page
from app.views.dataco_sales_analytics import render_dataco_sales_analytics_page
from app.views.dataco_forecast_performance import render_dataco_forecast_performance_page
from app.views.dataco_sku_explorer import render_dataco_sku_explorer_page
from app.views.dataco_inventory_optimization import render_dataco_inventory_optimization_page


def render_sidebar() -> tuple[str, str]:
    """Render sidebar dataset selection, navigation controls, and system metadata.

    Returns
    -------
    tuple[str, str]
        (selected_dataset, selected_page)
    """
    st.sidebar.title("Dataset Selection")
    selected_dataset = st.sidebar.selectbox(
        "Choose Dataset:",
        ["UCI Online Retail", "DataCo Smart Supply Chain"],
        index=0,
        key="app_dataset_selection",
        help="Select the operational dataset to explore. Metrics and recommendations remain strictly isolated."
    )

    st.sidebar.divider()
    st.sidebar.title("Navigation")

    if selected_dataset == "UCI Online Retail":
        pages = [
            "1. Project Overview",
            "2. Sales Analytics",
            "3. Forecast Model Comparison",
            "4. SKU Deep Dive & Explorer",
            "5. Inventory Optimization & ROP",
            "6. Simulation & What-If Planning",
        ]
        selected_page = st.sidebar.radio(
            "Go to page:",
            pages,
            index=0,
            key="uci_page_selection"
        )

        st.sidebar.divider()
        st.sidebar.markdown("### 📋 UCI System Information")
        st.sidebar.caption(
            "**Dataset**: UCI Online Retail (UK)  \n"
            "**Cutoff Date**: 2011-11-11  \n"
            "**Backtest Horizon**: 28 Days (Nov 12 – Dec 9, 2011)  \n"
            "**Primary Model**: Chronological Random Forest  \n"
            "**Baseline**: 7-Day Seasonal Naive  \n"
            "**Lead Time (L)**: 7 Days  \n"
            "**Review Period (R)**: 1 Day  \n"
            "**Protection Period (P)**: 8 Days  \n"
            "**Inventory Policy**: Periodic Review (R, s, S) Simulation"
        )
    else:
        pages = [
            "1. Project Overview",
            "2. Sales Analytics",
            "3. Forecast Model Comparison",
            "4. Product Catalog & Explorer",
            "5. Inventory Optimization & ROP",
        ]
        selected_page = st.sidebar.radio(
            "Go to page:",
            pages,
            index=0,
            key="dataco_page_selection"
        )

        st.sidebar.divider()
        st.sidebar.markdown("### 📋 DataCo System Information")
        st.sidebar.caption(
            "**Dataset**: DataCo Smart Supply Chain  \n"
            "**Demand Policy**: Gross Placed Demand (excl. fraud/cancels)  \n"
            "**Decision Origin**: 2017-08-31  \n"
            "**Rolling Benchmark**: Sept 1 – Oct 2, 2017 (32 days)  \n"
            "**Fixed-Origin Horizon**: 8 Days (Sept 1 – Sept 8, 2017)  \n"
            "**Primary Model**: Random Forest Regressor  \n"
            "**Lead Time (L)**: 7 Days (Simulated)  \n"
            "**Review Period (R)**: 1 Day (Simulated)  \n"
            "**Protection Period (P)**: 8 Days  \n"
            "**Inventory Policy**: Periodic Review (R, s, S) Simulation"
        )

    return selected_dataset, selected_page


def main() -> None:
    """Run the main Streamlit application interface."""
    st.set_page_config(
        page_title="Retail Sales Forecasting & Inventory Optimization",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📦 Retail Sales Forecasting & Inventory Optimization System")
    st.caption(
        "Interactive decision support dashboard for demand forecasting, backtest evaluation, "
        "and periodic review inventory replenishment."
    )

    selected_dataset, selected_page = render_sidebar()

    # Route based on selected dataset
    if selected_dataset == "UCI Online Retail":
        # UCI Methodological safeguard banners
        render_retrospective_warning()
        render_simulation_warning()
        st.write("")

        if selected_page.startswith("1."):
            render_overview_page()
        elif selected_page.startswith("2."):
            render_sales_analytics_page()
        elif selected_page.startswith("3."):
            render_forecast_performance_page()
        elif selected_page.startswith("4."):
            render_sku_explorer_page()
        elif selected_page.startswith("5."):
            render_inventory_optimization_page()
        elif selected_page.startswith("6."):
            render_scenario_analysis_page()

    else:
        # DataCo Smart Supply Chain Views
        st.write("")
        if selected_page.startswith("1."):
            render_dataco_overview_page()
        elif selected_page.startswith("2."):
            render_dataco_sales_analytics_page()
        elif selected_page.startswith("3."):
            render_dataco_forecast_performance_page()
        elif selected_page.startswith("4."):
            render_dataco_sku_explorer_page()
        elif selected_page.startswith("5."):
            render_dataco_inventory_optimization_page()


if __name__ == "__main__":
    main()
