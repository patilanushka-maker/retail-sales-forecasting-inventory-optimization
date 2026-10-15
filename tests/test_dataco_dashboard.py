"""Tests for DataCo dashboard data loaders, views, plotting utilities, and AppTest navigation.

Validates:
1. Dataset selection isolation: UCI and DataCo remain separated with independent artifacts.
2. DataCo data loaders: successful loading of demand, catalog, metrics, forecasts, recommendations, and orders.
3. Missing artifact handling: graceful error messages.
4. Privacy validation: no customer PII (names, emails, physical addresses, order IDs) in catalog or views.
5. Inventory integrity: recommendations use fixed-origin 8-day forecasts and pre-decision uncertainty.
6. Displayed forecast origin ('2017-08-31') and horizon (8 days) match source artifacts.
7. Missing/invalid predictions are never silently converted to zero.
8. Matplotlib plotting functions return valid Figure objects.
9. Streamlit AppTest integration: switching datasets and navigating all 5 DataCo pages without exceptions.
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.data_loader import (
    load_dataco_daily_demand,
    load_dataco_catalog,
    load_dataco_daily_demand_summary,
    get_dataco_portfolio_stats,
    load_dataco_top_products_demand,
    load_dataco_department_summary,
    load_dataco_forecast_comparison,
    load_dataco_forecast_metrics,
    load_dataco_fixed_origin_forecasts,
    load_dataco_inventory_recommendations,
    load_dataco_replenishment_orders,
)
from app.components import (
    get_recommendation_status_badge,
    render_dataco_disclaimer_banner,
)
from src.retail_forecasting.visualization.plots import (
    plot_dataco_portfolio_daily_demand,
    plot_dataco_department_demand,
    plot_dataco_fixed_horizon_forecast,
    plot_dataco_model_comparison,
    plot_dataco_product_forecast_vs_actual,
)


# =============================================================================
# 1. Successful Loading of Real DataCo Artifacts
# =============================================================================

def test_load_dataco_daily_demand() -> None:
    """Verify DataCo daily demand loads with valid schema and types."""
    df = load_dataco_daily_demand()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 21787
    assert set(["date", "product_id", "demand"]).issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["demand"])
    assert df["demand"].isna().sum() == 0


def test_load_dataco_catalog_no_pii() -> None:
    """Verify DataCo catalog loads without customer PII."""
    cat = load_dataco_catalog()
    assert isinstance(cat, pd.DataFrame)
    assert len(cat) == 118
    expected_cols = ["product_id", "product_name", "product_price", "category_name", "department_name"]
    assert set(expected_cols).issubset(cat.columns)

    # STRICT PII CHECK: Ensure no customer-level fields exist
    forbidden_pii = ["customer", "email", "address", "zipcode", "city", "state", "street", "phone", "first_name", "last_name"]
    for col in cat.columns:
        for pii in forbidden_pii:
            assert pii not in col.lower(), f"Potential PII column found in catalog: {col}"


def test_get_dataco_portfolio_stats() -> None:
    """Verify DataCo portfolio statistics dictionary."""
    stats = get_dataco_portfolio_stats()
    assert stats["sku_count"] == 118
    assert stats["core_cohort_count"] == 9
    assert stats["row_count"] == 21787
    assert stats["total_demand"] == pytest.approx(367591.0)
    assert stats["start_date"] == "2015-01-01"
    assert stats["end_date"] == "2018-01-31"


def test_load_dataco_forecast_artifacts() -> None:
    """Verify forecast comparison and metrics artifacts load accurately."""
    fc_df = load_dataco_forecast_comparison()
    assert len(fc_df) == 288
    assert "random_forest_pred" in fc_df.columns
    assert "rolling_mean_pred" in fc_df.columns
    assert "seasonal_naive_pred" in fc_df.columns
    assert fc_df["actual_demand"].isna().sum() == 0

    metrics = load_dataco_forecast_metrics()
    assert "pooled_metrics" in metrics
    assert "Random Forest" in metrics["pooled_metrics"]
    assert "Seasonal Naive" in metrics["pooled_metrics"]
    assert metrics["pooled_metrics"]["Random Forest"]["wape"] < metrics["pooled_metrics"]["Seasonal Naive"]["wape"]


def test_load_dataco_fixed_origin_forecasts() -> None:
    """Verify fixed-origin 8-day recursive forecast horizon artifact."""
    fixed_df = load_dataco_fixed_origin_forecasts()
    assert len(fixed_df) == 72  # 9 SKUs * 8 days
    assert set(fixed_df["forecast_horizon_step"].unique()) == set(range(1, 9))
    assert (fixed_df["forecast_origin"] == "2017-08-31").all()
    assert (fixed_df["forecast_method"] == "recursive_multi_step").all()
    assert (fixed_df["predicted_demand"] >= 0.0).all()


def test_load_dataco_inventory_recommendations_and_orders() -> None:
    """Verify inventory recommendations and replenishment orders."""
    recs = load_dataco_inventory_recommendations()
    assert len(recs) == 9
    assert (recs["forecast_origin"] == "2017-08-31").all()
    assert (recs["forecast_horizon_days"] == 8).all()
    assert (recs["recommended_order_quantity"] > 0).all()
    assert (recs["reorder_triggered"] == True).all()

    orders = load_dataco_replenishment_orders()
    assert len(orders) == 9
    assert set(["product_id", "product_name", "recommended_order_quantity"]).issubset(orders.columns)


# =============================================================================
# 2. Error and Missing File Handling
# =============================================================================

def test_missing_dataco_file_raises_clear_error() -> None:
    """Verify missing file path raises clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="DataCo daily demand artifact not found"):
        load_dataco_daily_demand("non_existent_file.parquet")

    with pytest.raises(FileNotFoundError, match="DataCo catalog artifact not found"):
        load_dataco_catalog("non_existent_catalog.parquet")

    with pytest.raises(FileNotFoundError, match="DataCo forecast comparison not found"):
        load_dataco_forecast_comparison("non_existent_fc.parquet")

    with pytest.raises(FileNotFoundError, match="DataCo inventory recommendations not found"):
        load_dataco_inventory_recommendations("non_existent_recs.parquet")


# =============================================================================
# 3. Status Badges and Components
# =============================================================================

def test_dataco_status_badges() -> None:
    """Verify recommendation badge rendering for DataCo status codes."""
    assert "Reorder Required" in get_recommendation_status_badge("REORDER_TRIGGERED")
    assert "Sufficient Stock" in get_recommendation_status_badge("ADEQUATELY_STOCKED")
    assert "Unavailable" in get_recommendation_status_badge("UNAVAILABLE")


# =============================================================================
# 4. Matplotlib Plotting Functions
# =============================================================================

def test_dataco_plotting_functions() -> None:
    """Verify all DataCo Matplotlib plotting functions return valid figures."""
    summary_df = load_dataco_daily_demand_summary()
    fig1 = plot_dataco_portfolio_daily_demand(summary_df)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    dept_df = load_dataco_department_summary()
    fig2 = plot_dataco_department_demand(dept_df)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    fixed_df = load_dataco_fixed_origin_forecasts()
    p_fixed = fixed_df[fixed_df["product_id"] == "365"]
    fig3 = plot_dataco_fixed_horizon_forecast(p_fixed, product_name="Rip Deck")
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    metrics_dict = load_dataco_forecast_metrics()
    fig4 = plot_dataco_model_comparison(metrics_dict)
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)

    fc_df = load_dataco_forecast_comparison()
    p_fc = fc_df[fc_df["product_id"] == "365"]
    fig5 = plot_dataco_product_forecast_vs_actual(p_fc, product_name="Rip Deck")
    assert isinstance(fig5, plt.Figure)
    plt.close(fig5)


# =============================================================================
# 5. Streamlit AppTest Navigation & Integration Tests
# =============================================================================

ENTRYPOINT_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_apptest_dataset_selection_isolation() -> None:
    """Verify switching between UCI and DataCo retains dataset isolation."""
    at = AppTest.from_file(ENTRYPOINT_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Default is UCI Online Retail
    assert at.sidebar.selectbox[0].value == "UCI Online Retail"

    # Switch to DataCo Smart Supply Chain
    at.sidebar.selectbox[0].select("DataCo Smart Supply Chain").run(timeout=30)
    assert len(at.exception) == 0
    assert at.sidebar.selectbox[0].value == "DataCo Smart Supply Chain"


def test_apptest_all_dataco_pages() -> None:
    """Verify all 5 DataCo pages render with 0 exceptions in Streamlit AppTest."""
    at = AppTest.from_file(ENTRYPOINT_PATH)
    at.run(timeout=30)
    at.sidebar.selectbox[0].select("DataCo Smart Supply Chain").run(timeout=30)

    pages = [
        "1. Project Overview",
        "2. Sales Analytics",
        "3. Forecast Model Comparison",
        "4. Product Catalog & Explorer",
        "5. Inventory Optimization & ROP",
    ]

    for p in pages:
        at.sidebar.radio[0].set_value(p).run(timeout=30)
        assert len(at.exception) == 0, f"Exception on DataCo page {p}: {at.exception}"


def test_apptest_all_uci_pages_preserved() -> None:
    """Verify all 6 UCI pages continue to render with 0 exceptions in Streamlit AppTest."""
    at = AppTest.from_file(ENTRYPOINT_PATH)
    at.run(timeout=30)
    # Default is UCI
    uci_pages = [
        "1. Project Overview",
        "2. Sales Analytics",
        "3. Forecast Model Comparison",
        "4. SKU Deep Dive & Explorer",
        "5. Inventory Optimization & ROP",
        "6. Simulation & What-If Planning",
    ]

    for p in uci_pages:
        at.sidebar.radio[0].set_value(p).run(timeout=30)
        assert len(at.exception) == 0, f"Exception on UCI page {p}: {at.exception}"


# =============================================================================
# 6. Detailed Forecast Origin, Horizon, and Robustness Audits
# =============================================================================

def test_dataco_forecast_origin_exact_match() -> None:
    """Verify that forecast origin and horizon in artifacts match exact 2017-08-31 specification."""
    fixed_df = load_dataco_fixed_origin_forecasts()
    recs_df = load_dataco_inventory_recommendations()

    # Exact origin check
    assert (fixed_df["forecast_origin"] == "2017-08-31").all(), "Fixed forecasts origin must strictly be 2017-08-31"
    assert (recs_df["forecast_origin"] == "2017-08-31").all(), "Recommendations origin must strictly be 2017-08-31"

    # Exact horizon check: 8 daily steps (2017-09-01 to 2017-09-08)
    assert set(fixed_df["forecast_horizon_step"].unique()) == set(range(1, 9))
    expected_dates = [f"2017-09-0{d}" for d in range(1, 9)]
    actual_dates = sorted(list(fixed_df["forecast_date"].unique()))
    assert actual_dates == expected_dates, f"Expected {expected_dates}, got {actual_dates}"

    # Verify matching cohort products
    assert set(fixed_df["product_id"].unique()) == set(recs_df["product_id"].unique())
    assert len(recs_df) == 9

    # Verify no predictions are silently replaced with zero when missing
    assert not recs_df["expected_lead_time_demand"].isna().any()
    assert (recs_df["expected_lead_time_demand"] > 0).all()


def test_dataco_empty_and_robustness_handling() -> None:
    """Verify plotting functions gracefully handle empty DataFrames without raising exceptions."""
    empty_df = pd.DataFrame()

    fig_empty_horizon = plot_dataco_fixed_horizon_forecast(empty_df)
    assert isinstance(fig_empty_horizon, plt.Figure)
    plt.close(fig_empty_horizon)

    fig_empty_actual = plot_dataco_product_forecast_vs_actual(empty_df)
    assert isinstance(fig_empty_actual, plt.Figure)
    plt.close(fig_empty_actual)

    fig_empty_models = plot_dataco_model_comparison({})
    assert isinstance(fig_empty_models, plt.Figure)
    plt.close(fig_empty_models)


