"""Tests for dashboard data loaders, views logic, and shared component helper logic.

Validates:
1. Successful loading of valid artifacts.
2. Missing-file handling.
3. Missing-required-column handling.
4. Invalid date or malformed artifact handling.
5. Duplicate product-date key detection.
6. Preservation of legitimate unavailable recommendation records.
7. Preservation of retrospective demonstration and simulated-data indicators.
8. Metric loading from the real JSON structure.
9. Shared component helper formatting logic without requiring a browser.
10. Correct demand aggregation by date.
11. Correct ranking of products by total recorded demand.
12. Product-level filtering by date range and zero-demand day preservation.
13. Edge case handling for period statistics (empty selections, out-of-range dates).
14. Accurate derivation of inventory recommendation counts.
15. Accurate extraction of forecast metrics and graceful missing field handling.
16. Matplotlib plot generation functions.
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.data_loader import (
    get_project_root,
    load_daily_demand,
    load_daily_demand_summary,
    get_daily_demand_portfolio_stats,
    load_top_products_demand,
    get_available_product_ids,
    load_product_daily_demand,
    compute_product_period_statistics,
    load_forecast_comparison,
    get_evaluated_product_ids,
    load_product_forecast_comparison,
    compute_single_product_forecast_metrics,
    load_forecast_metrics,
    load_inventory_recommendations,
    get_inventory_recommendation_product_ids,
    compute_inventory_summary_kpis,
    filter_inventory_recommendations,
    get_single_product_inventory_details,
    load_replenishment_orders,
    get_replenishment_orders_summary,
    verify_all_dashboard_artifacts,
    check_product_data_coverage,
    generate_sku_synthesis,
)
from app.components import (
    format_metric_value,
    get_recommendation_status_badge,
    format_inventory_table,
)
from src.retail_forecasting.visualization.plots import (
    plot_portfolio_daily_demand,
    plot_top_products_bar,
    plot_product_daily_demand,
    plot_forecast_vs_actual,
)


# =============================================================================
# 1. Successful Loading of Real Artifacts
# =============================================================================

def test_successful_loading_real_artifacts():
    """Verify that all production artifacts load successfully with valid schemas."""
    # Forecast Metrics JSON
    metrics = load_forecast_metrics()
    assert isinstance(metrics, dict)
    assert "overall_metrics" in metrics
    assert "random_forest" in metrics["overall_metrics"]
    assert "seasonal_naive" in metrics["overall_metrics"]
    assert "wape" in metrics["overall_metrics"]["random_forest"]

    # Forecast Comparison Parquet
    df_fc = load_forecast_comparison()
    assert isinstance(df_fc, pd.DataFrame)
    assert len(df_fc) > 0
    assert pd.api.types.is_datetime64_any_dtype(df_fc["date"])
    expected_cols = {"date", "product_id", "actual_demand", "random_forest_pred", "seasonal_naive_pred"}
    assert expected_cols.issubset(set(df_fc.columns))
    assert df_fc[list(expected_cols)].isna().sum().sum() == 0

    # Inventory Recommendations
    df_rec = load_inventory_recommendations()
    assert isinstance(df_rec, pd.DataFrame)
    assert len(df_rec) > 0
    assert "recommendation_status" in df_rec.columns
    assert "forecast_horizon_days" in df_rec.columns
    assert "suggested_order_qty" in df_rec.columns

    # Replenishment Orders CSV
    df_orders = load_replenishment_orders()
    assert isinstance(df_orders, pd.DataFrame)
    assert len(df_orders) > 0
    assert (df_orders["suggested_order_qty"] > 0).all()
    assert (df_orders["inventory_position"] <= df_orders["reorder_point"]).all()

    # Verification helper
    status = verify_all_dashboard_artifacts()
    assert all(info["exists"] for info in status.values())


def test_daily_demand_summary_loading():
    """Verify loading aggregated summary of daily demand avoids massive row transfers."""
    # Daily aggregation dataframe
    summary_df = load_daily_demand_summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) > 300
    assert "total_demand" in summary_df.columns
    assert "active_products_count" in summary_df.columns

    # High-level portfolio statistics dictionary
    stats = get_daily_demand_portfolio_stats()
    assert isinstance(stats, dict)
    assert stats["row_count"] > 1_000_000
    assert stats["sku_count"] > 3_000
    assert stats["start_date"] == "2010-12-01"
    assert stats["end_date"] == "2011-12-09"
    assert stats["total_days"] == 374
    assert stats["total_demand"] > 0


# =============================================================================
# 2. Missing-File Handling
# =============================================================================

def test_missing_file_handling(tmp_path):
    """Verify that all loader functions raise FileNotFoundError with clear messages when files are absent."""
    missing_path = tmp_path / "non_existent.parquet"
    missing_json = tmp_path / "non_existent.json"
    missing_csv = tmp_path / "non_existent.csv"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_daily_demand(file_path=missing_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        load_forecast_comparison(file_path=missing_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        load_forecast_metrics(file_path=missing_json)

    with pytest.raises(FileNotFoundError, match="not found"):
        load_inventory_recommendations(file_path=missing_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        load_replenishment_orders(file_path=missing_csv)


# =============================================================================
# 3. Missing-Required-Column Handling
# =============================================================================

def test_missing_required_column_handling(tmp_path):
    """Verify ValueError is raised if any expected column is missing from data files."""
    # Missing 'demand' column in daily demand
    bad_demand_df = pd.DataFrame({
        "date": ["2011-01-01", "2011-01-02"],
        "product_id": ["P1", "P1"],
    })
    bad_demand_path = tmp_path / "bad_demand.parquet"
    bad_demand_df.to_parquet(bad_demand_path)

    with pytest.raises(ValueError, match="missing required columns"):
        load_daily_demand(file_path=bad_demand_path)

    # Missing 'random_forest_pred' in forecast comparison
    bad_fc_df = pd.DataFrame({
        "date": ["2011-11-12", "2011-11-13"],
        "product_id": ["P1", "P1"],
        "actual_demand": [10.0, 15.0],
        "seasonal_naive_pred": [8.0, 12.0],
    })
    bad_fc_path = tmp_path / "bad_fc.parquet"
    bad_fc_df.to_parquet(bad_fc_path)

    with pytest.raises(ValueError, match="missing required columns"):
        load_forecast_comparison(file_path=bad_fc_path)


# =============================================================================
# 4. Invalid Date or Malformed Artifact Handling
# =============================================================================

def test_invalid_date_handling(tmp_path):
    """Verify that unparseable date values trigger a clear ValueError."""
    bad_date_df = pd.DataFrame({
        "date": ["not-a-date", "2011-99-99"],
        "product_id": ["P1", "P2"],
        "actual_demand": [10.0, 15.0],
        "random_forest_pred": [9.0, 14.0],
        "seasonal_naive_pred": [8.0, 12.0],
    })
    bad_date_path = tmp_path / "bad_dates.parquet"
    bad_date_df.to_parquet(bad_date_path)

    with pytest.raises(ValueError, match="contains unparseable date"):
        load_forecast_comparison(file_path=bad_date_path)


# =============================================================================
# 5. Duplicate Product-Date Key Detection
# =============================================================================

def test_duplicate_key_detection(tmp_path):
    """Verify duplicate (product_id, date) rows raise a ValueError."""
    dup_df = pd.DataFrame({
        "date": ["2011-11-12", "2011-11-12"],  # Duplicate date for P1
        "product_id": ["P1", "P1"],
        "actual_demand": [10.0, 15.0],
        "random_forest_pred": [9.0, 14.0],
        "seasonal_naive_pred": [8.0, 12.0],
    })
    dup_path = tmp_path / "dup.parquet"
    dup_df.to_parquet(dup_path)

    with pytest.raises(ValueError, match="Detected .* duplicate"):
        load_forecast_comparison(file_path=dup_path)


# =============================================================================
# 6. Preservation of Legitimate Unavailable Recommendation Records
# =============================================================================

def test_preservation_of_unavailable_recommendation_records():
    """Verify that unavailable recommendations (H < L + R) are preserved and not coerced to 0."""
    df_rec = load_inventory_recommendations()

    # Verify presence of unavailable status
    unavailable = df_rec[df_rec["recommendation_status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON"]
    assert len(unavailable) > 0, "Should contain items with insufficient forecast horizon"

    # Legitimate nulls must NOT be replaced with 0
    assert unavailable["safety_stock"].isna().all(), "Unavailable items must have null safety_stock"
    assert unavailable["reorder_point"].isna().all(), "Unavailable items must have null reorder_point"
    assert unavailable["target_stock"].isna().all(), "Unavailable items must have null target_stock"
    assert unavailable["inventory_position"].isna().all(), "Unavailable items must have null inventory_position"
    assert (unavailable["reorder_triggered"] == False).all()

    # In contrast, available recommendations must have non-null numeric values
    available = df_rec[df_rec["recommendation_status"].isin(["REORDER_NOW", "STOCK_ADEQUATE"])]
    assert len(available) > 0
    assert available["safety_stock"].notna().all()
    assert available["reorder_point"].notna().all()
    assert available["suggested_order_qty"].notna().all()


# =============================================================================
# 7. Preservation of Retrospective and Simulated-Data Indicators
# =============================================================================

def test_preservation_of_retrospective_and_simulated_indicators():
    """Verify required methodology metadata columns are preserved exactly as booleans/integers."""
    df_rec = load_inventory_recommendations()

    # Retrospective backtest flag
    assert "is_retrospective_demonstration" in df_rec.columns
    assert (df_rec["is_retrospective_demonstration"] == True).all()

    # Simulated operational input flag
    assert "is_simulated_operational_input" in df_rec.columns
    assert (df_rec["is_simulated_operational_input"] == True).all()

    # Uncertainty fallback indicator
    assert "is_uncertainty_fallback" in df_rec.columns
    assert set(df_rec["is_uncertainty_fallback"].unique()).issubset({True, False})

    # Forecast horizon days
    assert "forecast_horizon_days" in df_rec.columns
    assert (df_rec["forecast_horizon_days"] >= 1).all()
    assert (df_rec["forecast_horizon_days"] <= 28).all()


# =============================================================================
# 8. Metric Loading from Real JSON Structure
# =============================================================================

def test_metrics_json_structure_and_validation(tmp_path):
    """Verify metrics loading validates expected top-level and model-level fields."""
    metrics = load_forecast_metrics()

    # Must have overall_metrics dictionary
    assert "overall_metrics" in metrics
    models = metrics["overall_metrics"]

    # Both models present
    assert "random_forest" in models
    assert "seasonal_naive" in models

    # Check numeric metric presence
    for model_name in ["random_forest", "seasonal_naive"]:
        model_metrics = models[model_name]
        for key in ["mae", "wape", "rmse"]:
            assert key in model_metrics
            assert isinstance(model_metrics[key], (int, float))

    # Test error handling on corrupted JSON structure
    bad_json_path = tmp_path / "bad_metrics.json"
    with open(bad_json_path, "w") as f:
        json.dump({"not_overall_metrics": {}}, f)

    with pytest.raises(ValueError, match="missing required top-level key"):
        load_forecast_metrics(file_path=bad_json_path)


# =============================================================================
# 9. Shared Component Helper Logic (Unit Testable without Streamlit Context)
# =============================================================================

def test_format_metric_value():
    """Verify format_metric_value formats various data types properly."""
    # Float
    assert format_metric_value(12.3456, format_type="float", decimals=2) == "12.35"
    assert format_metric_value(12.3456, format_type="float", decimals=1) == "12.3"

    # Integer
    assert format_metric_value(1234567, format_type="integer") == "1,234,567"
    assert format_metric_value(12.8, format_type="integer") == "13"

    # Percent (between 0 and 1)
    assert format_metric_value(0.1425, format_type="percent", decimals=1) == "14.2%"
    assert format_metric_value(15.2, format_type="percent", decimals=1) == "15.2%"

    # Currency
    assert format_metric_value(1500.5, format_type="currency", decimals=2) == "£1,500.50"
    assert format_metric_value(1500.5, format_type="currency", prefix="$", decimals=2) == "$1,500.50"

    # None and NaN
    assert format_metric_value(None) == "N/A"
    assert format_metric_value(np.nan) == "N/A"


def test_get_recommendation_status_badge():
    """Verify status badge translation strings."""
    assert "Reorder Required" in get_recommendation_status_badge("REORDER_NOW")
    assert "Sufficient Stock" in get_recommendation_status_badge("STOCK_ADEQUATE")
    assert "Recommendation Unavailable" in get_recommendation_status_badge("UNAVAILABLE_INSUFFICIENT_HORIZON")
    assert "UNKNOWN" in get_recommendation_status_badge("UNKNOWN")


def test_format_inventory_table():
    """Verify formatting inventory table adds status badges and renames headers."""
    test_df = pd.DataFrame({
        "product_id": ["P1", "P2"],
        "recommendation_status": ["REORDER_NOW", "UNAVAILABLE_INSUFFICIENT_HORIZON"],
        "simulated_on_hand": [10.0, 5.0],
        "suggested_order_qty": [50.0, np.nan],
    })

    formatted = format_inventory_table(test_df)
    assert "Product ID" in formatted.columns
    assert "Status" in formatted.columns
    assert "Simulated On-Hand Stock" in formatted.columns
    assert "Suggested Order Qty (Simulation)" in formatted.columns

    # Verify status column content
    status_values = formatted["Status"].tolist()
    assert any("Reorder Required" in s for s in status_values)
    assert any("Recommendation Unavailable" in s for s in status_values)


# =============================================================================
# 10. Sales Analytics: Demand Aggregation by Date
# =============================================================================

def test_demand_aggregation_by_date(tmp_path):
    """Verify aggregation correctly sums demand across products by date and sorts chronologically."""
    synthetic_demand = pd.DataFrame({
        "date": ["2011-01-02", "2011-01-01", "2011-01-02", "2011-01-01"],
        "product_id": ["SKU1", "SKU1", "SKU2", "SKU2"],
        "demand": [10.0, 5.0, 20.0, 15.0],
    })
    path = tmp_path / "synth_demand.parquet"
    synthetic_demand.to_parquet(path)

    summary = load_daily_demand_summary(file_path=path)
    assert len(summary) == 2
    # Verify chronological sorting
    assert summary["date"].iloc[0] == pd.Timestamp("2011-01-01")
    assert summary["date"].iloc[1] == pd.Timestamp("2011-01-02")
    # Verify exact summation
    assert summary.loc[summary["date"] == "2011-01-01", "total_demand"].iloc[0] == 20.0
    assert summary.loc[summary["date"] == "2011-01-02", "total_demand"].iloc[0] == 30.0


# =============================================================================
# 11. Sales Analytics: Ranking Products by Total Recorded Demand
# =============================================================================

def test_ranking_of_products_by_total_recorded_demand(tmp_path):
    """Verify products are ranked descending by total demand without using forecasts."""
    synthetic_demand = pd.DataFrame({
        "date": ["2011-01-01"] * 4,
        "product_id": ["P_LOW", "P_TOP", "P_MID", "P_ZERO"],
        "demand": [10.0, 500.0, 50.0, 0.0],
    })
    path = tmp_path / "synth_ranking.parquet"
    synthetic_demand.to_parquet(path)

    top_df = load_top_products_demand(n=3, file_path=path)
    assert len(top_df) == 3
    assert top_df["product_id"].tolist() == ["P_TOP", "P_MID", "P_LOW"]
    assert top_df["total_demand"].tolist() == [500.0, 50.0, 10.0]

    # Real artifact verification
    real_top10 = load_top_products_demand(n=10)
    assert len(real_top10) == 10
    assert real_top10["total_demand"].is_monotonic_decreasing


# =============================================================================
# 12. Sales Analytics: Product-Level Filtering and Zero-Demand Day Preservation
# =============================================================================

def test_product_level_filtering_and_zero_demand_preservation():
    """Verify that date filtering preserves zero-demand days and calculates stats accurately."""
    product_data = pd.DataFrame({
        "date": pd.date_range("2011-01-01", periods=10, freq="D"),
        "product_id": ["SKU_TEST"] * 10,
        "demand": [10.0, 0.0, 0.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 15.0],
    })

    # Filter for first 5 days (2011-01-01 to 2011-01-05)
    stats = compute_product_period_statistics(
        df_product=product_data,
        start_date="2011-01-01",
        end_date="2011-01-05",
    )

    assert stats["total_days"] == 5
    # Zero days must NOT be dropped
    assert stats["zero_demand_days"] == 3
    assert stats["positive_demand_days"] == 2
    # Total demand = 10 + 0 + 0 + 5 + 0 = 15
    assert stats["total_demand"] == 15.0
    # Mean demand = 15 / 5 = 3.0 (must include zero-demand days in denominator)
    assert stats["mean_demand"] == 3.0
    assert len(stats["filtered_df"]) == 5


# =============================================================================
# 13. Sales Analytics: Edge Cases for Period Statistics
# =============================================================================

def test_compute_product_period_statistics_edge_cases():
    """Verify graceful handling of empty DataFrame, unmatched date bounds, and single-day windows."""
    # Empty DataFrame
    empty_stats = compute_product_period_statistics(pd.DataFrame(columns=["date", "demand"]))
    assert empty_stats["total_demand"] == 0.0
    assert empty_stats["mean_demand"] == 0.0
    assert empty_stats["total_days"] == 0

    # Unmatched date range
    product_data = pd.DataFrame({
        "date": [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")],
        "demand": [10.0, 20.0],
    })
    out_of_range_stats = compute_product_period_statistics(
        product_data,
        start_date="2012-01-01",
        end_date="2012-01-05",
    )
    assert out_of_range_stats["total_days"] == 0
    assert out_of_range_stats["total_demand"] == 0.0

    # Single day window
    single_day = compute_product_period_statistics(
        product_data,
        start_date="2011-01-01",
        end_date="2011-01-01",
    )
    assert single_day["total_days"] == 1
    assert single_day["total_demand"] == 10.0
    assert single_day["mean_demand"] == 10.0


# =============================================================================
# 14. Overview: Inventory Recommendation Status Derivation
# =============================================================================

def test_inventory_recommendation_counts_derived():
    """Verify recommendation status counts partition the dataset accurately."""
    df_rec = load_inventory_recommendations()

    reorder_now = (df_rec["recommendation_status"] == "REORDER_NOW").sum()
    stock_adequate = (df_rec["recommendation_status"] == "STOCK_ADEQUATE").sum()
    unavailable = (df_rec["recommendation_status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON").sum()

    total = len(df_rec)
    assert total > 0
    assert reorder_now + stock_adequate + unavailable == total
    assert reorder_now > 1000
    assert stock_adequate > 1000
    assert unavailable > 0


# =============================================================================
# 15. Overview: Forecast Metrics Extraction and Missing Keys Handling
# =============================================================================

def test_forecast_metrics_extraction_and_missing_keys_handling(tmp_path):
    """Verify metrics can be extracted safely even if optional sections or fields are omitted."""
    metrics = load_forecast_metrics()

    # Verify real metrics extracted correctly
    overall = metrics["overall_metrics"]
    assert overall["random_forest"]["wape"] < overall["seasonal_naive"]["wape"]
    assert overall["random_forest"]["mae"] < overall["seasonal_naive"]["mae"]

    # Verify macro WAPE trade-off
    macro = metrics.get("macro_averaged_metrics", {})
    assert "random_forest" in macro
    assert "seasonal_naive" in macro
    assert macro["random_forest"]["macro_wape"] > macro["seasonal_naive"]["macro_wape"]


# =============================================================================
# 16. Matplotlib Plots Generation
# =============================================================================

def test_sales_analytics_plots_generation():
    """Verify that Matplotlib plotting functions return valid figure instances and close cleanly."""
    # Portfolio trend plot
    synth_summary = pd.DataFrame({
        "date": pd.date_range("2011-01-01", periods=14, freq="D"),
        "total_demand": [100 + i * 5 for i in range(14)],
    })
    fig1 = plot_portfolio_daily_demand(synth_summary)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # Top products bar plot
    synth_top = pd.DataFrame({
        "product_id": [f"SKU_{i}" for i in range(10)],
        "total_demand": [1000 - i * 50 for i in range(10)],
    })
    fig2 = plot_top_products_bar(synth_top)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    # Product daily demand plot
    synth_prod = pd.DataFrame({
        "date": pd.date_range("2011-01-01", periods=10, freq="D"),
        "demand": [0, 5, 0, 10, 0, 0, 25, 0, 0, 12],
    })
    fig3 = plot_product_daily_demand(synth_prod, product_id="SKU_TEST")
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)


# =============================================================================
# 17. Forecast Performance: Product Filtering & Chronological Ordering
# =============================================================================

def test_forecast_product_filtering_and_chronological_ordering(tmp_path):
    """Verify product filtering isolates target SKU and orders dates chronologically."""
    synthetic_fc = pd.DataFrame({
        "date": ["2011-11-15", "2011-11-12", "2011-11-14", "2011-11-12"],
        "product_id": ["SKU_A", "SKU_A", "SKU_A", "SKU_B"],
        "actual_demand": [10.0, 5.0, 8.0, 20.0],
        "random_forest_pred": [9.0, 6.0, 7.5, 18.0],
        "seasonal_naive_pred": [8.0, 5.0, 7.0, 17.0],
    })
    path = tmp_path / "synth_fc.parquet"
    synthetic_fc.to_parquet(path)

    sub_a = load_product_forecast_comparison("SKU_A", file_path=path)
    assert len(sub_a) == 3
    assert (sub_a["product_id"] == "SKU_A").all()
    # Chronological sort: 11-12, 11-14, 11-15
    assert sub_a["date"].iloc[0] == pd.Timestamp("2011-11-12")
    assert sub_a["date"].iloc[1] == pd.Timestamp("2011-11-14")
    assert sub_a["date"].iloc[2] == pd.Timestamp("2011-11-15")


# =============================================================================
# 18. Forecast Performance: Accurate Metrics Calculation (MAE, RMSE, WAPE)
# =============================================================================

def test_compute_single_product_forecast_metrics_accuracy():
    """Verify exact calculation of MAE, RMSE, and WAPE against synthetic fixture."""
    # actual: [10.0, 20.0, 30.0] -> sum = 60.0
    # rf:     [12.0, 18.0, 33.0] -> abs err: [2, 2, 3] -> sum=7 -> MAE=7/3=2.3333, RMSE=sqrt((4+4+9)/3)=2.3805, WAPE=7/60*100=11.6667
    # sn:     [15.0, 15.0, 25.0] -> abs err: [5, 5, 5] -> sum=15 -> MAE=5.0, RMSE=5.0, WAPE=15/60*100=25.0
    eval_slice = pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=3, freq="D"),
        "product_id": ["TEST_SKU"] * 3,
        "actual_demand": [10.0, 20.0, 30.0],
        "random_forest_pred": [12.0, 18.0, 33.0],
        "seasonal_naive_pred": [15.0, 15.0, 25.0],
    })

    metrics = compute_single_product_forecast_metrics(eval_slice)
    assert metrics["sample_size"] == 3
    assert metrics["missing_count"] == 0
    assert metrics["total_actual"] == 60.0

    rf = metrics["random_forest"]
    assert pytest.approx(rf["mae"], rel=1e-3) == 2.3333
    assert pytest.approx(rf["rmse"], rel=1e-3) == 2.3805
    assert pytest.approx(rf["wape"], rel=1e-3) == 11.6667
    assert rf["wape_defined"] is True

    sn = metrics["seasonal_naive"]
    assert sn["mae"] == 5.0
    assert sn["rmse"] == 5.0
    assert sn["wape"] == 25.0
    assert sn["wape_defined"] is True


# =============================================================================
# 19. Forecast Performance: Safe Zero-Denominator WAPE Handling
# =============================================================================

def test_forecast_metrics_safe_zero_denominator_wape():
    """Verify that when total actual demand is zero, WAPE is flagged undefined rather than dividing by zero."""
    zero_actuals = pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=5, freq="D"),
        "product_id": ["ZERO_SKU"] * 5,
        "actual_demand": [0.0, 0.0, 0.0, 0.0, 0.0],
        "random_forest_pred": [1.0, 0.5, 2.0, 0.0, 1.0],
        "seasonal_naive_pred": [0.0, 1.0, 0.0, 1.0, 0.0],
    })

    metrics = compute_single_product_forecast_metrics(zero_actuals)
    assert metrics["total_actual"] == 0.0

    rf = metrics["random_forest"]
    assert rf["wape_defined"] is False
    assert np.isnan(rf["wape"])
    assert rf["mae"] == 0.9  # (1+0.5+2+0+1)/5 = 4.5/5 = 0.9

    sn = metrics["seasonal_naive"]
    assert sn["wape_defined"] is False
    assert np.isnan(sn["wape"])


# =============================================================================
# 20. Forecast Performance: Missing Prediction Handling Without Substitution
# =============================================================================

def test_forecast_metrics_missing_predictions_handling():
    """Verify that missing predictions are reported and not silently substituted with actuals."""
    slice_with_missing = pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=4, freq="D"),
        "product_id": ["PARTIAL_SKU"] * 4,
        "actual_demand": [10.0, 20.0, 30.0, 40.0],
        "random_forest_pred": [10.0, np.nan, 30.0, 40.0],  # 1 missing
        "seasonal_naive_pred": [10.0, 20.0, 30.0, np.nan],  # 1 missing
    })

    metrics = compute_single_product_forecast_metrics(slice_with_missing)
    # Rows with NaN in either prediction should be identified
    assert metrics["missing_count"] == 2
    assert metrics["sample_size"] == 2  # Only row 0 and 2 are complete


# =============================================================================
# 21. Forecast Performance: Empty DataFrame & Nonexistent Product Handling
# =============================================================================

def test_forecast_metrics_empty_dataframe_and_nonexistent_product():
    """Verify graceful handling when a product has no evaluation records."""
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["date", "product_id", "actual_demand", "random_forest_pred", "seasonal_naive_pred"])
    empty_metrics = compute_single_product_forecast_metrics(empty_df)
    assert empty_metrics["sample_size"] == 0
    assert empty_metrics["missing_count"] == 0
    assert empty_metrics["random_forest"]["wape_defined"] is False

    # None input
    none_metrics = compute_single_product_forecast_metrics(None)
    assert none_metrics["sample_size"] == 0

    # Nonexistent SKU in real evaluation artifact
    nonexistent = load_product_forecast_comparison("SKU_DEFINITELY_DOES_NOT_EXIST_99999")
    assert isinstance(nonexistent, pd.DataFrame)
    assert len(nonexistent) == 0


# =============================================================================
# 22. Forecast Performance: Missing Columns Handling
# =============================================================================

def test_forecast_metrics_missing_columns_handling():
    """Verify compute_single_product_forecast_metrics handles DataFrames missing expected columns."""
    incomplete_df = pd.DataFrame({
        "date": ["2011-11-12"],
        "actual_demand": [10.0],
        # random_forest_pred missing
    })
    metrics = compute_single_product_forecast_metrics(incomplete_df)
    assert metrics["sample_size"] == 0
    assert metrics["missing_count"] == 1


# =============================================================================
# 23. Forecast Performance: Actual vs Predicted Plot Generation
# =============================================================================

def test_plot_forecast_vs_actual_generation():
    """Verify plot_forecast_vs_actual generates valid figure with actuals, RF, and baseline."""
    dates = pd.date_range("2011-11-12", periods=7, freq="D")
    y_actual = pd.Series([10, 15, 8, 20, 25, 12, 18], index=dates)
    y_pred = pd.Series([11, 14, 9, 18, 22, 14, 17], index=dates)
    baseline = pd.Series([10, 10, 10, 15, 15, 15, 15], index=dates)

    fig = plot_forecast_vs_actual(
        dates=dates,
        y_actual=y_actual,
        y_pred=y_pred,
        baseline_pred=baseline,
        title="Test Forecast vs Actual",
    )
    assert isinstance(fig, plt.Figure)
    # Check that axes contain curves
    assert len(fig.axes[0].lines) >= 3
    plt.close(fig)


# =============================================================================
# 24. Inventory Optimization: Recommendation Status Counts
# =============================================================================

def test_inventory_summary_kpis_status_counts():
    """Verify compute_inventory_summary_kpis derives exact status counts from fixture."""
    synthetic_rec = pd.DataFrame({
        "product_id": ["P1", "P2", "P3", "P4", "P5"],
        "recommendation_status": [
            "REORDER_NOW",
            "REORDER_NOW",
            "STOCK_ADEQUATE",
            "STOCK_ADEQUATE",
            "UNAVAILABLE_INSUFFICIENT_HORIZON",
        ],
        "suggested_order_qty": [10, 20, 0, 0, np.nan],
    })

    kpis = compute_inventory_summary_kpis(synthetic_rec)
    assert kpis["is_valid"] is True
    assert kpis["total_products"] == 5
    assert kpis["reorder_now_count"] == 2
    assert kpis["stock_adequate_count"] == 2
    assert kpis["unavailable_count"] == 1
    assert kpis["has_duplicates"] is False
    assert kpis["duplicate_count"] == 0


# =============================================================================
# 25. Inventory Optimization: Total Suggested Qty Excludes Unavailable
# =============================================================================

def test_inventory_summary_kpis_total_suggested_qty_excludes_unavailable():
    """Verify total suggested order quantity sums eligible rows only and ignores unavailable."""
    synthetic_rec = pd.DataFrame({
        "product_id": ["P1", "P2", "P3", "P4"],
        "recommendation_status": [
            "REORDER_NOW",
            "REORDER_NOW",
            "STOCK_ADEQUATE",
            "UNAVAILABLE_INSUFFICIENT_HORIZON",
        ],
        "suggested_order_qty": [50.0, 75.0, 0.0, 999.0],  # 999 must NOT be counted!
    })

    kpis = compute_inventory_summary_kpis(synthetic_rec)
    # Total suggested quantity across eligible recommendations should be 50 + 75 = 125
    assert kpis["total_suggested_order_qty"] == 125


# =============================================================================
# 26. Inventory Optimization: Empty Artifact and Missing Columns Handling
# =============================================================================

def test_inventory_summary_kpis_empty_and_missing_columns():
    """Verify graceful handling when recommendation DataFrame is empty or missing columns."""
    # Empty DataFrame
    empty_kpis = compute_inventory_summary_kpis(pd.DataFrame())
    assert empty_kpis["is_valid"] is False
    assert empty_kpis["total_products"] == 0
    assert empty_kpis["reorder_now_count"] == 0

    # None DataFrame
    none_kpis = compute_inventory_summary_kpis(None)
    assert none_kpis["is_valid"] is False

    # Missing recommendation_status column
    bad_df = pd.DataFrame({"product_id": ["P1", "P2"]})
    bad_kpis = compute_inventory_summary_kpis(bad_df)
    assert bad_kpis["is_valid"] is False
    assert bad_kpis["reorder_now_count"] is None


# =============================================================================
# 27. Inventory Optimization: Duplicate Product ID Detection
# =============================================================================

def test_inventory_summary_kpis_duplicate_detection():
    """Verify duplicate product_id entries are identified and reported."""
    dup_df = pd.DataFrame({
        "product_id": ["P1", "P1", "P2"],  # P1 duplicated
        "recommendation_status": ["REORDER_NOW", "STOCK_ADEQUATE", "STOCK_ADEQUATE"],
        "suggested_order_qty": [10, 0, 0],
    })

    kpis = compute_inventory_summary_kpis(dup_df)
    assert kpis["has_duplicates"] is True
    assert kpis["duplicate_count"] == 1


# =============================================================================
# 28. Inventory Optimization: Status Filtering
# =============================================================================

def test_inventory_recommendations_status_filtering():
    """Verify filtering recommendations by status isolates expected subsets."""
    test_df = pd.DataFrame({
        "product_id": ["P1", "P2", "P3"],
        "recommendation_status": ["REORDER_NOW", "STOCK_ADEQUATE", "UNAVAILABLE_INSUFFICIENT_HORIZON"],
    })

    # Filter REORDER_NOW
    f_reorder = filter_inventory_recommendations(test_df, status_filter="REORDER_NOW")
    assert len(f_reorder) == 1
    assert f_reorder["product_id"].iloc[0] == "P1"

    # Filter All
    f_all = filter_inventory_recommendations(test_df, status_filter="All")
    assert len(f_all) == 3

    # Filter Non-matching status
    f_none = filter_inventory_recommendations(test_df, status_filter="NON_EXISTENT")
    assert len(f_none) == 0


# =============================================================================
# 29. Inventory Optimization: Product ID Search Filtering
# =============================================================================

def test_inventory_recommendations_product_search_filtering():
    """Verify case-insensitive search by product ID substring works safely."""
    test_df = pd.DataFrame({
        "product_id": ["85123A", "85123B", "22741", "POST"],
        "recommendation_status": ["REORDER_NOW"] * 4,
    })

    # Substring search (case insensitive)
    f_search = filter_inventory_recommendations(test_df, search_query="85123")
    assert len(f_search) == 2
    assert set(f_search["product_id"]) == {"85123A", "85123B"}

    # Whitespace trimming
    f_trim = filter_inventory_recommendations(test_df, search_query="  post  ")
    assert len(f_trim) == 1
    assert f_trim["product_id"].iloc[0] == "POST"

    # Null product IDs handling
    null_id_df = pd.DataFrame({
        "product_id": ["85123A", None, np.nan],
        "recommendation_status": ["REORDER_NOW"] * 3,
    })
    f_null_safe = filter_inventory_recommendations(null_id_df, search_query="85123")
    assert len(f_null_safe) == 1


# =============================================================================
# 30. Inventory Optimization: Product IDs List and Null Handling
# =============================================================================

def test_inventory_product_ids_and_null_handling(tmp_path):
    """Verify get_inventory_recommendation_product_ids handles nulls and sorts IDs."""
    test_df = pd.DataFrame({
        "product_id": ["SKU_Z", "SKU_A", None, "SKU_M"],
        "as_of_date": ["2011-11-11"] * 4,
        "forecast_start_date": ["2011-11-12"] * 4,
        "forecast_end_date": ["2011-12-09"] * 4,
        "is_retrospective_demonstration": [True] * 4,
        "recommendation_type": ["TEST"] * 4,
        "forecast_model": ["RF"] * 4,
        "forecast_horizon_days": [28] * 4,
        "service_level": [0.95] * 4,
        "z_score": [1.6449] * 4,
        "lead_time_days": [7] * 4,
        "review_period_days": [1] * 4,
        "protection_period_days": [8] * 4,
        "is_simulated_operational_input": [True] * 4,
        "expected_daily_demand": [1.0] * 4,
        "expected_lead_time_demand": [7.0] * 4,
        "expected_protection_period_demand": [8.0] * 4,
        "forecast_error_std": [2.0] * 4,
        "error_history_count": [28] * 4,
        "is_uncertainty_fallback": [False] * 4,
        "safety_stock": [5.0] * 4,
        "reorder_point": [12.0] * 4,
        "simulated_on_hand": [10.0] * 4,
        "simulated_on_order": [0.0] * 4,
        "simulated_backorders": [0.0] * 4,
        "inventory_position": [10.0] * 4,
        "target_stock": [13.0] * 4,
        "suggested_order_qty": [3] * 4,
        "reorder_triggered": [True] * 4,
        "economic_order_quantity": [50.0] * 4,
        "recommendation_status": ["REORDER_NOW"] * 4,
    })
    path = tmp_path / "synth_recs.parquet"
    test_df.to_parquet(path)

    ids = get_inventory_recommendation_product_ids(file_path=path)
    # None dropped, sorted
    assert ids == ["SKU_A", "SKU_M", "SKU_Z"]


# =============================================================================
# 31. Inventory Optimization: Single Product Details and Unavailable Details
# =============================================================================

def test_get_single_product_inventory_details_and_unavailable_details():
    """Verify single-product details extraction preserves status and handles duplicates."""
    test_df = pd.DataFrame({
        "product_id": ["SKU_OK", "SKU_UNAVAIL", "SKU_DUP", "SKU_DUP"],
        "recommendation_status": [
            "REORDER_NOW",
            "UNAVAILABLE_INSUFFICIENT_HORIZON",
            "STOCK_ADEQUATE",
            "STOCK_ADEQUATE",
        ],
        "inventory_position": [10.0, np.nan, 20.0, 20.0],
        "safety_stock": [5.0, np.nan, 8.0, 8.0],
        "suggested_order_qty": [15, np.nan, 0, 0],
    })

    # Available SKU
    det_ok = get_single_product_inventory_details("SKU_OK", df=test_df)
    assert det_ok is not None
    assert det_ok["recommendation_status"] == "REORDER_NOW"
    assert det_ok["suggested_order_qty"] == 15
    assert det_ok["_has_duplicates"] is False

    # Unavailable SKU
    det_unavail = get_single_product_inventory_details("SKU_UNAVAIL", df=test_df)
    assert det_unavail is not None
    assert det_unavail["recommendation_status"] == "UNAVAILABLE_INSUFFICIENT_HORIZON"
    assert np.isnan(det_unavail["safety_stock"])

    # Duplicate SKU
    det_dup = get_single_product_inventory_details("SKU_DUP", df=test_df)
    assert det_dup is not None
    assert det_dup["_has_duplicates"] is True
    assert det_dup["_duplicate_count"] == 2

    # Nonexistent SKU
    assert get_single_product_inventory_details("NON_EXISTENT", df=test_df) is None


# =============================================================================
# 32. Inventory Optimization: Replenishment Orders Summary Edge Cases
# =============================================================================

def test_replenishment_orders_summary_edge_cases(tmp_path):
    """Verify replenishment orders summary handles empty DataFrames and missing files."""
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["product_id", "suggested_order_qty"])
    s_empty = get_replenishment_orders_summary(df_orders=empty_df)
    assert s_empty["is_available"] is True
    assert s_empty["order_count"] == 0
    assert s_empty["total_order_units"] == 0

    # Non-existent file
    missing_path = tmp_path / "non_existent_orders.csv"
    s_missing = get_replenishment_orders_summary(file_path=missing_path)
    assert s_missing["is_available"] is False
    assert s_missing["order_count"] == 0


# =============================================================================
# 33. Inventory Optimization: Real Artifact Reconciliation & Simulated Flags
# =============================================================================

def test_replenishment_orders_real_reconciliation_and_simulated_flags():
    """Verify real replenishment orders reconcile with REORDER_NOW items and preserve simulated metadata."""
    rec_df = load_inventory_recommendations()
    orders_df = load_replenishment_orders()

    reorder_now_items = rec_df[rec_df["recommendation_status"] == "REORDER_NOW"]
    assert len(orders_df) == len(reorder_now_items), "Every REORDER_NOW item must map to a replenishment order"
    assert orders_df["suggested_order_qty"].sum() == reorder_now_items["suggested_order_qty"].sum()

    # Verify simulated and retrospective flags
    assert (orders_df["is_retrospective_demonstration"] == True).all()
    assert (orders_df["is_simulated_operational_input"] == True).all()
    assert (orders_df["suggested_order_qty"] > 0).all()
    assert (orders_df["inventory_position"] <= orders_df["reorder_point"]).all()


# =============================================================================
# 34. SKU Explorer: Product Selector Population and Null Handling
# =============================================================================

def test_sku_explorer_product_selector_population_and_null_ids():
    """Verify product selector population filters nulls, empty strings, and sorts IDs."""
    synth_demand = pd.DataFrame({
        "date": pd.date_range("2011-01-01", periods=6, freq="D"),
        "product_id": ["SKU_B", None, "SKU_A", np.nan, "  ", "SKU_C"],
        "demand": [10, 0, 5, 0, 0, 8],
    })

    ids = get_available_product_ids(df=synth_demand)
    # None, NaN, whitespace stripped and dropped; sorted unique
    assert ids == ["SKU_A", "SKU_B", "SKU_C"]

    # Null, empty, and NaN product_id checks in check_product_data_coverage
    cov_none = check_product_data_coverage(None)
    assert cov_none["is_valid_id"] is False
    assert cov_none["has_demand"] is False

    cov_nan = check_product_data_coverage(np.nan)
    assert cov_nan["is_valid_id"] is False

    cov_empty = check_product_data_coverage("   ")
    assert cov_empty["is_valid_id"] is False


# =============================================================================
# 35. SKU Explorer: Historical Demand Product and Date-Range Filtering
# =============================================================================

def test_sku_explorer_historical_demand_product_and_date_range_filtering():
    """Verify single-product isolation and date-range filtering."""
    synth_demand = pd.DataFrame({
        "date": pd.date_range("2011-01-01", periods=5, freq="D").tolist() * 2,
        "product_id": ["SKU_1"] * 5 + ["SKU_2"] * 5,
        "demand": [10.0, 0.0, 20.0, 0.0, 30.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    })

    # Product isolation
    df_sku1 = load_product_daily_demand("SKU_1", df=synth_demand)
    assert len(df_sku1) == 5
    assert (df_sku1["product_id"] == "SKU_1").all()

    # Date range filtering strictly on SKU_1: Jan 2 to Jan 4 (3 days: demands [0, 20, 0])
    stats = compute_product_period_statistics(
        df_product=df_sku1,
        start_date="2011-01-02",
        end_date="2011-01-04",
    )
    assert stats["total_days"] == 3
    assert stats["total_demand"] == 20.0
    assert stats["mean_demand"] == pytest.approx(20.0 / 3.0)
    assert stats["positive_demand_days"] == 1
    assert stats["zero_demand_days"] == 2


# =============================================================================
# 36. SKU Explorer: Accurate Demand Metrics and Zero-Day Counts Preservation
# =============================================================================

def test_sku_explorer_demand_metrics_and_zero_day_counts():
    """Verify exact total, mean, positive-day, and zero-day calculations."""
    synth_prod = pd.DataFrame({
        "date": pd.date_range("2011-05-01", periods=10, freq="D"),
        "product_id": ["TARGET_SKU"] * 10,
        "demand": [0.0, 15.0, 0.0, 0.0, 25.0, 0.0, 10.0, 0.0, 0.0, 50.0],
    })

    stats = compute_product_period_statistics(synth_prod)
    assert stats["total_days"] == 10
    assert stats["total_demand"] == 100.0  # 15 + 25 + 10 + 50
    assert stats["mean_demand"] == 10.0    # 100 / 10
    assert stats["positive_demand_days"] == 4
    assert stats["zero_demand_days"] == 6


# =============================================================================
# 37. SKU Explorer: Product with Historical Demand but No Forecast Records
# =============================================================================

def test_sku_explorer_coverage_demand_only_no_forecast_or_inventory():
    """Verify coverage correctly reports product in demand but absent in forecast and inventory."""
    synth_demand = pd.DataFrame({
        "date": ["2011-01-01", "2011-01-02"],
        "product_id": ["DEMAND_ONLY_SKU", "DEMAND_ONLY_SKU"],
        "demand": [10.0, 20.0],
    })
    synth_forecast = pd.DataFrame({
        "date": ["2011-11-12"],
        "product_id": ["OTHER_SKU"],
        "actual_demand": [5.0],
        "random_forest_pred": [5.0],
        "seasonal_naive_pred": [5.0],
    })
    synth_rec = pd.DataFrame({
        "product_id": ["OTHER_SKU"],
        "recommendation_status": ["STOCK_ADEQUATE"],
    })

    cov = check_product_data_coverage(
        "DEMAND_ONLY_SKU",
        demand_df=synth_demand,
        forecast_df=synth_forecast,
        inventory_df=synth_rec,
    )

    assert cov["has_demand"] is True
    assert cov["demand_count"] == 2
    assert cov["has_forecast"] is False
    assert cov["forecast_count"] == 0
    assert cov["has_inventory"] is False
    assert cov["is_fully_covered"] is False

    # Also verify against real artifact SKU '21690' (sparse SKU in demand catalog without forecast)
    real_cov = check_product_data_coverage("21690")
    assert real_cov["has_demand"] is True
    assert real_cov["has_forecast"] is False
    assert real_cov["has_inventory"] is False


# =============================================================================
# 38. SKU Explorer: Product with Forecast Records but No Inventory Recommendation
# =============================================================================

def test_sku_explorer_coverage_forecast_present_no_inventory():
    """Verify coverage reporting for SKU present in forecast but absent in inventory recommendations."""
    synth_demand = pd.DataFrame({
        "date": ["2011-11-12"],
        "product_id": ["FC_ONLY_SKU"],
        "demand": [15.0],
    })
    synth_forecast = pd.DataFrame({
        "date": ["2011-11-12"],
        "product_id": ["FC_ONLY_SKU"],
        "actual_demand": [15.0],
        "random_forest_pred": [14.0],
        "seasonal_naive_pred": [13.0],
    })
    synth_rec = pd.DataFrame({
        "product_id": ["ANOTHER_SKU"],
        "recommendation_status": ["STOCK_ADEQUATE"],
    })

    cov = check_product_data_coverage(
        "FC_ONLY_SKU",
        demand_df=synth_demand,
        forecast_df=synth_forecast,
        inventory_df=synth_rec,
    )
    assert cov["has_demand"] is True
    assert cov["has_forecast"] is True
    assert cov["has_inventory"] is False
    assert cov["is_fully_covered"] is False


# =============================================================================
# 39. SKU Explorer: Correct Product-Level MAE, RMSE, and WAPE Calculations
# =============================================================================

def test_sku_explorer_forecast_metrics_mae_rmse_wape():
    """Verify compute_single_product_forecast_metrics accuracy on clean evaluation rows."""
    eval_slice = pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=4, freq="D"),
        "product_id": ["SKU_METRIC_TEST"] * 4,
        "actual_demand": [10.0, 20.0, 10.0, 0.0],  # sum = 40.0
        "random_forest_pred": [12.0, 18.0, 8.0, 2.0],  # abs err: [2, 2, 2, 2] -> sum=8
        "seasonal_naive_pred": [14.0, 16.0, 6.0, 4.0],  # abs err: [4, 4, 4, 4] -> sum=16
    })

    metrics = compute_single_product_forecast_metrics(eval_slice)
    assert metrics["sample_size"] == 4
    assert metrics["total_actual"] == 40.0

    # RF: MAE = 8/4 = 2.0; RMSE = sqrt((4+4+4+4)/4) = 2.0; WAPE = (8/40)*100 = 20.0%
    rf = metrics["random_forest"]
    assert rf["mae"] == 2.0
    assert rf["rmse"] == 2.0
    assert rf["wape"] == 20.0
    assert rf["wape_defined"] is True

    # SN: MAE = 16/4 = 4.0; RMSE = 4.0; WAPE = (16/40)*100 = 40.0%
    sn = metrics["seasonal_naive"]
    assert sn["mae"] == 4.0
    assert sn["rmse"] == 4.0
    assert sn["wape"] == 40.0
    assert sn["wape_defined"] is True


# =============================================================================
# 40. SKU Explorer: Zero-Denominator WAPE Handling
# =============================================================================

def test_sku_explorer_zero_denominator_wape():
    """Verify that when actual demand sum is zero, WAPE is flagged undefined rather than dividing by zero."""
    zero_slice = pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=3, freq="D"),
        "product_id": ["ZERO_DEMAND_SKU"] * 3,
        "actual_demand": [0.0, 0.0, 0.0],
        "random_forest_pred": [1.0, 2.0, 1.0],
        "seasonal_naive_pred": [2.0, 1.0, 0.0],
    })

    metrics = compute_single_product_forecast_metrics(zero_slice)
    assert metrics["total_actual"] == 0.0
    assert metrics["random_forest"]["wape_defined"] is False
    assert np.isnan(metrics["random_forest"]["wape"])
    assert metrics["seasonal_naive"]["wape_defined"] is False
    assert np.isnan(metrics["seasonal_naive"]["wape"])


# =============================================================================
# 41. SKU Explorer: Missing Forecast Values Reported Without Substitution
# =============================================================================

def test_sku_explorer_missing_forecast_values_no_substitution():
    """Verify missing predictions are identified in missing_count without substitution."""
    slice_with_nans = pd.DataFrame({
        "date": pd.date_range("2011-11-12", periods=5, freq="D"),
        "product_id": ["PARTIAL_SKU"] * 5,
        "actual_demand": [10.0, 20.0, 30.0, 40.0, 50.0],
        "random_forest_pred": [10.0, np.nan, 30.0, 40.0, np.nan],  # 2 missing
        "seasonal_naive_pred": [10.0, 20.0, np.nan, 40.0, 50.0],  # 1 missing
    })

    metrics = compute_single_product_forecast_metrics(slice_with_nans)
    # Rows 1, 2, and 4 have missing values -> 3 missing rows, 2 complete rows (0 and 3)
    assert metrics["missing_count"] == 3
    assert metrics["sample_size"] == 2


# =============================================================================
# 42. SKU Explorer: Duplicate Inventory Recommendation Handling
# =============================================================================

def test_sku_explorer_duplicate_inventory_recommendations_handling():
    """Verify duplicate recommendation records are flagged and prevented from arbitrary selection."""
    synth_recs = pd.DataFrame({
        "product_id": ["DUP_SKU", "DUP_SKU", "CLEAN_SKU"],
        "recommendation_status": ["REORDER_NOW", "STOCK_ADEQUATE", "STOCK_ADEQUATE"],
        "suggested_order_qty": [100, 0, 0],
    })

    cov = check_product_data_coverage("DUP_SKU", inventory_df=synth_recs)
    assert cov["has_inventory"] is True
    assert cov["inventory_has_duplicates"] is True
    assert cov["inventory_duplicate_count"] == 2
    assert cov["inventory_status"] == "DUPLICATE_RECORDS"
    assert cov["is_fully_covered"] is False

    details = get_single_product_inventory_details("DUP_SKU", df=synth_recs)
    assert details is not None
    assert details["_has_duplicates"] is True
    assert details["_duplicate_count"] == 2


# =============================================================================
# 43. SKU Explorer: Empty Demand, Forecast, or Inventory Artifacts
# =============================================================================

def test_sku_explorer_empty_artifacts_handling():
    """Verify safe execution when demand, forecast, or inventory tables are empty."""
    empty_demand = pd.DataFrame(columns=["date", "product_id", "demand"])
    empty_forecast = pd.DataFrame(columns=["date", "product_id", "actual_demand", "random_forest_pred", "seasonal_naive_pred"])
    empty_inventory = pd.DataFrame(columns=["product_id", "recommendation_status"])

    cov = check_product_data_coverage(
        "ANY_SKU",
        demand_df=empty_demand,
        forecast_df=empty_forecast,
        inventory_df=empty_inventory,
    )
    assert cov["has_demand"] is False
    assert cov["has_forecast"] is False
    assert cov["has_inventory"] is False

    p_demand = load_product_daily_demand("ANY_SKU", df=empty_demand)
    assert isinstance(p_demand, pd.DataFrame)
    assert p_demand.empty

    p_fc = load_product_forecast_comparison("ANY_SKU", df=empty_forecast)
    assert isinstance(p_fc, pd.DataFrame)
    assert p_fc.empty

    p_inv = get_single_product_inventory_details("ANY_SKU", df=empty_inventory)
    assert p_inv is None


# =============================================================================
# 44. SKU Explorer: Product ID With No Matching Records
# =============================================================================

def test_sku_explorer_nonexistent_product_id():
    """Verify query for nonexistent SKU returns clean empty structures without raising exceptions."""
    nonexistent_id = "NON_EXISTENT_SKU_99999"

    cov = check_product_data_coverage(nonexistent_id)
    assert cov["is_valid_id"] is True
    assert cov["has_demand"] is False
    assert cov["has_forecast"] is False
    assert cov["has_inventory"] is False

    demand = load_product_daily_demand(nonexistent_id)
    assert demand.empty

    fc = load_product_forecast_comparison(nonexistent_id)
    assert fc.empty

    inv = get_single_product_inventory_details(nonexistent_id)
    assert inv is None


# =============================================================================
# 45. SKU Explorer: Retrospective and Simulated Data Labeling in Synthesis
# =============================================================================

def test_sku_explorer_retrospective_and_simulated_labeling():
    """Verify that synthesis includes retrospective backtest dates, simulated operational labels, and governance disclaimers."""
    coverage = {
        "product_id": "TEST_SKU",
        "has_demand": True,
        "has_forecast": True,
        "has_inventory": True,
        "inventory_has_duplicates": False,
        "is_fully_covered": True,
    }
    demand_stats = {
        "total_demand": 500.0,
        "mean_demand": 5.0,
        "total_days": 100,
        "positive_demand_days": 80,
        "zero_demand_days": 20,
    }
    forecast_metrics = {
        "sample_size": 28,
        "total_actual": 140.0,
        "random_forest": {"mae": 1.2, "rmse": 1.5, "wape": 12.0, "wape_defined": True},
        "seasonal_naive": {"mae": 2.0, "rmse": 2.4, "wape": 20.0, "wape_defined": True},
    }
    inventory_details = {
        "recommendation_status": "REORDER_NOW",
        "inventory_position": 10.0,
        "reorder_point": 25.0,
        "suggested_order_qty": 30.0,
        "safety_stock": 8.0,
        "expected_lead_time_demand": 17.0,
        "_has_duplicates": False,
    }

    synthesis = generate_sku_synthesis(
        coverage=coverage,
        demand_stats=demand_stats,
        forecast_metrics=forecast_metrics,
        inventory_details=inventory_details,
    )

    # Check retrospective forecast labeling
    assert "retrospective evaluation window" in synthesis["forecast_commentary"].lower()
    assert "Nov 12 – Dec 9, 2011" in synthesis["forecast_commentary"]
    assert "universally superior" in synthesis["forecast_commentary"]

    # Check simulated operational inventory labeling
    assert "simulated demonstration assumptions" in synthesis["inventory_commentary"]
    assert "REORDER_NOW" in synthesis["inventory_commentary"]

    # Check demand interpretation (zero demand does not prove stock availability)
    assert "unobserved stockouts" in synthesis["demand_commentary"]


# =============================================================================
# 46. SKU Explorer: View Import and Callable Verification
# =============================================================================

def test_sku_explorer_view_import_and_callable():
    """Verify SKU Deep Dive & Explorer view module imports and exposes render function."""
    from app.views.sku_explorer import render_sku_explorer_page
    assert callable(render_sku_explorer_page)
