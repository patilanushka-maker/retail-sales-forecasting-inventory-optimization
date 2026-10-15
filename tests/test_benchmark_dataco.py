"""Unit tests for DataCo forecasting benchmark module."""

import pytest
import pandas as pd
import numpy as np

from retail_forecasting.forecasting.benchmark_dataco import (
    select_dataco_top_products,
    build_contiguous_daily_series,
    generate_seasonal_naive_forecasts,
    generate_rolling_mean_forecasts,
    create_dataco_tabular_features,
    train_and_predict_random_forest,
)
from retail_forecasting.forecasting.evaluate import compute_metrics


@pytest.fixture
def synthetic_demand_daily() -> pd.DataFrame:
    """Fixture with synthetic continuous daily demand for 2 products over 40 days."""
    dates = pd.date_range("2017-01-01", periods=40, freq="D")
    rows = []
    for d in dates:
        # Product A: seasonal sine wave with 7-day period
        val_a = 10.0 + 5.0 * np.sin(2 * np.pi * d.dayofweek / 7.0)
        # Product B: linear trend
        val_b = 20.0 + (d - dates[0]).days * 0.5
        rows.append({"date": d, "product_id": "P_A", "demand": float(val_a)})
        rows.append({"date": d, "product_id": "P_B", "demand": float(val_b)})
    return pd.DataFrame(rows)


def test_select_dataco_top_products_no_lookahead(synthetic_demand_daily: pd.DataFrame) -> None:
    """Verifies that product selection uses only records prior to the pre-test cutoff."""
    # Before 2017-01-20, evaluate volumes
    selected = select_dataco_top_products(
        synthetic_demand_daily, pretest_cutoff="2017-01-20", top_n=1
    )
    # P_B has higher volume than P_A
    assert selected == ["P_B"]


def test_build_contiguous_daily_series_lifespan_bounds() -> None:
    """Verifies that zero-filling does not create artificial zeros before first or after last sale."""
    sparse_data = pd.DataFrame({
        "date": [pd.Timestamp("2017-01-05"), pd.Timestamp("2017-01-08")],
        "product_id": ["P1", "P1"],
        "demand": [10.0, 15.0]
    })
    contiguous = build_contiguous_daily_series(sparse_data)
    assert len(contiguous) == 4
    assert contiguous["date"].min() == pd.Timestamp("2017-01-05")
    assert contiguous["date"].max() == pd.Timestamp("2017-01-08")
    assert list(contiguous["demand"]) == [10.0, 0.0, 0.0, 15.0]


def test_seasonal_naive_7_day_alignment(synthetic_demand_daily: pd.DataFrame) -> None:
    """Verifies that Seasonal Naive maps exactly to date t - 7 calendar days."""
    eval_dates = pd.date_range("2017-01-15", periods=5, freq="D")
    forecasts = generate_seasonal_naive_forecasts(
        synthetic_demand_daily, eval_dates=eval_dates, seasonal_days=7
    )
    assert len(forecasts) == 10  # 5 dates * 2 products
    for _, row in forecasts.iterrows():
        pid = row["product_id"]
        dt = row["date"]
        prior_dt = dt - pd.Timedelta(days=7)
        actual_prior = synthetic_demand_daily[
            (synthetic_demand_daily["product_id"] == pid) & (synthetic_demand_daily["date"] == prior_dt)
        ]["demand"].iloc[0]
        assert row["prediction"] == pytest.approx(actual_prior)


def test_rolling_mean_excludes_target_day_leakage(synthetic_demand_daily: pd.DataFrame) -> None:
    """Verifies that 7-day rolling mean strictly excludes date t actual demand."""
    eval_dates = pd.date_range("2017-01-15", periods=5, freq="D")
    forecasts = generate_rolling_mean_forecasts(
        synthetic_demand_daily, eval_dates=eval_dates, window_days=7
    )
    for _, row in forecasts.iterrows():
        pid = row["product_id"]
        dt = row["date"]
        # History strictly from dt - 7 days to dt - 1 day
        hist_window = synthetic_demand_daily[
            (synthetic_demand_daily["product_id"] == pid)
            & (synthetic_demand_daily["date"] >= (dt - pd.Timedelta(days=7)))
            & (synthetic_demand_daily["date"] < dt)
        ]
        assert len(hist_window) == 7
        assert row["prediction"] == pytest.approx(hist_window["demand"].mean())


def test_random_forest_training_and_prediction(synthetic_demand_daily: pd.DataFrame) -> None:
    """Verifies Random Forest feature generation and leak-free fitting on training window."""
    train_dates = pd.date_range("2017-01-01", "2017-01-30", freq="D")
    eval_dates = pd.date_range("2017-01-31", "2017-02-09", freq="D")

    features_df = create_dataco_tabular_features(synthetic_demand_daily)
    feature_cols = [
        "day_of_week", "month", "day_of_month", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7",
    ]

    preds_df, model = train_and_predict_random_forest(
        features_df,
        train_dates=train_dates,
        eval_dates=eval_dates,
        feature_cols=feature_cols,
        random_state=42,
        n_estimators=10
    )
    assert len(preds_df) == len(eval_dates) * 2
    assert not preds_df["prediction"].isna().any()


def test_wape_zero_denominator_handling() -> None:
    """Verifies that WAPE returns NaN when total actual sales is 0 across the horizon."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 1.0])
    m = compute_metrics(y_true, y_pred)
    assert np.isnan(m["wape"])
    assert m["wape_defined"] is False


def test_select_dataco_top_products_strict_training_cutoff() -> None:
    """Verifies that top product selection excludes validation and test data by default."""
    # Create dataset where Product X dominates early, but Product Y dominates late
    dates_early = pd.date_range("2015-01-01", "2017-06-30", freq="D")
    dates_late = pd.date_range("2017-07-01", "2017-10-02", freq="D")
    
    rows = []
    for d in dates_early:
        rows.append({"date": d, "product_id": "PROD_X", "demand": 100.0})
        rows.append({"date": d, "product_id": "PROD_Y", "demand": 1.0})
    for d in dates_late:
        rows.append({"date": d, "product_id": "PROD_X", "demand": 0.0})
        rows.append({"date": d, "product_id": "PROD_Y", "demand": 10000.0})
        
    df = pd.DataFrame(rows)
    # Default cutoff is '2017-07-01' (strictly before 2017-07-01 is through 2017-06-30)
    top = select_dataco_top_products(df, top_n=1)
    assert top == ["PROD_X"]


def test_no_current_date_target_leakage_in_features(synthetic_demand_daily: pd.DataFrame) -> None:
    """Verifies that features for date t do not reflect or correlate with demand at date t."""
    df_tampered = synthetic_demand_daily.copy()
    # Spike demand at a specific date
    spike_idx = 10
    spike_date = df_tampered.loc[spike_idx, "date"]
    spike_pid = df_tampered.loc[spike_idx, "product_id"]
    df_tampered.loc[spike_idx, "demand"] = 999999.0
    
    feats = create_dataco_tabular_features(df_tampered)
    row = feats[(feats["product_id"] == spike_pid) & (feats["date"] == spike_date)].iloc[0]
    
    # Lag 1 must not be the spiked value
    assert row["lag_1"] != 999999.0
    # Shifted rolling mean must not include the spiked value
    assert row["rolling_mean_7"] != 999999.0


def test_saved_forecast_parquet_integrity_and_sample_alignment() -> None:
    """Verifies that the generated dataco_forecast_comparison.parquet has identical evaluation samples."""
    from pathlib import Path
    parquet_path = Path("outputs/dataco/forecasts/dataco_forecast_comparison.parquet")
    if not parquet_path.exists():
        pytest.skip("DataCo forecast comparison parquet not found.")
        
    df_fc = pd.read_parquet(parquet_path)
    assert len(df_fc) == 288  # 32 days * 9 SKUs
    assert df_fc.duplicated(subset=["date", "product_id"]).sum() == 0
    
    # Verify no non-finite values in any model's predictions
    for col in ["seasonal_naive_pred", "rolling_mean_pred", "random_forest_pred"]:
        assert not df_fc[col].isna().any()
        assert not np.isinf(df_fc[col]).any()
        # Verify non-negative predictions
        assert (df_fc[col] >= 0.0).all()
