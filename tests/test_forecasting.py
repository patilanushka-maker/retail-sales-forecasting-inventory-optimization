"""Unit and integration tests for Phase 5 forecasting.

Verifies:
1. Chronological splitting.
2. No random row splitting.
3. Common cutoff across products.
4. No train/test target overlap.
5. Correct seven-calendar-day seasonal-naive lookup.
6. Baseline predictions are calculated separately by product.
7. Missing historical baseline values are handled explicitly (retained as NaN).
8. Correct MAE computation.
9. Correct RMSE computation.
10. Correct WAPE computation.
11. Zero-denominator WAPE behavior (returns NaN with wape_defined=False).
12. Empty metric input handling.
13. Non-finite metric inputs rejection (raises ValueError).
14. Reproducible Random Forest training with fixed random_state.
15. Exclusion of target demand from model features.
16. Non-negative prediction clipping policy (clips negative values to 0.0).
17. Baseline and Random Forest use identical evaluation rows.
18. Rolling evaluation does not use future target values.
19. Test-period actuals are not used to fit the model.
20. Output forecast schema correctness.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from retail_forecasting.features.build_features import build_feature_pipeline
from retail_forecasting.forecasting.baseline import (
    SeasonalNaiveForecaster,
    generate_seasonal_naive_predictions,
)
from retail_forecasting.forecasting.evaluate import (
    compute_metrics,
    compute_per_product_metrics,
)
from retail_forecasting.forecasting.predict import generate_forecasts
from retail_forecasting.forecasting.train import (
    DEFAULT_FEATURE_COLS,
    run_rolling_evaluation,
    temporal_train_test_split,
    train_random_forest,
)


@pytest.fixture
def synthetic_multiproduct_ts() -> pd.DataFrame:
    """Generate a clean 60-day synthetic dataset for 2 products with weekly seasonality."""
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    records = []
    for dt in dates:
        day_idx = dt.dayofweek
        # Product A has baseline 10 + day_idx
        demand_a = float(10 + (day_idx * 2))
        records.append({"date": dt, "product_id": "SKU_A", "demand": demand_a})
        # Product B has baseline 50 + day_idx
        demand_b = float(50 + (day_idx * 5))
        records.append({"date": dt, "product_id": "SKU_B", "demand": demand_b})

    df = pd.DataFrame(records)
    return df


@pytest.fixture
def synthetic_feature_df(synthetic_multiproduct_ts: pd.DataFrame) -> pd.DataFrame:
    """Generate mock feature dataset matching the pipeline contract."""
    df = synthetic_multiproduct_ts.copy()
    # Add calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Add mock lag and rolling features
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby("product_id")["demand"].shift(lag)

    for w in [7, 14, 28]:
        shifted = df.groupby("product_id")["demand"].shift(1)
        df[f"rolling_mean_{w}"] = (
            shifted.groupby(df["product_id"]).rolling(w, min_periods=w).mean().reset_index(drop=True)
        )
        df[f"rolling_std_{w}"] = (
            shifted.groupby(df["product_id"]).rolling(w, min_periods=w).std().reset_index(drop=True)
        )

    # Eligibility flag: valid lag_28
    df["is_model_eligible"] = df["lag_28"].notna()
    return df


# 1. Chronological splitting
def test_chronological_splitting(synthetic_multiproduct_ts: pd.DataFrame) -> None:
    """Test that split places all dates <= cutoff in train and > cutoff in test."""
    train_df, test_df, report = temporal_train_test_split(
        synthetic_multiproduct_ts, date_col="date", test_days=14
    )
    assert not train_df.empty
    assert not test_df.empty
    cutoff = pd.to_datetime(report["cutoff_date"])
    assert train_df["date"].max() <= cutoff
    assert test_df["date"].min() > cutoff


# 2. No random row splitting
def test_no_random_row_splitting(synthetic_multiproduct_ts: pd.DataFrame) -> None:
    """Verify that multiple split invocations produce identical deterministic partitions."""
    train1, test1, _ = temporal_train_test_split(synthetic_multiproduct_ts, test_days=14)
    train2, test2, _ = temporal_train_test_split(synthetic_multiproduct_ts, test_days=14)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


# 3. Common cutoff across products
def test_common_cutoff_across_products(synthetic_multiproduct_ts: pd.DataFrame) -> None:
    """Verify that the cutoff date is identical for all product partitions."""
    train_df, test_df, report = temporal_train_test_split(
        synthetic_multiproduct_ts, date_col="date", test_days=20, product_col="product_id"
    )
    cutoff = pd.to_datetime(report["cutoff_date"])
    for pid, group in train_df.groupby("product_id"):
        assert group["date"].max() <= cutoff
    for pid, group in test_df.groupby("product_id"):
        assert group["date"].min() > cutoff


# 4. No train/test target overlap
def test_no_train_test_target_overlap(synthetic_multiproduct_ts: pd.DataFrame) -> None:
    """Verify that train and test date sets are strictly disjoint."""
    train_df, test_df, _ = temporal_train_test_split(
        synthetic_multiproduct_ts, date_col="date", test_days=14
    )
    train_dates = set(train_df["date"].unique())
    test_dates = set(test_df["date"].unique())
    assert train_dates.isdisjoint(test_dates)


# 5. Correct seven-calendar-day seasonal-naive lookup
def test_correct_seven_calendar_day_seasonal_naive_lookup() -> None:
    """Verify exact 7-calendar-day lookup rather than row index shift."""
    # Create irregular date sequence: missing 2021-01-02
    dates = pd.to_datetime(["2021-01-01", "2021-01-03", "2021-01-08"])
    df = pd.DataFrame({
        "date": dates,
        "product_id": ["P1", "P1", "P1"],
        "demand": [100.0, 200.0, 300.0]
    })
    forecaster = SeasonalNaiveForecaster(seasonal_days=7)
    forecaster.fit(df)
    preds = forecaster.predict(df)
    # On 2021-01-08: 7 calendar days prior is 2021-01-01 (demand = 100.0)
    assert preds[2] == 100.0
    # On 2021-01-01 and 2021-01-03: 7 days prior does not exist, must be NaN
    assert np.isnan(preds[0])
    assert np.isnan(preds[1])


# 6. Baseline predictions are calculated separately by product
def test_baseline_calculated_separately_by_product() -> None:
    """Verify that product A and product B get their own respective historical lookups."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-08", "2021-01-08"]),
        "product_id": ["P_A", "P_B", "P_A", "P_B"],
        "demand": [15.0, 95.0, 20.0, 110.0]
    })
    forecaster = SeasonalNaiveForecaster(seasonal_days=7)
    forecaster.fit(df)
    test_rows = df[df["date"] == pd.to_datetime("2021-01-08")]
    preds = forecaster.predict(test_rows)
    # P_A should get 15.0, P_B should get 95.0
    assert preds[0] == 15.0
    assert preds[1] == 95.0


# 7. Missing historical baseline values are handled explicitly
def test_missing_historical_baseline_handled_explicitly() -> None:
    """Ensure missing historical observations are returned as NaN, never silently zero-filled."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-08"]),
        "product_id": ["P_NEW"],
        "demand": [50.0]
    })
    forecaster = SeasonalNaiveForecaster(seasonal_days=7)
    forecaster.fit(df)
    preds = forecaster.predict(df)
    assert np.isnan(preds[0])
    assert not (preds[0] == 0.0)


# 8. Correct MAE
def test_correct_mae() -> None:
    """Validate MAE calculation against known hand calculation."""
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    # Abs errors: 2, 2, 3 -> sum = 7 -> mean = 7/3 = 2.3333...
    res = compute_metrics(y_true, y_pred)
    assert pytest.approx(res["mae"], rel=1e-3) == 7.0 / 3.0


# 9. Correct RMSE
def test_correct_rmse() -> None:
    """Validate RMSE calculation against known formula."""
    y_true = [10.0, 20.0]
    y_pred = [13.0, 16.0]
    # Errors: 3, 4 -> squared: 9, 16 -> mean: 12.5 -> sqrt(12.5) = 3.5355...
    res = compute_metrics(y_true, y_pred)
    assert pytest.approx(res["rmse"], rel=1e-3) == np.sqrt(12.5)


# 10. Correct WAPE
def test_correct_wape() -> None:
    """Validate WAPE percentage calculation."""
    y_true = [10.0, 20.0]
    y_pred = [12.0, 18.0]
    # Abs errors: 2, 2 -> sum = 4. Total actual = 30 -> WAPE = (4/30)*100 = 13.3333%
    res = compute_metrics(y_true, y_pred)
    assert pytest.approx(res["wape"], rel=1e-3) == (4.0 / 30.0) * 100.0
    assert res["wape_defined"] is True


# 11. Zero-denominator WAPE behavior
def test_zero_denominator_wape_behavior() -> None:
    """Verify that when sum(actual) == 0, WAPE is returned as NaN with wape_defined=False."""
    y_true = [0.0, 0.0, 0.0]
    y_pred = [1.0, 2.0, 1.0]
    res = compute_metrics(y_true, y_pred)
    assert np.isnan(res["wape"])
    assert res["wape_defined"] is False
    assert res["sample_size"] == 3


# 12. Empty metric input
def test_empty_metric_input() -> None:
    """Verify graceful handling of empty inputs."""
    res = compute_metrics([], [])
    assert res["sample_size"] == 0
    assert np.isnan(res["mae"])
    assert np.isnan(res["rmse"])
    assert np.isnan(res["wape"])
    assert res["wape_defined"] is False


# 13. Non-finite metric inputs
def test_non_finite_metric_inputs() -> None:
    """Verify that compute_metrics raises ValueError if inputs contain NaN or Inf."""
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics([10.0, np.nan], [10.0, 12.0])

    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics([10.0, 12.0], [10.0, np.inf])


# 14. Reproducible Random Forest training
def test_reproducible_random_forest_training() -> None:
    """Verify that setting random_state ensures identical predictions."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.randn(50))

    model1 = train_random_forest(X, y, random_state=42)
    model2 = train_random_forest(X, y, random_state=42)

    preds1 = model1.predict(X)
    preds2 = model2.predict(X)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-10, atol=1e-10)


# 15. Exclusion of target demand from model features
def test_exclusion_of_target_demand_from_model_features(synthetic_feature_df: pd.DataFrame) -> None:
    """Verify that target column is rejected if included in feature_cols."""
    bad_features = ["lag_1", "demand"]
    with pytest.raises(ValueError, match="must not be included"):
        run_rolling_evaluation(synthetic_feature_df, feature_cols=bad_features, target_col="demand")


# 16. Non-negative prediction policy
def test_non_negative_prediction_policy() -> None:
    """Verify that generate_forecasts clips negative model predictions to 0.0."""
    class MockNegativeRegressor:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([-5.0, 0.0, 15.0, -0.01])

    X = pd.DataFrame({"f1": [1, 2, 3, 4]})
    preds = generate_forecasts(MockNegativeRegressor(), X, clip_non_negative=True)
    assert (preds >= 0.0).all()
    assert preds[0] == 0.0
    assert preds[1] == 0.0
    assert preds[2] == 15.0
    assert preds[3] == 0.0


# 17. Baseline and Random Forest use the same evaluation rows
def test_baseline_and_rf_same_evaluation_rows(synthetic_feature_df: pd.DataFrame) -> None:
    """Verify that both models are evaluated on identical aligned observations."""
    result = run_rolling_evaluation(
        synthetic_feature_df,
        test_days=14,
        random_state=42
    )
    eval_table = result["evaluation_table"]
    assert not eval_table.empty
    assert eval_table["random_forest_pred"].notna().all()
    assert eval_table["seasonal_naive_pred"].notna().all()
    assert len(eval_table) == result["metrics"]["evaluation_sample_size"]


# 18. Rolling evaluation does not use future target values
def test_rolling_evaluation_no_future_target_leakage(synthetic_feature_df: pd.DataFrame) -> None:
    """Verify that predictions at date t do not change if dates t+1 onward are modified."""
    df1 = synthetic_feature_df.copy()
    result1 = run_rolling_evaluation(df1, test_days=14, random_state=42)
    preds1 = result1["evaluation_table"]["seasonal_naive_pred"].values

    # Alter the last 5 days of test actuals
    df2 = synthetic_feature_df.copy()
    max_date = df2["date"].max()
    altered_cutoff = max_date - pd.Timedelta(days=5)
    df2.loc[df2["date"] > altered_cutoff, "demand"] += 99999.0

    # Predictions before altered_cutoff should be completely unaffected
    result2 = run_rolling_evaluation(df2, test_days=14, random_state=42)
    tbl1 = result1["evaluation_table"]
    tbl2 = result2["evaluation_table"]

    early_mask1 = tbl1["date"] <= altered_cutoff
    early_mask2 = tbl2["date"] <= altered_cutoff

    np.testing.assert_array_equal(
        tbl1.loc[early_mask1, "seasonal_naive_pred"].values,
        tbl2.loc[early_mask2, "seasonal_naive_pred"].values,
    )


# 19. Test-period actuals are not used to fit the model
def test_test_period_actuals_not_used_to_fit_model(synthetic_feature_df: pd.DataFrame) -> None:
    """Verify model parameters are invariant to modifications of test-period targets."""
    df1 = synthetic_feature_df.copy()
    result1 = run_rolling_evaluation(df1, test_days=14, random_state=42)

    df2 = synthetic_feature_df.copy()
    cutoff_date = pd.to_datetime(result1["split_report"]["cutoff_date"])
    # Drastically mutate test targets
    df2.loc[df2["date"] > cutoff_date, "demand"] *= 500.0

    result2 = run_rolling_evaluation(df2, test_days=14, random_state=42)

    # Check that tree predictions on a fixed vector are identical
    sample_X = df1[DEFAULT_FEATURE_COLS].dropna().head(10)
    np.testing.assert_array_equal(
        result1["model"].predict(sample_X),
        result2["model"].predict(sample_X)
    )


# 20. Output forecast schema is correct
def test_output_forecast_schema(synthetic_feature_df: pd.DataFrame) -> None:
    """Verify that the evaluation table conforms to the required contract."""
    result = run_rolling_evaluation(synthetic_feature_df, test_days=14, random_state=42)
    eval_tbl = result["evaluation_table"]
    expected_cols = [
        "date",
        "product_id",
        "actual_demand",
        "random_forest_pred",
        "seasonal_naive_pred",
    ]
    for col in expected_cols:
        assert col in eval_tbl.columns
    assert pd.api.types.is_datetime64_any_dtype(eval_tbl["date"])
    assert pd.api.types.is_numeric_dtype(eval_tbl["actual_demand"])
    assert pd.api.types.is_numeric_dtype(eval_tbl["random_forest_pred"])
    assert pd.api.types.is_numeric_dtype(eval_tbl["seasonal_naive_pred"])


# 21. End-to-end temporal causality: no future leakage and sensitivity to past observations
def test_end_to_end_causality_and_no_future_leakage() -> None:
    """Verify end-to-end pipeline causality:
    1. Modifying future actuals cannot alter earlier predictions for RF or Seasonal Naive.
    2. Modifying a past actual observation changes subsequent model predictions.
    """
    # 70-day synthetic dataset with weekly seasonality across 2 products
    dates = pd.date_range("2021-01-01", periods=70, freq="D")
    records = []
    for dt in dates:
        day_idx = dt.dayofweek
        records.append({"date": dt, "product_id": "SKU_1", "demand": float(10 + (day_idx * 2))})
        records.append({"date": dt, "product_id": "SKU_2", "demand": float(50 + (day_idx * 5))})
    raw_df = pd.DataFrame(records)

    # Base pipeline execution
    feat_base = build_feature_pipeline(raw_df)
    res_base = run_rolling_evaluation(feat_base, test_days=14, random_state=42)
    tbl_base = res_base["evaluation_table"]

    # 1. Mutate future actuals (last 4 days of test horizon)
    max_date = raw_df["date"].max()
    future_cutoff = max_date - pd.Timedelta(days=4)
    raw_future_mutated = raw_df.copy()
    raw_future_mutated.loc[raw_future_mutated["date"] > future_cutoff, "demand"] += 5000.0

    feat_future = build_feature_pipeline(raw_future_mutated)
    res_future = run_rolling_evaluation(feat_future, test_days=14, random_state=42)
    tbl_future = res_future["evaluation_table"]

    # Verify no future leakage on both models
    early_mask = tbl_base["date"] <= future_cutoff
    np.testing.assert_array_equal(
        tbl_base.loc[early_mask, "random_forest_pred"].values,
        tbl_future.loc[early_mask, "random_forest_pred"].values,
        err_msg="Future actuals leaked into earlier Random Forest predictions!",
    )
    np.testing.assert_array_equal(
        tbl_base.loc[early_mask, "seasonal_naive_pred"].values,
        tbl_future.loc[early_mask, "seasonal_naive_pred"].values,
        err_msg="Future actuals leaked into earlier Seasonal Naive predictions!",
    )

    # 2. Mutate past observation at cutoff_date (which is t-1 relative to first test day)
    cutoff_date = pd.to_datetime(res_base["split_report"]["cutoff_date"])
    raw_past_mutated = raw_df.copy()
    raw_past_mutated.loc[raw_past_mutated["date"] == cutoff_date, "demand"] += 1000.0

    feat_past = build_feature_pipeline(raw_past_mutated)
    res_past = run_rolling_evaluation(feat_past, test_days=14, random_state=42)
    tbl_past = res_past["evaluation_table"]

    first_test_day = cutoff_date + pd.Timedelta(days=1)
    mask_day1 = tbl_base["date"] == first_test_day

    rf_base_d1 = tbl_base.loc[mask_day1, "random_forest_pred"].values
    rf_past_d1 = tbl_past.loc[mask_day1, "random_forest_pred"].values
    assert not np.array_equal(rf_base_d1, rf_past_d1), (
        "Random forest prediction at test day 1 failed to reflect changed demand on prior day."
    )

