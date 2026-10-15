"""Execution script for DataCo forecasting benchmark experiments."""

import json
import sys
import time
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from retail_forecasting.forecasting.benchmark_dataco import (
    DATACO_CORE_9_PRODUCT_IDS,
    select_dataco_top_products,
    build_contiguous_daily_series,
    generate_seasonal_naive_forecasts,
    generate_rolling_mean_forecasts,
    create_dataco_tabular_features,
    train_and_predict_random_forest,
    SPLIT_TRAIN_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_START,
    SPLIT_VAL_END,
    SPLIT_TEST_START,
    SPLIT_TEST_END,
)
from retail_forecasting.forecasting.evaluate import compute_metrics, compute_per_product_metrics

DEMAND_PATH = Path("data/processed/dataco_daily_product_demand.parquet")
CATALOG_PATH = Path("data/processed/dataco_product_catalog.parquet")

OUTPUT_FORECASTS_DIR = Path("outputs/dataco/forecasts")
OUTPUT_METRICS_DIR = Path("outputs/dataco/metrics")
OUTPUT_FORECASTS_PATH = OUTPUT_FORECASTS_DIR / "dataco_forecast_comparison.parquet"
OUTPUT_METRICS_PATH = OUTPUT_METRICS_DIR / "dataco_forecast_metrics.json"


def main() -> None:
    print("=" * 70)
    print("DATACO DAILY FORECASTING BENCHMARK (PHASE 8C)")
    print("=" * 70)
    t0 = time.time()

    # 1. Load processed demand and catalog
    demand_df = pd.read_parquet(DEMAND_PATH)
    catalog_df = pd.read_parquet(CATALOG_PATH)
    print(f"Loaded processed demand: {len(demand_df):,} records, {demand_df['product_id'].nunique()} SKUs.")

    # 2. Select top 9 products strictly using training-period data (before 2017-07-01)
    selected_pids = select_dataco_top_products(
        demand_df, pretest_cutoff="2017-07-01", top_n=9
    )
    print(f"Selected Core 9 Products (Training-period volume through 2017-06-30): {selected_pids}")
    assert set(selected_pids) == set(DATACO_CORE_9_PRODUCT_IDS), "Cohort mismatch!"

    # 3. Construct contiguous daily series with lifespan zero-filling
    contiguous_df = build_contiguous_daily_series(demand_df, product_ids=selected_pids)
    print(f"Constructed contiguous daily series for Core 9: {len(contiguous_df):,} records.")
    for pid in selected_pids:
        sub = contiguous_df[contiguous_df["product_id"] == pid]
        print(f"  SKU {pid:4s}: {len(sub):,} days ({sub['date'].min().date()} to {sub['date'].max().date()}), zero-days = {(sub['demand']==0).sum()}")

    # 4. Define evaluation windows
    train_dates = pd.date_range(start=SPLIT_TRAIN_START, end=SPLIT_TRAIN_END, freq="D")
    val_dates = pd.date_range(start=SPLIT_VAL_START, end=SPLIT_VAL_END, freq="D")
    test_dates = pd.date_range(start=SPLIT_TEST_START, end=SPLIT_TEST_END, freq="D")

    print(f"\nChronological Split Configuration:")
    print(f"  Training Window  : {train_dates.min().date()} to {train_dates.max().date()} ({len(train_dates)} days)")
    print(f"  Validation Window: {val_dates.min().date()} to {val_dates.max().date()} ({len(val_dates)} days)")
    print(f"  Test Window      : {test_dates.min().date()} to {test_dates.max().date()} ({len(test_dates)} days)")

    # 5. Generate forecasts on Validation and Test windows
    # Baseline A: Seasonal Naive
    sn_val = generate_seasonal_naive_forecasts(contiguous_df, eval_dates=val_dates)
    sn_test = generate_seasonal_naive_forecasts(contiguous_df, eval_dates=test_dates)

    # Baseline B: Rolling Mean (7-day shifted)
    rm_val = generate_rolling_mean_forecasts(contiguous_df, eval_dates=val_dates, window_days=7)
    rm_test = generate_rolling_mean_forecasts(contiguous_df, eval_dates=test_dates, window_days=7)

    # Benchmark C: Random Forest
    feature_cols = [
        "day_of_week", "month", "day_of_month", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        "rolling_mean_28", "rolling_std_28",
    ]
    features_df = create_dataco_tabular_features(contiguous_df)
    rf_val, _ = train_and_predict_random_forest(
        features_df, train_dates=train_dates, eval_dates=val_dates, feature_cols=feature_cols
    )
    # Fit on train + val for test evaluation, or strictly train
    rf_test, fitted_rf = train_and_predict_random_forest(
        features_df, train_dates=train_dates, eval_dates=test_dates, feature_cols=feature_cols
    )

    print("\nForecasts generated successfully across all three models.")

    # 6. Evaluation metrics on identical test observations
    # Confirm identical product-date rows
    test_pairs = test_dates.size * len(selected_pids)
    print(f"Total test product-date observations: {test_pairs} (32 days * 9 SKUs = 288 rows)")

    models = {
        "Seasonal Naive": sn_test,
        "Rolling Mean (7d)": rm_test,
        "Random Forest": rf_test,
    }

    metrics_summary: Dict[str, Any] = {
        "metadata": {
            "dataset": "DataCo Smart Supply Chain",
            "demand_policy": "Gross Placed Retail Demand (excl. CANCELED & SUSPECTED_FRAUD)",
            "cohort": selected_pids,
            "train_window": [SPLIT_TRAIN_START, SPLIT_TRAIN_END],
            "validation_window": [SPLIT_VAL_START, SPLIT_VAL_END],
            "test_window": [SPLIT_TEST_START, SPLIT_TEST_END],
            "evaluation_protocol": "Rolling 1-day-ahead out-of-sample forecast",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pooled_metrics": {},
        "per_product_metrics": {},
    }

    comparison_frames = []

    print("\n--- MODEL ACCURACY ON IDENTICAL TEST OBSERVATIONS (2017-09-01 to 2017-10-02) ---")
    print(f"{'Model Name':<20s} | {'Sample':<7s} | {'MAE':<8s} | {'RMSE':<8s} | {'WAPE (%)':<10s} | {'Unavail':<7s}")
    print("-" * 75)

    for name, pred_df in models.items():
        # Check unavailable predictions
        unavail_cnt = int(pred_df["prediction"].isna().sum())
        clean_df = pred_df.dropna(subset=["prediction", "actual_demand"])

        m = compute_metrics(clean_df["actual_demand"], clean_df["prediction"])
        prod_metrics_df, macro_summary = compute_per_product_metrics(clean_df)

        metrics_summary["pooled_metrics"][name] = {
            "sample_size": m["sample_size"],
            "unavailable_predictions": unavail_cnt,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "wape": m["wape"],
            "macro_mae": macro_summary["macro_mae"],
            "macro_rmse": macro_summary["macro_rmse"],
            "macro_wape": macro_summary["macro_wape"],
        }
        metrics_summary["per_product_metrics"][name] = prod_metrics_df.to_dict(orient="records")

        print(f"{name:<20s} | {m['sample_size']:<7d} | {m['mae']:<8.4f} | {m['rmse']:<8.4f} | {m['wape']:<10.2f}% | {unavail_cnt:<7d}")

        # Prepare comparison frame
        formatted_df = clean_df.copy()
        formatted_df["model_name"] = name
        comparison_frames.append(formatted_df)

    # 7. Merge into comparison parquet artifact
    full_comparison_df = pd.merge(
        sn_test.rename(columns={"prediction": "seasonal_naive_pred"}),
        rm_test[["date", "product_id", "prediction"]].rename(columns={"prediction": "rolling_mean_pred"}),
        on=["date", "product_id"]
    )
    full_comparison_df = pd.merge(
        full_comparison_df,
        rf_test[["date", "product_id", "prediction"]].rename(columns={"prediction": "random_forest_pred"}),
        on=["date", "product_id"]
    )

    # Attach product metadata (name, price, department)
    full_comparison_df = pd.merge(
        full_comparison_df,
        catalog_df[["product_id", "product_name", "category_name", "department_name", "product_price"]],
        on="product_id",
        how="left"
    )

    # 8. Save output artifacts
    OUTPUT_FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    full_comparison_df.to_parquet(OUTPUT_FORECASTS_PATH, index=False)
    print(f"\nSaved forecast comparison parquet: {OUTPUT_FORECASTS_PATH} ({OUTPUT_FORECASTS_PATH.stat().st_size:,} bytes)")

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved forecast metrics json: {OUTPUT_METRICS_PATH} ({OUTPUT_METRICS_PATH.stat().st_size:,} bytes)")

    print(f"\nBenchmark completed successfully in {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    main()
