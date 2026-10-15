"""Execution script to generate DataCo inventory replenishment recommendations.

Audit & Correction Updates:
- Multi-step recursive forecasting from fixed origin (2017-08-31) for 8 days (2017-09-01 to 2017-09-08).
- Strictly zero future actuals consumed in forecast horizon generation.
- Uncertainty estimated strictly from pre-decision validation residuals (2017-07-01 to 2017-08-31).
- Dedicated artifact outputs for fixed-origin forecasts, validation forecasts, recommendations, and orders.
"""

import sys
import time
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from retail_forecasting.forecasting.benchmark_dataco import (
    DATACO_CORE_9_PRODUCT_IDS,
    build_contiguous_daily_series,
    generate_dataco_fixed_origin_forecasts,
    generate_dataco_validation_forecasts,
    SPLIT_TRAIN_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_START,
    SPLIT_VAL_END,
)
from retail_forecasting.inventory.reorder_recommendations_dataco import (
    estimate_dataco_forecast_uncertainty,
    generate_dataco_inventory_recommendations,
)

CONFIG_PATH = Path("config/dataco_inventory_config.yaml")

DEFAULT_CONFIG = {
    "paths": {
        "demand_file": "data/processed/dataco_daily_product_demand.parquet",
        "product_catalog_file": "data/processed/dataco_product_catalog.parquet",
        "forecast_comparison_file": "outputs/dataco/forecasts/dataco_forecast_comparison.parquet",
        "validation_forecasts_file": "outputs/dataco/forecasts/dataco_validation_forecasts.parquet",
        "fixed_origin_forecasts_file": "outputs/dataco/inventory/dataco_fixed_origin_forecasts.parquet",
        "output_recommendations_parquet": "outputs/dataco/inventory/dataco_inventory_recommendations.parquet",
        "output_recommendations_csv": "outputs/dataco/inventory/dataco_replenishment_orders.csv",
    },
    "inventory_policy": {
        "service_level_target": 0.95,
        "assumed_supplier_lead_time_days": 7,
        "review_period_days": 1,
        "rounding_mode": "ceil",
        "simulated_default_on_hand": 50.0,
        "simulated_default_on_order": 0.0,
        "simulated_default_backorders": 0.0,
        "simulated_order_cost_fixed": 50.0,
        "simulated_holding_cost_annual_per_unit": 2.0,
    },
    "forecast_contract": {
        "selected_model": "Random Forest",
        "forecast_horizon_days": 8,
        "forecast_origin_date": "2017-08-31",
        "forecast_method": "recursive_multi_step",
        "residual_cutoff_date": "2017-08-31",
        "uncertainty_method": "sample_std",
        "min_observations_uncertainty": 14,
    }
}


def load_config(config_path: Path) -> dict:
    """Load configuration YAML if pyyaml is installed, else return default dict."""
    try:
        import yaml
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    return loaded
    except ImportError:
        pass
    return DEFAULT_CONFIG


def main() -> None:
    print("=" * 75)
    print("DATACO INVENTORY OPTIMIZATION RECOMMENDATION GENERATOR (PHASE 9)")
    print("=" * 75)
    t0 = time.time()

    # 1. Load config
    cfg = load_config(CONFIG_PATH)

    paths_cfg = cfg["paths"]
    policy_cfg = cfg["inventory_policy"]
    fc_cfg = cfg["forecast_contract"]

    demand_path = Path(paths_cfg.get("demand_file", "data/processed/dataco_daily_product_demand.parquet"))
    cat_path = Path(paths_cfg["product_catalog_file"])
    val_fc_path = Path(paths_cfg.get("validation_forecasts_file", "outputs/dataco/forecasts/dataco_validation_forecasts.parquet"))
    fixed_fc_path = Path(paths_cfg.get("fixed_origin_forecasts_file", "outputs/dataco/inventory/dataco_fixed_origin_forecasts.parquet"))
    out_parquet = Path(paths_cfg["output_recommendations_parquet"])
    out_csv = Path(paths_cfg["output_recommendations_csv"])

    print(f"Loading processed demand from {demand_path}...")
    demand_df = pd.read_parquet(demand_path)
    print(f"Loading product catalog from {cat_path}...")
    catalog_df = pd.read_parquet(cat_path)

    # 2. Build contiguous series for Core 9
    contiguous_df = build_contiguous_daily_series(
        demand_df, product_ids=DATACO_CORE_9_PRODUCT_IDS
    )
    print(f"Built contiguous daily series for Core 9: {len(contiguous_df):,} rows.")

    # 3. Generate or load validation forecasts (strictly <= 2017-08-31)
    train_dates = pd.date_range(start=SPLIT_TRAIN_START, end=SPLIT_TRAIN_END, freq="D")
    val_dates = pd.date_range(start=SPLIT_VAL_START, end=SPLIT_VAL_END, freq="D")

    print(f"\nStep 1: Generating leak-free pre-decision validation forecasts ({val_dates.min().date()} to {val_dates.max().date()})...")
    val_preds_df = generate_dataco_validation_forecasts(
        contiguous_df=contiguous_df,
        train_dates=train_dates,
        val_dates=val_dates,
        product_ids=DATACO_CORE_9_PRODUCT_IDS,
        random_state=42,
    )
    val_fc_path.parent.mkdir(parents=True, exist_ok=True)
    val_preds_df.to_parquet(val_fc_path, index=False)
    print(f"Saved validation forecasts: {val_fc_path} ({len(val_preds_df)} rows, max date: {val_preds_df['date'].max().date()}).")

    # 4. Estimate forecast uncertainty strictly on pre-decision residuals (<= 2017-08-31)
    print(f"\nStep 2: Estimating residual uncertainty with cutoff date {fc_cfg['residual_cutoff_date']} ({fc_cfg['uncertainty_method']})...")
    uncertainty_df = estimate_dataco_forecast_uncertainty(
        historical_forecasts_df=val_preds_df,
        forecast_origin_date=fc_cfg["residual_cutoff_date"],
        pred_col="prediction",
        actual_col="actual_demand",
        min_observations=int(fc_cfg.get("min_observations_uncertainty", 14)),
        method=fc_cfg.get("uncertainty_method", "sample_std"),
    )
    print(uncertainty_df[["product_id", "sample_size", "error_mean", "error_std", "error_rmse", "uncertainty_value", "uncertainty_status"]].to_string(index=False))

    # 5. Generate fixed-origin multi-step recursive forecasts
    origin_date = fc_cfg["forecast_origin_date"]
    horizon_days = int(fc_cfg["forecast_horizon_days"])
    print(f"\nStep 3: Generating fixed-origin recursive forecasts (origin: {origin_date}, horizon: {horizon_days} days)...")
    fixed_origin_forecasts_df = generate_dataco_fixed_origin_forecasts(
        contiguous_df=contiguous_df,
        forecast_origin=origin_date,
        horizon_days=horizon_days,
        product_ids=DATACO_CORE_9_PRODUCT_IDS,
        train_dates=train_dates,
        random_state=42,
    )
    fixed_fc_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_origin_forecasts_df.to_parquet(fixed_fc_path, index=False)
    print(f"Saved fixed-origin forecasts: {fixed_fc_path} ({len(fixed_origin_forecasts_df)} rows).")

    # Display forecast sample
    sample_pid = DATACO_CORE_9_PRODUCT_IDS[0]
    p_fc = fixed_origin_forecasts_df[fixed_origin_forecasts_df["product_id"] == sample_pid]
    print(f"\nForecast horizon preview for SKU {sample_pid}:")
    print(p_fc[["forecast_horizon_step", "forecast_date", "predicted_demand"]].to_string(index=False))

    # 6. Generate inventory recommendations
    print(f"\nStep 4: Computing inventory recommendations under CSL = {policy_cfg['service_level_target']*100:.0f}%, "
          f"L = {policy_cfg['assumed_supplier_lead_time_days']}d, R = {policy_cfg['review_period_days']}d...")

    recs_df = generate_dataco_inventory_recommendations(
        forecast_comparison_df=fixed_origin_forecasts_df,
        catalog_df=catalog_df,
        uncertainty_df=uncertainty_df,
        service_level_target=float(policy_cfg["service_level_target"]),
        lead_time_days=int(policy_cfg["assumed_supplier_lead_time_days"]),
        review_period_days=int(policy_cfg["review_period_days"]),
        simulated_default_on_hand=float(policy_cfg["simulated_default_on_hand"]),
        simulated_default_on_order=float(policy_cfg["simulated_default_on_order"]),
        simulated_default_backorders=float(policy_cfg["simulated_default_backorders"]),
        simulated_order_cost_fixed=float(policy_cfg["simulated_order_cost_fixed"]),
        simulated_holding_cost_annual_per_unit=float(policy_cfg["simulated_holding_cost_annual_per_unit"]),
        rounding_mode=policy_cfg["rounding_mode"],
        forecast_origin_date=origin_date,
        forecast_horizon_days=horizon_days,
        pred_col="predicted_demand",
        date_col="forecast_date",
        uncertainty_method=fc_cfg.get("uncertainty_method", "sample_std"),
    )

    print("\n--- GENERATED DATACO REORDER RECOMMENDATIONS ---")
    cols_to_show = [
        "product_id", "product_name", "forecast_origin", "expected_lead_time_demand",
        "expected_protection_period_demand", "safety_stock", "reorder_point",
        "target_stock_level", "inventory_position", "recommended_order_quantity",
        "reorder_triggered", "optional_eoq_benchmark"
    ]
    print(recs_df[cols_to_show].to_string(index=False))

    # 7. Save recommendation artifacts
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    recs_df.to_parquet(out_parquet, index=False)
    print(f"\nSaved recommendations parquet: {out_parquet} ({out_parquet.stat().st_size:,} bytes)")

    # Save actionable replenishment orders (reorder_triggered == True)
    reorder_orders = recs_df[recs_df["reorder_triggered"]].copy()
    reorder_orders.to_csv(out_csv, index=False)
    print(f"Saved replenishment orders CSV: {out_csv} ({len(reorder_orders)} orders triggered)")

    print(f"\nCompleted inventory recommendation pipeline in {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    main()
