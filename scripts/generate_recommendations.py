"""Inventory replenishment recommendation generator script.

Executes the Phase 6 inventory optimization workflow:
1. Ingests out-of-sample forecasts from Phase 5 (outputs/forecasts/forecast_comparison.parquet).
2. Estimates forecast error uncertainty (daily standard deviation) per product with catalog fallback.
3. Applies configured inventory control parameters (service level, lead time, review period).
4. Sizing safety stock, dynamic reorder points (ROP), target stock (S), and order quantities (Q).
5. Persists comprehensive inventory recommendations and filtered replenishment purchase orders.

Operational Boundary Disclaimer:
All stock levels (on-hand, on-order, backorders), lead times, and operational costs
are SIMULATED parameters. The underlying UCI dataset does not record real warehouse
inventory or supplier contracts.

Usage:
    python scripts/generate_recommendations.py --config config/config.yaml
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from retail_forecasting.inventory.reorder_recommendations import (
    generate_inventory_recommendations,
    generate_replenishment_orders,
    estimate_forecast_uncertainty,
)
from retail_forecasting.utils.logging_config import setup_logger

logger = setup_logger("generate_recommendations")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate inventory replenishment recommendations.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--forecasts",
        type=str,
        default="outputs/forecasts/forecast_comparison.parquet",
        help="Path to out-of-sample forecast comparison parquet.",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="outputs/inventory/inventory_recommendations.parquet",
        help="Path to save full recommendations parquet table.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/inventory/inventory_recommendations.csv",
        help="Path to save full recommendations CSV table.",
    )
    parser.add_argument(
        "--output-orders",
        type=str,
        default="outputs/inventory/replenishment_orders.csv",
        help="Path to save filtered purchase order recommendations CSV.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration YAML if pyyaml is installed, else return empty dict."""
    try:
        import yaml
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    except ImportError:
        logger.info("PyYAML not installed; operating with standard default configuration.")
    return {}


def main() -> None:
    """Execute end-to-end inventory optimization and replenishment generation routine."""
    args = parse_args()
    forecasts_path = Path(args.forecasts)

    if not forecasts_path.exists():
        logger.error("Forecast comparison file not found at: %s", forecasts_path.resolve())
        sys.exit(1)

    logger.info("Loading configuration from: %s", args.config)
    config = load_config(args.config)

    # Extract inventory configuration parameters with sensible defaults
    inv_cfg = config.get("inventory", {})
    service_level = float(inv_cfg.get("target_service_level", inv_cfg.get("default_service_level", 0.95)))
    lead_time_days = int(inv_cfg.get("lead_time_days", inv_cfg.get("default_lead_time_days", 7)))
    review_period_days = int(inv_cfg.get("review_period_days", 1))
    simulated_order_cost = float(inv_cfg.get("simulated_order_cost", inv_cfg.get("order_cost_per_order", 50.0)))
    simulated_holding_cost = float(inv_cfg.get("simulated_holding_cost_annual_per_unit", 2.0))
    random_seed = int(config.get("project", {}).get("random_seed", 42))

    logger.info("Loading out-of-sample forecast comparisons from: %s", forecasts_path)
    t0 = time.time()
    forecast_df = pd.read_parquet(forecasts_path)
    logger.info(
        "Loaded %s forecast rows across %s unique products in %.2fs",
        len(forecast_df),
        forecast_df["product_id"].nunique(),
        time.time() - t0,
    )

    logger.info(
        "Executing inventory optimization engine:\n"
        "  Target Service Level: %.2f%%\n"
        "  Lead Time: %s days\n"
        "  Review Period: %s days (Protection Period: %s days)\n"
        "  Simulated Order Cost: $%.2f\n"
        "  Simulated Holding Cost/Unit/Yr: $%.2f",
        service_level * 100.0,
        lead_time_days,
        review_period_days,
        lead_time_days + review_period_days,
        simulated_order_cost,
        simulated_holding_cost,
    )

    t1 = time.time()
    recommendations_df = generate_inventory_recommendations(
        forecast_df=forecast_df,
        service_level=service_level,
        lead_time_days=lead_time_days,
        review_period_days=review_period_days,
        simulated_order_cost=simulated_order_cost,
        simulated_holding_cost=simulated_holding_cost,
        pred_col="random_forest_pred",
        actual_col="actual_demand",
        product_col="product_id",
        date_col="date",
        rounding_mode="ceil",
        as_of_date="2011-11-11",
        is_retrospective_demonstration=True,
        random_state=random_seed,
    )
    duration = time.time() - t1
    logger.info(
        "Generated inventory recommendations in %.2fs (Retrospective backtest demonstration as of 2011-11-11)",
        duration,
    )

    # Filter to active replenishment orders
    orders_df = generate_replenishment_orders(recommendations_df)

    total_prods = len(recommendations_df)
    orders_count = len(orders_df)
    order_ratio = (orders_count / total_prods * 100.0) if total_prods > 0 else 0.0
    total_order_units = int(orders_df["suggested_order_qty"].sum()) if not orders_df.empty else 0

    logger.info(
        "Inventory Recommendations Summary:\n"
        "  Total Evaluated Products: %s\n"
        "  Products Triggering Replenishment (IP <= ROP): %s (%.2f%%)\n"
        "  Total Recommended Order Units: %s units\n"
        "  Average Safety Stock Buffer: %.2f units\n"
        "  Average Reorder Point: %.2f units",
        total_prods,
        orders_count,
        order_ratio,
        total_order_units,
        recommendations_df["safety_stock"].mean(),
        recommendations_df["reorder_point"].mean(),
    )

    # Persist outputs
    out_parquet = Path(args.output_parquet)
    out_csv = Path(args.output_csv)
    out_orders = Path(args.output_orders)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_orders.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving full recommendations to: %s and %s", out_parquet, out_csv)
    recommendations_df.to_parquet(out_parquet, index=False)
    recommendations_df.to_csv(out_csv, index=False)

    logger.info("Saving active replenishment orders to: %s", out_orders)
    orders_df.to_csv(out_orders, index=False)

    logger.info("Phase 6 inventory optimization workflow completed successfully!")


if __name__ == "__main__":
    main()
