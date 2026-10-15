"""Model training and evaluation script.

Constructs feature matrices, splits train/test chronologically over a 28-day test horizon,
fits Seasonal Naive baseline and Random Forest regression models, logs metrics (MAE, RMSE, WAPE),
and saves forecast comparisons, metrics, and trained model artifacts.

Usage:
    python scripts/train_model.py --config config/config.yaml
"""

import argparse
import json
from pathlib import Path
import sys
import time
import joblib
import pandas as pd

from retail_forecasting.features.build_features import build_feature_pipeline
from retail_forecasting.forecasting.train import run_rolling_evaluation
from retail_forecasting.utils.logging_config import setup_logger

logger = setup_logger("train_model")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train demand forecasting models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/daily_product_demand.parquet",
        help="Path to processed daily demand parquet file.",
    )
    parser.add_argument(
        "--output-forecasts",
        type=str,
        default="outputs/forecasts/forecast_comparison.parquet",
        help="Path to save out-of-sample forecast comparison parquet.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="outputs/metrics/forecast_metrics.json",
        help="Path to save evaluation metrics JSON.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/random_forest_forecaster.joblib",
        help="Path to save trained Random Forest model artifact.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration YAML if pyyaml is installed, else return default dict."""
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
    """Execute end-to-end model training, rolling evaluation, and persistence routine."""
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input daily demand dataset not found at: %s", input_path.resolve())
        sys.exit(1)

    logger.info("Loading configuration from: %s", args.config)
    config = load_config(args.config)

    # Extract model and forecasting configuration
    forecasting_cfg = config.get("forecasting", {})
    test_days = int(forecasting_cfg.get("test_size_days", 28))
    rf_cfg = forecasting_cfg.get("random_forest", {})
    n_estimators = int(rf_cfg.get("n_estimators", 100))
    max_depth = int(rf_cfg.get("max_depth", 10))
    random_state = int(rf_cfg.get("random_state", config.get("project", {}).get("random_seed", 42)))

    logger.info("Loading processed daily product demand from: %s", input_path)
    t0 = time.time()
    demand_df = pd.read_parquet(input_path)
    logger.info(
        "Loaded daily demand dataset: %s rows across %s products in %.2fs",
        len(demand_df),
        demand_df["product_id"].nunique(),
        time.time() - t0,
    )

    logger.info("Constructing feature pipeline (calendar, lags, shifted rolling)...")
    t1 = time.time()
    feature_df = build_feature_pipeline(demand_df)
    logger.info(
        "Feature pipeline constructed: %s rows, %s columns in %.2fs. Eligible rows: %s",
        len(feature_df),
        len(feature_df.columns),
        time.time() - t1,
        feature_df["is_model_eligible"].sum() if "is_model_eligible" in feature_df.columns else "N/A",
    )

    logger.info(
        "Executing rolling 1-day-ahead chronological evaluation (test_days=%s, trees=%s, max_depth=%s)...",
        test_days,
        n_estimators,
        max_depth,
    )
    t2 = time.time()
    eval_results = run_rolling_evaluation(
        feature_df=feature_df,
        test_days=test_days,
        model_params={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": -1,
        },
        random_state=random_state,
    )
    eval_duration = time.time() - t2
    logger.info("Rolling evaluation completed successfully in %.2fs", eval_duration)

    split_rep = eval_results["split_report"]
    logger.info(
        "Chronological Split Summary:\n"
        "  Training Span: %s to %s (%s rows, %s products)\n"
        "  Test Span: %s to %s (%s rows, %s products)\n"
        "  Cutoff Date: %s (Test Window: %s days)",
        split_rep["min_date"],
        split_rep["cutoff_date"],
        split_rep["train_rows"],
        split_rep["train_products_count"],
        split_rep["test_start_date"],
        split_rep["test_end_date"],
        split_rep["test_rows"],
        split_rep["test_products_count"],
        split_rep["cutoff_date"],
        split_rep["test_horizon_days"],
    )

    excl = eval_results["exclusions"]
    logger.info(
        "Exclusions Audit:\n"
        "  Total Test Rows: %s\n"
        "  Evaluated Rows: %s\n"
        "  Excluded Rows: %s (%.2f%%)\n"
        "  Reasons: %s",
        excl["total_test_observations"],
        excl["evaluated_observations"],
        excl["excluded_observations"],
        excl["excluded_percentage"],
        excl["reasons"],
    )

    metrics = eval_results["metrics"]
    logger.info(
        "Overall Forecast Evaluation Metrics (Sample Size: %s):\n"
        "  Seasonal Naive Baseline:\n"
        "    MAE:  %.4f\n"
        "    RMSE: %.4f\n"
        "    WAPE: %.2f%%\n"
        "  Random Forest Regressor:\n"
        "    MAE:  %.4f\n"
        "    RMSE: %.4f\n"
        "    WAPE: %.2f%%\n"
        "  Macro-Averaged Per-Product WAPE:\n"
        "    Seasonal Naive: %.2f%%\n"
        "    Random Forest:  %.2f%%",
        metrics["evaluation_sample_size"],
        metrics["seasonal_naive"]["mae"],
        metrics["seasonal_naive"]["rmse"],
        metrics["seasonal_naive"]["wape"],
        metrics["random_forest"]["mae"],
        metrics["random_forest"]["rmse"],
        metrics["random_forest"]["wape"],
        metrics["macro_averaged"]["seasonal_naive"]["macro_wape"],
        metrics["macro_averaged"]["random_forest"]["macro_wape"],
    )

    # Save artifacts
    forecasts_path = Path(args.output_forecasts)
    metrics_path = Path(args.output_metrics)
    model_path = Path(args.output_model)

    forecasts_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save out-of-sample forecast comparison table
    logger.info("Saving forecast comparisons to: %s", forecasts_path)
    eval_table = eval_results["evaluation_table"]
    eval_table.to_parquet(forecasts_path, index=False)

    # 2. Save evaluation metrics JSON
    metrics_payload = {
        "split_summary": split_rep,
        "exclusions_audit": excl,
        "overall_metrics": {
            "sample_size": metrics["evaluation_sample_size"],
            "seasonal_naive": metrics["seasonal_naive"],
            "random_forest": metrics["random_forest"],
        },
        "macro_averaged_metrics": metrics["macro_averaged"],
        "model_parameters": {
            "model_type": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "features_used": eval_results["feature_cols"],
            "non_negative_constraint": "clip(0.0, None)",
        },
        "execution_metadata": {
            "duration_seconds": round(eval_duration, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    logger.info("Saving forecast metrics summary to: %s", metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, default=str)

    # 3. Save fitted Random Forest model
    logger.info("Saving trained Random Forest model artifact to: %s", model_path)
    joblib.dump(eval_results["model"], model_path)

    logger.info("Phase 5 forecasting experiment successfully finished!")


if __name__ == "__main__":
    main()
