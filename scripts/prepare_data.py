"""Data preparation script for retail sales demand forecasting.

Loads raw UCI Online Retail data, validates schema integrity, cleans and classifies transactions,
aggregates daily product-level demand with lifespan zero-filling, and saves interim/processed artifacts.

Usage:
    python scripts/prepare_data.py --input "data/raw/Online Retail.xlsx"
"""

import argparse
import json
from pathlib import Path
import sys
from retail_forecasting.data.load_data import load_raw_sales_data
from retail_forecasting.data.validate_data import validate_dataset
from retail_forecasting.data.preprocess import preprocess_pipeline
from retail_forecasting.utils.logging_config import setup_logger

logger = setup_logger("prepare_data")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean and prepare daily retail demand data.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/Online Retail.xlsx",
        help="Path to raw transaction data file (.xlsx, .csv, .parquet).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration YAML if pyyaml is installed and file exists, else return default dict."""
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
    """Execute end-to-end data preparation workflow."""
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path.resolve())
        sys.exit(1)

    logger.info("Loading configuration from: %s", args.config)
    config = load_config(args.config)
    data_proc_cfg = config.get("data_processing", {})
    prep_cfg = data_proc_cfg.get("preprocessing", {})

    logger.info("Loading raw transaction records from: %s", input_path)
    raw_df = load_raw_sales_data(input_path)
    logger.info("Loaded raw shape: %s rows, %s columns", len(raw_df), len(raw_df.columns))

    logger.info("Validating dataset integrity...")
    val_report = validate_dataset(raw_df, strict=False)
    logger.info(
        "Validation complete. Null CustomerID: %s, Missing Descriptions: %s",
        val_report["missing_and_duplicates"]["missing_counts"].get("CustomerID", 0),
        val_report["missing_and_duplicates"]["missing_counts"].get("Description", 0),
    )

    logger.info("Running Phase 3 cleaning, transaction classification, and daily aggregation...")
    results = preprocess_pipeline(raw_df, config=prep_cfg)

    cleaned_df = results["cleaned_transactions"]
    daily_demand_df = results["daily_demand"]
    audit_summary = results["audit_summary"]
    zero_report = results["zero_filling_report"]

    logger.info(
        "Transaction Classification Audit:\n  Positive sales: %s\n  Cancellations: %s\n  Returns only: %s\n  Invalid: %s",
        audit_summary["category_counts"].get("positive_sale", 0),
        audit_summary["category_counts"].get("cancellation", 0),
        audit_summary["category_counts"].get("return_only", 0),
        audit_summary["category_counts"].get("invalid", 0),
    )
    logger.info(
        "Zero-Filling Summary:\n  Total product-days: %s\n  Observed active days: %s\n  Zero-filled days: %s (Ratio: %s)\n  Sparse products (<7 days): %s",
        zero_report["total_product_days"],
        zero_report["observed_sales_days"],
        zero_report["zero_filled_days"],
        zero_report["zero_fill_ratio"],
        zero_report["sparse_products_count"],
    )

    # Save outputs to configuration-specified paths
    interim_file = Path(prep_cfg.get("interim_cleaned_file", "data/interim/cleaned_transactions.parquet"))
    processed_file = Path(prep_cfg.get("processed_demand_file", "data/processed/daily_product_demand.parquet"))
    metrics_file = Path(prep_cfg.get("quality_summary_file", "outputs/metrics/data_quality_summary.json"))

    interim_file.parent.mkdir(parents=True, exist_ok=True)
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving interim cleaned transactions to: %s", interim_file)
    cleaned_df.to_parquet(interim_file, index=False)

    logger.info("Saving processed daily product demand to: %s", processed_file)
    daily_demand_df.to_parquet(processed_file, index=False)

    quality_payload = {
        "validation_report": val_report,
        "classification_audit": audit_summary,
        "zero_filling_report": zero_report,
    }
    logger.info("Saving quality and audit summary to: %s", metrics_file)
    with open(metrics_file, "w", encoding="utf-8") as fh:
        json.dump(quality_payload, fh, indent=2, default=str)

    logger.info("Phase 3 data preparation successfully completed!")


if __name__ == "__main__":
    main()
