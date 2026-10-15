"""Main application CLI orchestrator for Retail Sales Forecasting & Inventory Optimization.

Provides a unified command-line entry point to run data preparation,
model training, evaluation, and inventory recommendation tasks.

Usage:
    python main.py --help
"""

import argparse
import sys
from retail_forecasting.utils.logging_config import setup_logger

logger = setup_logger("main")


def parse_args() -> argparse.Namespace:
    """Parse CLI orchestration commands."""
    parser = argparse.ArgumentParser(
        description="Retail Demand Forecasting & Inventory Optimization System CLI."
    )
    subparsers = parser.add_subparsers(dest="command", help="Target task to execute")

    # Prepare data subcommand
    prep_parser = subparsers.add_parser("prepare-data", help="Ingest and clean raw data")

    # Train model subcommand
    train_parser = subparsers.add_parser("train", help="Train forecasting models and evaluate")

    # Inventory recommendation subcommand
    rec_parser = subparsers.add_parser("recommend", help="Generate inventory recommendations")

    return parser.parse_args()


def main() -> None:
    """Main execution dispatcher."""
    args = parse_args()

    if args.command is None:
        logger.info(
            "Retail Demand Forecasting & Inventory Optimization System CLI initialized.\n"
            "Run 'python main.py --help' to inspect available task commands.\n"
            "Full pipeline execution will be configured in subsequent implementation phases."
        )
        return

    logger.warning("Selected task '%s' is not yet implemented.", args.command)


if __name__ == "__main__":
    main()
