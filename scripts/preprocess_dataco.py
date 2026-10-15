import sys
import time
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from retail_forecasting.data.preprocess_dataco import (
    load_raw_dataco_data,
    validate_dataco_raw_columns,
    parse_dataco_dates_and_keys,
    apply_demand_inclusion_policy,
    aggregate_dataco_daily_demand,
    extract_dataco_product_catalog,
    verify_dataco_preprocessing_reconciliation,
)

RAW_PATH = Path("data/raw/DataCoSupplyChainDataset.csv")
PROCESSED_DEMAND_PATH = Path("data/processed/dataco_daily_product_demand.parquet")
PROCESSED_CATALOG_PATH = Path("data/processed/dataco_product_catalog.parquet")


def main() -> None:
    print(f"Starting DataCo preprocessing on {RAW_PATH}...")
    t0 = time.time()

    # 1. Load raw data
    raw_df = load_raw_dataco_data(RAW_PATH, encoding="latin1")
    print(f"Loaded raw dataset: {len(raw_df):,} rows, {len(raw_df.columns)} columns in {time.time()-t0:.2f}s")

    # 2. Validate required columns
    validate_dataco_raw_columns(raw_df)
    print("Schema validation passed: All 53 canonical columns verified.")

    # 3. Parse dates, product IDs, and quantities
    parsed_df = parse_dataco_dates_and_keys(raw_df)
    print("Parsed dates and standardized product identifiers.")

    # 4. Apply demand-inclusion policy
    included_df, excluded_df, audit = apply_demand_inclusion_policy(parsed_df)
    print(f"Demand inclusion policy applied:")
    print(f"  Included: {len(included_df):,} rows ({audit['included_quantity']:,} units)")
    print(f"  Excluded: {len(excluded_df):,} rows ({audit['excluded_quantity']:,} units)")
    print(f"  Excluded statuses: {audit['excluded_by_status_breakdown']['row_counts']}")

    # 5. Aggregate daily product demand
    daily_demand_df = aggregate_dataco_daily_demand(included_df)
    print(f"Daily product demand aggregated: {len(daily_demand_df):,} records across "
          f"{daily_demand_df['product_id'].nunique()} products and {daily_demand_df['date'].nunique()} dates.")

    # 6. Extract product catalog
    catalog_df = extract_dataco_product_catalog(raw_df)
    print(f"Extracted clean product catalog: {len(catalog_df):,} SKUs (PII-free).")

    # 7. Mathematical reconciliation checks
    recon = verify_dataco_preprocessing_reconciliation(
        raw_df, included_df, excluded_df, daily_demand_df, audit
    )
    print("Accounting reconciliation verified:", recon)

    # 8. Save artifacts
    PROCESSED_DEMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_demand_df.to_parquet(PROCESSED_DEMAND_PATH, index=False)
    print(f"Saved daily demand artifact to {PROCESSED_DEMAND_PATH} ({PROCESSED_DEMAND_PATH.stat().st_size:,} bytes)")

    catalog_df.to_parquet(PROCESSED_CATALOG_PATH, index=False)
    print(f"Saved catalog artifact to {PROCESSED_CATALOG_PATH} ({PROCESSED_CATALOG_PATH.stat().st_size:,} bytes)")

    print(f"\nProcessing finished successfully in {time.time()-t0:.2f}s.")


if __name__ == "__main__":
    main()
