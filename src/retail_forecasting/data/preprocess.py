"""Data cleaning, transaction classification, and daily demand aggregation pipeline.

Transforms raw UCI Online Retail transaction logs into:
1. Cleaned and classified transaction logs (saved to data/interim/).
2. Daily product-level demand time series with zero-filling (saved to data/processed/).
3. Data quality and policy impact audit summaries (saved to outputs/metrics/).

Core Policies:
- Invariance: Input DataFrames and raw source files are never mutated.
- Transaction Classification: Records are cleanly categorized as ordinary positive sales,
  cancellations ('C' prefix), negative returns, zero-quantity adjustments, or invalid.
- Demand Target Definition: Positive retail demand forecasting models customer purchase volume.
  Cancellations and returns are tracked for ledger audit but excluded from the positive demand target.
- Observation Window & Zero-Filling: Non-selling days are zero-filled strictly between each
  product's first and last observed sale dates (product lifespan window), preventing artificial
  zero-inflation prior to product introduction or after discontinuation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from retail_forecasting.data.validate_data import ONLINE_RETAIL_REQUIRED_COLUMNS


DEFAULT_SPECIAL_STOCK_CODES = [
    "POST",
    "D",
    "M",
    "BANK CHARGES",
    "AMAZONFEE",
    "CRUK",
    "S",
    "B",
    "DOT",
    "PADS",
]


def standardize_columns(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Strip whitespace and standardize column headers.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataframe.
    column_mapping : Optional[Dict[str, str]]
        Optional renaming dictionary.

    Returns
    -------
    pd.DataFrame
        New DataFrame with standardized column names.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    if column_mapping:
        out = out.rename(columns=column_mapping)
    return out


def parse_transaction_fields(
    df: pd.DataFrame,
    date_col: str = "InvoiceDate",
    qty_col: str = "Quantity",
    price_col: str = "UnitPrice"
) -> pd.DataFrame:
    """Parse dates to datetime and numeric fields to numeric types.

    Adds a normalized calendar date column ('date') representing the day of transaction.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataframe.
    date_col : str
        Name of transaction timestamp column.
    qty_col : str
        Name of quantity column.
    price_col : str
        Name of unit price column.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed dates and numeric values. Invalid values are coerced
        to NaNs for classification without raising errors.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    out = df.copy()

    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        # Extract calendar date (midnight normalized)
        out["date"] = out[date_col].dt.normalize()
    else:
        out["date"] = pd.NaT

    if qty_col in out.columns:
        out[qty_col] = pd.to_numeric(out[qty_col], errors="coerce")

    if price_col in out.columns:
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    return out


def clean_product_identifiers(
    df: pd.DataFrame,
    stock_col: str = "StockCode",
    desc_col: str = "Description"
) -> pd.DataFrame:
    """Clean and standardize product SKU codes and text descriptions.

    Creates standardized 'product_id' column from 'StockCode'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    stock_col : str
        Name of product code column.
    desc_col : str
        Name of description column.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized product_id and cleaned Description.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    out = df.copy()

    if stock_col in out.columns:
        # Cast to string, strip whitespace, and upper-case
        out["product_id"] = (
            out[stock_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )
    else:
        out["product_id"] = np.nan

    if desc_col in out.columns:
        out[desc_col] = (
            out[desc_col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )

    return out


def classify_transactions(
    df: pd.DataFrame,
    invoice_col: str = "InvoiceNo",
    qty_col: str = "Quantity",
    date_col: str = "date",
    product_col: str = "product_id"
) -> pd.DataFrame:
    """Classify transaction records into mutually exclusive categories.

    Categories:
    - 'invalid': Unparseable date, missing product_id, or null/NaN Quantity.
    - 'cancellation': InvoiceNo begins with 'C' or 'c' (formal invoice cancellation).
    - 'return_only': Negative Quantity without a 'C' invoice prefix (damaged, loss, or return entry).
    - 'zero_quantity': Quantity equals exactly zero.
    - 'positive_sale': Valid purchase record (Quantity > 0 and not cancelled).

    Parameters
    ----------
    df : pd.DataFrame
        Input parsed dataframe.
    invoice_col : str
        Invoice number column.
    qty_col : str
        Quantity column.
    date_col : str
        Parsed calendar date column.
    product_col : str
        Cleaned product ID column.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with a 'transaction_type' column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    out = df.copy()

    # Identify structural invalidity
    is_invalid = (
        out[date_col].isna()
        | out[product_col].isna()
        | out[qty_col].isna()
    )

    # Robust check for invoice cancellation prefix 'C'
    invoice_str = out[invoice_col].astype(str).str.strip().str.upper() if invoice_col in out.columns else pd.Series([""] * len(out))
    is_cancellation = invoice_str.str.startswith("C")

    # Numeric quantity checks
    qty = out[qty_col].fillna(0)
    is_negative = qty < 0
    is_zero = qty == 0
    is_positive = qty > 0

    # Mutually exclusive priority assignment
    conditions = [
        is_invalid,
        is_cancellation,
        is_negative & (~is_cancellation),
        is_zero & (~is_cancellation),
        is_positive & (~is_cancellation),
    ]
    choices = [
        "invalid",
        "cancellation",
        "return_only",
        "zero_quantity",
        "positive_sale",
    ]

    out["transaction_type"] = np.select(conditions, choices, default="invalid")
    return out


def summarize_transaction_classification(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive audit metrics for transaction categories.

    Ensures no double-counting across categories.

    Parameters
    ----------
    df : pd.DataFrame
        Classified dataframe containing 'transaction_type' and 'Quantity'.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with row counts, total quantities, percentages, and date range.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    total_records = len(df)
    if total_records == 0:
        return {
            "total_records": 0,
            "category_counts": {},
            "category_quantities": {},
            "category_percentages": {},
            "date_range": None,
        }

    counts = df["transaction_type"].value_counts().to_dict()
    quantities = (
        df.groupby("transaction_type")["Quantity"].sum().to_dict()
        if "Quantity" in df.columns
        else {}
    )

    percentages = {
        cat: round(cnt / total_records * 100.0, 2)
        for cat, cnt in counts.items()
    }

    min_date = str(df["date"].min()) if "date" in df.columns and df["date"].notna().any() else None
    max_date = str(df["date"].max()) if "date" in df.columns and df["date"].notna().any() else None

    return {
        "total_records": total_records,
        "category_counts": counts,
        "category_quantities": quantities,
        "category_percentages": percentages,
        "date_range": {"min": min_date, "max": max_date},
        "unique_products": int(df["product_id"].nunique()) if "product_id" in df.columns else 0,
    }


def filter_positive_demand(
    df: pd.DataFrame,
    exclude_special_codes: bool = True,
    special_codes: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filter records to isolate clean, positive consumer demand transactions.

    Policy:
    - Retains only records where transaction_type == 'positive_sale'.
    - Excludes non-merchandise service codes (e.g., POST, D, BANK CHARGES) when requested.
    - Excludes non-positive unit prices (free/promotional samples or loss adjustments).

    Parameters
    ----------
    df : pd.DataFrame
        Classified dataframe.
    exclude_special_codes : bool
        If True, filters out accounting and administrative product codes.
    special_codes : Optional[List[str]]
        List of codes to exclude. Defaults to DEFAULT_SPECIAL_STOCK_CODES.

    Returns
    -------
    pd.DataFrame
        Filtered positive demand transactions.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if special_codes is None:
        special_codes = DEFAULT_SPECIAL_STOCK_CODES

    # Filter to positive sales only
    filtered = df[df["transaction_type"] == "positive_sale"].copy()

    # Filter non-positive unit price (free items or debit adjustments)
    if "UnitPrice" in filtered.columns:
        filtered = filtered[filtered["UnitPrice"] > 0]

    # Filter non-product special administrative codes
    if exclude_special_codes and "product_id" in filtered.columns:
        filtered = filtered[~filtered["product_id"].isin(special_codes)]

    return filtered


def aggregate_daily_demand(
    df: pd.DataFrame,
    date_col: str = "date",
    id_cols: Optional[List[str]] = None,
    target_col: str = "demand",
    product_col: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate transactional sales into daily product-level demand series.

    Maintains backwards compatibility with both (id_cols, target_col) and
    (product_col, target_col) conventions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned positive-demand transaction dataframe.
    date_col : str
        Date column name.
    id_cols : Optional[List[str]]
        List of identifier columns. If None and product_col provided, uses [product_col].
        Defaults to ['product_id'].
    target_col : str
        Quantity column to sum (e.g. 'demand' or 'Quantity').
    product_col : Optional[str]
        Explicit single product column.

    Returns
    -------
    pd.DataFrame
        Aggregated daily demand with columns: [date_col, product_id, demand].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        return pd.DataFrame(columns=[date_col, "product_id", "demand"])

    # Determine grouping column
    group_col = "product_id"
    if product_col and product_col in df.columns:
        group_col = product_col
    elif id_cols and len(id_cols) > 0 and id_cols[0] in df.columns:
        group_col = id_cols[0]
    elif "StockCode" in df.columns and "product_id" not in df.columns:
        group_col = "StockCode"

    # Determine quantity column to sum
    sum_col = "Quantity"
    if target_col in df.columns:
        sum_col = target_col
    elif "Quantity" in df.columns:
        sum_col = "Quantity"
    elif "sales" in df.columns:
        sum_col = "sales"

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe columns: {list(df.columns)}")

    if group_col not in df.columns:
        raise ValueError(f"Product grouping column '{group_col}' not found in dataframe.")

    aggregated = (
        df.groupby([date_col, group_col], as_index=False)[sum_col]
        .sum()
        .rename(columns={group_col: "product_id", sum_col: "demand"})
    )

    # Ensure demand is numeric and non-negative
    aggregated["demand"] = pd.to_numeric(aggregated["demand"], errors="coerce").fillna(0.0)
    aggregated = aggregated[aggregated["demand"] >= 0]

    # Ensure consistent sorting
    aggregated = aggregated.sort_values(by=["product_id", date_col]).reset_index(drop=True)
    return aggregated


def apply_zero_filling(
    daily_df: pd.DataFrame,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand",
    window_policy: str = "product_lifespan",
    min_active_days_sparse: int = 7
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fill non-selling days with zero demand under an explicit observation window policy.

    Observation Window Policies:
    - 'product_lifespan': For each product, builds a continuous calendar from its first
      observed sale date to its last observed sale date. Days with no sales in between
      are filled with 0.0. This prevents artificial zero inflation before product launch
      or after product retirement.
    - 'global': Fills every day from the global minimum date to the global maximum date
      for all products.
    - 'none': Returns daily_df unmodified without zero filling.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Aggregated daily demand DataFrame with columns [date, product_id, demand].
    date_col : str
        Date column name.
    product_col : str
        Product identifier column name.
    demand_col : str
        Demand column name.
    window_policy : str
        One of 'product_lifespan', 'global', 'none'.
    min_active_days_sparse : int
        Threshold of active observed selling days below which a product is diagnosed as sparse.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (zero_filled_df, zero_filling_report)
    """
    if not isinstance(daily_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(daily_df).__name__}")

    if daily_df.empty:
        empty_report = {
            "window_policy": window_policy,
            "total_product_days": 0,
            "observed_sales_days": 0,
            "zero_filled_days": 0,
            "zero_fill_ratio": 0.0,
            "sparse_products_count": 0,
            "sparse_products": [],
        }
        return daily_df.copy(), empty_report

    if window_policy == "none":
        report = {
            "window_policy": "none",
            "total_product_days": len(daily_df),
            "observed_sales_days": len(daily_df),
            "zero_filled_days": 0,
            "zero_fill_ratio": 0.0,
            "sparse_products_count": 0,
            "sparse_products": [],
        }
        return daily_df.copy(), report

    # Determine product lifespans and sparse histories
    product_stats = (
        daily_df.groupby(product_col)
        .agg(
            first_date=(date_col, "min"),
            last_date=(date_col, "max"),
            active_days=(date_col, "count"),
        )
        .reset_index()
    )

    sparse_products = product_stats[
        product_stats["active_days"] < min_active_days_sparse
    ][product_col].tolist()

    global_min = daily_df[date_col].min()
    global_max = daily_df[date_col].max()

    reindexed_frames: List[pd.DataFrame] = []

    for _, row in product_stats.iterrows():
        p_id = row[product_col]
        if window_policy == "product_lifespan":
            p_start = row["first_date"]
            p_end = row["last_date"]
        else:  # global policy
            p_start = global_min
            p_end = global_max

        # Generate complete daily date range for this product's window
        full_dates = pd.date_range(start=p_start, end=p_end, freq="D", name=date_col)
        product_grid = pd.DataFrame({date_col: full_dates})
        product_grid[product_col] = p_id

        # Merge with observed demand for this product
        p_subset = daily_df[daily_df[product_col] == p_id]
        merged = pd.merge(product_grid, p_subset, on=[date_col, product_col], how="left")
        merged[demand_col] = merged[demand_col].fillna(0.0)

        reindexed_frames.append(merged)

    zero_filled_df = pd.concat(reindexed_frames, ignore_index=True)
    zero_filled_df = zero_filled_df.sort_values(by=[product_col, date_col]).reset_index(drop=True)

    total_product_days = len(zero_filled_df)
    observed_sales_days = len(daily_df)
    zero_filled_days = total_product_days - observed_sales_days
    zero_fill_ratio = (
        round(zero_filled_days / total_product_days, 4)
        if total_product_days > 0
        else 0.0
    )

    report = {
        "window_policy": window_policy,
        "total_product_days": total_product_days,
        "observed_sales_days": observed_sales_days,
        "zero_filled_days": zero_filled_days,
        "zero_fill_ratio": zero_fill_ratio,
        "sparse_products_count": len(sparse_products),
        "sparse_products": sparse_products[:50],  # capped sample for logging
    }

    return zero_filled_df, report


def clean_sales_transactions(
    df: pd.DataFrame,
    exclude_special_codes: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Primary pipeline for transaction cleaning and classification.

    Performs:
    1. Column name standardization.
    2. Date and numeric parsing.
    3. Product SKU string hygiene.
    4. Mutually exclusive transaction classification.
    5. Diagnostic auditing.

    Does not mutate input df.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction DataFrame.
    exclude_special_codes : bool
        Whether special non-merchandise codes are flagged.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (cleaned_transactions_df, audit_summary_report)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    std_df = standardize_columns(df)
    parsed_df = parse_transaction_fields(std_df)
    cleaned_df = clean_product_identifiers(parsed_df)
    classified_df = classify_transactions(cleaned_df)

    # Ensure consistent dtypes for parquet/arrow serialization
    if "InvoiceNo" in classified_df.columns:
        classified_df["InvoiceNo"] = classified_df["InvoiceNo"].astype(str).str.strip()
    if "StockCode" in classified_df.columns:
        classified_df["StockCode"] = classified_df["StockCode"].astype(str).str.strip()
    if "Country" in classified_df.columns:
        classified_df["Country"] = classified_df["Country"].astype(str).str.strip()
    if "CustomerID" in classified_df.columns:
        classified_df["CustomerID"] = pd.to_numeric(classified_df["CustomerID"], errors="coerce")

    audit_summary = summarize_transaction_classification(classified_df)

    return classified_df, audit_summary


def preprocess_pipeline(
    raw_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute end-to-end Phase 3 preprocessing workflow.

    Workflow:
    1. Clean and classify transactions -> cleaned_transactions (interim).
    2. Filter positive demand transactions.
    3. Aggregate daily product-level demand.
    4. Apply product-lifespan zero-filling and diagnose sparse products.
    5. Compile audit summaries.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw input records.
    config : Optional[Dict[str, Any]]
        Optional configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'cleaned_transactions': pd.DataFrame
        - 'daily_demand': pd.DataFrame
        - 'audit_summary': Dict[str, Any]
        - 'zero_filling_report': Dict[str, Any]
    """
    cfg = config or {}
    zero_policy = cfg.get("zero_filling_policy", "product_lifespan")
    sparse_thresh = cfg.get("sparse_threshold", 7)
    exclude_special = cfg.get("exclude_special_codes", True)

    cleaned_df, audit_summary = clean_sales_transactions(
        raw_df, exclude_special_codes=exclude_special
    )

    pos_demand_df = filter_positive_demand(
        cleaned_df, exclude_special_codes=exclude_special
    )

    daily_raw = aggregate_daily_demand(
        pos_demand_df, date_col="date", product_col="product_id", target_col="Quantity"
    )

    daily_filled, zero_report = apply_zero_filling(
        daily_raw,
        date_col="date",
        product_col="product_id",
        demand_col="demand",
        window_policy=zero_policy,
        min_active_days_sparse=sparse_thresh,
    )

    return {
        "cleaned_transactions": cleaned_df,
        "daily_demand": daily_filled,
        "audit_summary": audit_summary,
        "zero_filling_report": zero_report,
    }
