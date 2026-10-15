"""Data cleaning, demand definition, and daily aggregation pipeline for DataCo dataset.

Transforms raw DataCo Supply Chain records into:
1. Daily product-level demand time series (date, product_id, demand).
2. Clean, PII-free product catalog metadata (product_id, product_name, category, price, department).
3. Full data quality, status inclusion, and quantity reconciliation audit summaries.

Demand-Inclusion Policy:
- Excluded statuses: 'CANCELED' and 'SUSPECTED_FRAUD' (7,754 rows, 16,488 units).
  These correspond exactly to Delivery Status == 'Shipping canceled'.
- Included statuses: 'COMPLETE', 'PENDING_PAYMENT', 'PROCESSING', 'PENDING',
  'CLOSED', 'ON_HOLD', 'PAYMENT_REVIEW' (172,765 rows, 367,597 units).
- Product identifier: Clean string mapping from 'Product Card Id' (e.g. '19', '24').
- Date identifier: Calendar day of order placement ('order date (DateOrders)', truncated to YYYY-MM-DD).
- Privacy Safeguards: Strict exclusion of customer personal identifiers (names, emails,
  passwords, street addresses, exact coordinates).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from retail_forecasting.data.validate_data import DATACO_REQUIRED_COLUMNS


# Default excluded order statuses based on empirical shipping cancellation
DEFAULT_DATACO_EXCLUDED_STATUSES: Set[str] = {
    "CANCELED",
    "SUSPECTED_FRAUD",
}

# Default canonical demand columns
CANONICAL_DEMAND_COLUMNS: List[str] = ["date", "product_id", "demand"]


def load_raw_dataco_data(
    filepath: Union[str, Path],
    encoding: str = "latin1",
    **kwargs: Any
) -> pd.DataFrame:
    """Load raw DataCo CSV file safely with correct encoding without mutating file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the raw CSV file.
    encoding : str
        Character encoding (default 'latin1' / ISO-8859-1 for DataCo).
    **kwargs : Any
        Additional keyword arguments forwarded to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"DataCo raw file does not exist: '{path.resolve()}'")
    if not path.is_file():
        raise ValueError(f"Specified path is not a file: '{path.resolve()}'")

    return pd.read_csv(path, encoding=encoding, **kwargs)


def validate_dataco_raw_columns(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> bool:
    """Verify that expected columns are present in the raw DataCo DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataCo transaction DataFrame.
    required_columns : Optional[List[str]]
        Required column names. Defaults to DATACO_REQUIRED_COLUMNS.

    Returns
    -------
    bool
        True if all required columns are present.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    if required_columns is None:
        required_columns = DATACO_REQUIRED_COLUMNS

    actual_cols = set(df.columns)
    missing = [c for c in required_columns if c not in actual_cols]
    if missing:
        raise ValueError(f"DataCo raw DataFrame missing {len(missing)} required columns: {missing}")
    return True


def parse_dataco_dates_and_keys(
    df: pd.DataFrame,
    order_date_col: str = "order date (DateOrders)",
    product_id_col: str = "Product Card Id",
    quantity_col: str = "Order Item Quantity"
) -> pd.DataFrame:
    """Parse order dates safely to date and product IDs to standardized strings.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction DataFrame.
    order_date_col : str
        Order date column header.
    product_id_col : str
        Product SKU ID column header.
    quantity_col : str
        Quantity column header.

    Returns
    -------
    pd.DataFrame
        Copy with parsed 'date' (datetime64[ns] normalized to date), 'product_id' (string),
        and numeric 'quantity'.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

    out = df.copy()

    # Safely parse order dates
    parsed_dt = pd.to_datetime(out[order_date_col], errors="coerce")
    out["date"] = parsed_dt.dt.floor("D")

    # Safely parse product ID as string
    out["product_id"] = out[product_id_col].astype(str).str.strip()

    # Ensure quantity is numeric
    out["quantity"] = pd.to_numeric(out[quantity_col], errors="coerce")

    return out


def apply_demand_inclusion_policy(
    df: pd.DataFrame,
    status_col: str = "Order Status",
    excluded_statuses: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Apply documented demand inclusion policy to partition records into included vs excluded.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed transaction DataFrame.
    status_col : str
        Order status column header.
    excluded_statuses : Optional[Set[str]]
        Set of status strings to exclude. Defaults to DEFAULT_DATACO_EXCLUDED_STATUSES.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        - included_df: DataFrame of rows meeting demand inclusion policy.
        - excluded_df: DataFrame of rows excluded by policy or data validity checks.
        - audit_summary: Dictionary detailing row counts, quantity counts, and status breakdowns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

    if excluded_statuses is None:
        excluded_statuses = DEFAULT_DATACO_EXCLUDED_STATUSES

    out = df.copy()

    # 1. Structural validity checks
    is_invalid_date = out["date"].isna()
    is_invalid_product = out["product_id"].isna() | (out["product_id"] == "") | (out["product_id"] == "nan")
    is_invalid_qty = out["quantity"].isna() | (out["quantity"] <= 0)

    is_invalid = is_invalid_date | is_invalid_product | is_invalid_qty

    # 2. Status inclusion policy
    is_status_excluded = out[status_col].astype(str).str.strip().isin(excluded_statuses)

    # Combined mask
    is_included = (~is_invalid) & (~is_status_excluded)
    is_excluded = ~is_included

    included_df = out[is_included].copy()
    excluded_df = out[is_excluded].copy()

    # Detailed status audit
    status_counts = out[status_col].value_counts().to_dict()
    status_quantities = out.groupby(status_col)["quantity"].sum().to_dict()

    excluded_by_status = out[is_status_excluded & (~is_invalid)]
    excluded_by_status_counts = excluded_by_status[status_col].value_counts().to_dict()
    excluded_by_status_quantities = excluded_by_status.groupby(status_col)["quantity"].sum().to_dict()

    audit_summary: Dict[str, Any] = {
        "total_raw_rows": len(out),
        "total_raw_quantity": float(out["quantity"].sum()),
        "included_rows": len(included_df),
        "included_quantity": float(included_df["quantity"].sum()),
        "excluded_rows": len(excluded_df),
        "excluded_quantity": float(excluded_df["quantity"].sum()),
        "invalid_records": {
            "invalid_date_count": int(is_invalid_date.sum()),
            "invalid_product_count": int(is_invalid_product.sum()),
            "invalid_quantity_count": int(is_invalid_qty.sum()),
        },
        "excluded_by_status_breakdown": {
            "row_counts": excluded_by_status_counts,
            "quantity_counts": excluded_by_status_quantities,
        },
        "all_status_breakdown": {
            "row_counts": status_counts,
            "quantity_counts": status_quantities,
        },
        "inclusion_rules_summary": (
            f"Included orders with status not in {sorted(list(excluded_statuses))}. "
            "Excludes canceled orders and suspected fraud which were never shipped."
        ),
    }

    return included_df, excluded_df, audit_summary


def aggregate_dataco_daily_demand(
    included_df: pd.DataFrame,
    date_col: str = "date",
    product_col: str = "product_id",
    quantity_col: str = "quantity"
) -> pd.DataFrame:
    """Aggregate included transaction rows into daily product-level demand series.

    Parameters
    ----------
    included_df : pd.DataFrame
        Included positive transactions.
    date_col : str
        Normalized date column name.
    product_col : str
        Product SKU ID column name.
    quantity_col : str
        Quantity column name.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns ['date', 'product_id', 'demand'] sorted by
        [product_id, date]. Contains only observed active demand days without artificial zero-filling.
    """
    if not isinstance(included_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(included_df).__name__}")

    if included_df.empty:
        return pd.DataFrame(columns=CANONICAL_DEMAND_COLUMNS)

    agg_df = (
        included_df.groupby([product_col, date_col])[quantity_col]
        .sum()
        .reset_index()
        .rename(columns={quantity_col: "demand"})
    )

    # Standardize column order and dtypes
    agg_df["demand"] = agg_df["demand"].astype(float)
    agg_df["product_id"] = agg_df["product_id"].astype(str)
    agg_df["date"] = pd.to_datetime(agg_df["date"])

    agg_df = agg_df[CANONICAL_DEMAND_COLUMNS].sort_values(
        by=["product_id", "date"]
    ).reset_index(drop=True)

    return agg_df


def extract_dataco_product_catalog(
    df: pd.DataFrame,
    product_id_col: str = "Product Card Id",
    product_name_col: str = "Product Name",
    product_price_col: str = "Product Price",
    category_id_col: str = "Category Id",
    category_name_col: str = "Category Name",
    department_id_col: str = "Department Id",
    department_name_col: str = "Department Name"
) -> pd.DataFrame:
    """Extract clean, non-PII product catalog from DataCo records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataCo transaction DataFrame.
    product_id_col : str
        Product Card ID column.
    product_name_col : str
        Product Name column.
    product_price_col : str
        Product Price column.
    category_id_col : str
        Category ID column.
    category_name_col : str
        Category Name column.
    department_id_col : str
        Department ID column.
    department_name_col : str
        Department Name column.

    Returns
    -------
    pd.DataFrame
        PII-free product catalog table with one unique row per product_id.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

    catalog = (
        df.groupby(product_id_col)
        .agg({
            product_name_col: "first",
            product_price_col: "first",
            category_id_col: "first",
            category_name_col: "first",
            department_id_col: "first",
            department_name_col: "first",
        })
        .reset_index()
    )

    catalog = catalog.rename(
        columns={
            product_id_col: "product_id",
            product_name_col: "product_name",
            product_price_col: "product_price",
            category_id_col: "category_id",
            category_name_col: "category_name",
            department_id_col: "department_id",
            department_name_col: "department_name",
        }
    )

    catalog["product_id"] = catalog["product_id"].astype(str)
    catalog["product_price"] = catalog["product_price"].astype(float)
    catalog["category_id"] = catalog["category_id"].astype(int)
    catalog["department_id"] = catalog["department_id"].astype(int)

    return catalog.sort_values(by="product_id").reset_index(drop=True)


def verify_dataco_preprocessing_reconciliation(
    raw_df: pd.DataFrame,
    included_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    audit_summary: Dict[str, Any]
) -> Dict[str, bool]:
    """Perform rigorous mathematical accounting checks on preprocessing pipeline.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original raw DataFrame.
    included_df : pd.DataFrame
        Included transaction DataFrame.
    excluded_df : pd.DataFrame
        Excluded transaction DataFrame.
    aggregated_df : pd.DataFrame
        Aggregated daily demand DataFrame.
    audit_summary : Dict[str, Any]
        Audit summary dictionary.

    Returns
    -------
    Dict[str, bool]
        Dictionary of boolean verification results.

    Raises
    ------
    ValueError
        If any accounting check fails.
    """
    # 1. Row partition check
    row_partition_ok = (len(included_df) + len(excluded_df)) == len(raw_df)
    if not row_partition_ok:
        raise ValueError(
            f"Row count mismatch: included ({len(included_df)}) + excluded ({len(excluded_df)}) "
            f"!= raw ({len(raw_df)})"
        )

    # 2. Quantity reconciliation
    raw_qty = raw_df["Order Item Quantity"].sum()
    included_qty = included_df["quantity"].sum()
    excluded_qty = excluded_df["quantity"].sum()
    agg_qty = aggregated_df["demand"].sum()

    qty_partition_ok = np.isclose(included_qty + excluded_qty, raw_qty)
    agg_matches_included_ok = np.isclose(agg_qty, included_qty)

    if not qty_partition_ok:
        raise ValueError(
            f"Quantity partition mismatch: included ({included_qty}) + excluded ({excluded_qty}) "
            f"!= raw ({raw_qty})"
        )

    if not agg_matches_included_ok:
        raise ValueError(
            f"Aggregated demand mismatch: aggregated demand sum ({agg_qty}) "
            f"!= included quantity sum ({included_qty})"
        )

    # 3. No duplicate (date, product_id)
    dup_keys_count = int(aggregated_df.duplicated(subset=["date", "product_id"]).sum())
    no_duplicates_ok = dup_keys_count == 0
    if not no_duplicates_ok:
        raise ValueError(f"Encountered {dup_keys_count} duplicate (date, product_id) pairs in aggregation.")

    # 4. Non-negative demand
    non_negative_ok = bool((aggregated_df["demand"] >= 0).all())
    if not non_negative_ok:
        raise ValueError("Aggregated demand contains negative values.")

    return {
        "row_partition_ok": row_partition_ok,
        "qty_partition_ok": qty_partition_ok,
        "agg_matches_included_ok": agg_matches_included_ok,
        "no_duplicates_ok": no_duplicates_ok,
        "non_negative_ok": non_negative_ok,
    }


def audit_dataco_demand_sensitivity(
    raw_df: pd.DataFrame,
    policies: Optional[Dict[str, Set[str]]] = None
) -> pd.DataFrame:
    """Evaluate alternative demand definition policies and quantify sensitivity.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Input raw DataCo DataFrame.
    policies : Optional[Dict[str, Set[str]]]
        Dictionary mapping policy names to sets of included Order Status strings.

    Returns
    -------
    pd.DataFrame
        Summary table comparing included/excluded rows, units, daily records,
        and product coverage across evaluated policies.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(raw_df).__name__}")

    if policies is None:
        policies = {
            "Policy A (Current Baseline - Exclude Canceled & Fraud)": {
                "COMPLETE", "PENDING_PAYMENT", "PROCESSING", "PENDING", "CLOSED", "ON_HOLD", "PAYMENT_REVIEW"
            },
            "Policy B (Conservative Confirmed - COMPLETE & CLOSED)": {
                "COMPLETE", "CLOSED"
            },
            "Policy C (Strictly Complete - COMPLETE)": {
                "COMPLETE"
            },
            "Policy D (Paid / Settled - COMPLETE, CLOSED, PROCESSING)": {
                "COMPLETE", "CLOSED", "PROCESSING"
            },
        }

    parsed = parse_dataco_dates_and_keys(raw_df)
    total_raw_rows = len(raw_df)
    total_raw_units = raw_df["Order Item Quantity"].sum()

    results = []
    for pol_name, inc_statuses in policies.items():
        is_included = parsed["Order Status"].isin(inc_statuses)
        inc_df = parsed[is_included]
        exc_df = parsed[~is_included]

        inc_rows = len(inc_df)
        inc_units = inc_df["quantity"].sum()
        exc_rows = len(exc_df)
        exc_units = exc_df["quantity"].sum()

        agg = aggregate_dataco_daily_demand(inc_df)

        results.append({
            "policy_name": pol_name,
            "included_statuses": sorted(list(inc_statuses)),
            "included_rows": inc_rows,
            "included_rows_pct": round((inc_rows / total_raw_rows) * 100, 2),
            "included_units": float(inc_units),
            "included_units_pct": round((inc_units / total_raw_units) * 100, 2),
            "excluded_rows": exc_rows,
            "excluded_rows_pct": round((exc_rows / total_raw_rows) * 100, 2),
            "excluded_units": float(exc_units),
            "excluded_units_pct": round((exc_units / total_raw_units) * 100, 2),
            "daily_demand_rows": len(agg),
            "distinct_products": int(agg["product_id"].nunique()) if not agg.empty else 0,
            "first_order_date": str(agg["date"].min().date()) if not agg.empty else None,
            "last_order_date": str(agg["date"].max().date()) if not agg.empty else None,
        })

    return pd.DataFrame(results)


def audit_dataco_timeseries_readiness(
    daily_demand_df: pd.DataFrame
) -> pd.DataFrame:
    """Analyze product-level history length, calendar span, and sparsity.

    Parameters
    ----------
    daily_demand_df : pd.DataFrame
        Aggregated daily demand DataFrame with columns ['date', 'product_id', 'demand'].

    Returns
    -------
    pd.DataFrame
        Audit table containing per-product history metrics:
        [product_id, first_date, last_date, obs_sale_days, span_days, missing_days, sparsity_pct, total_demand].
    """
    if not isinstance(daily_demand_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(daily_demand_df).__name__}")

    records = []
    for pid, grp in daily_demand_df.groupby("product_id"):
        dates = grp["date"].sort_values()
        min_date = dates.min()
        max_date = dates.max()
        obs_days = len(dates)
        span = (max_date - min_date).days + 1
        missing = span - obs_days
        sparsity = (missing / span) * 100.0 if span > 0 else 0.0
        tot_qty = float(grp["demand"].sum())

        records.append({
            "product_id": str(pid),
            "first_date": str(min_date.date()),
            "last_date": str(max_date.date()),
            "obs_sale_days": int(obs_days),
            "span_days": int(span),
            "missing_days": int(missing),
            "sparsity_pct": round(sparsity, 2),
            "total_demand": tot_qty,
        })

    return pd.DataFrame(records).sort_values(by="total_demand", ascending=False).reset_index(drop=True)
