"""Data validation and schema integrity checking module for retail transaction datasets.

Verifies mandatory columns, data types, value ranges, and generates diagnostic
summaries for nulls, duplicates, and format anomalies.

UCI Online Retail Canonical Schema:
- InvoiceNo   : Transaction/Invoice ID (string/int; starting with 'C' indicates cancellation).
- StockCode   : Product/SKU identifier.
- Description : Product title/description.
- Quantity    : Quantity sold (negative indicates return/cancellation).
- InvoiceDate : Timestamp of invoice generation.
- UnitPrice   : Unit selling price (numeric).
- CustomerID  : Customer identifier (optional/nullable in wholesale guest checkouts).
- Country     : Destination/customer country name.

Important Boundary Note:
Validation checks integrity and reports diagnostic anomalies. Decisions regarding
how to filter or transform cancellations, negative quantities, returns, or missing
customer IDs/descriptions are explicitly deferred to the preprocessing phase.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# Canonical columns for the UCI Online Retail dataset
ONLINE_RETAIL_REQUIRED_COLUMNS: List[str] = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Canonical columns for the DataCo Smart Supply Chain dataset (53 columns)
DATACO_REQUIRED_COLUMNS: List[str] = [
    "Type",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Delivery Status",
    "Late_delivery_risk",
    "Category Id",
    "Category Name",
    "Customer City",
    "Customer Country",
    "Customer Email",
    "Customer Fname",
    "Customer Id",
    "Customer Lname",
    "Customer Password",
    "Customer Segment",
    "Customer State",
    "Customer Street",
    "Customer Zipcode",
    "Department Id",
    "Department Name",
    "Latitude",
    "Longitude",
    "Market",
    "Order City",
    "Order Country",
    "Order Customer Id",
    "order date (DateOrders)",
    "Order Id",
    "Order Item Cardprod Id",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Id",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Order Region",
    "Order State",
    "Order Status",
    "Order Zipcode",
    "Product Card Id",
    "Product Category Id",
    "Product Description",
    "Product Image",
    "Product Name",
    "Product Price",
    "Product Status",
    "shipping date (DateOrders)",
    "Shipping Mode",
]


def validate_schema(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    allow_extra_columns: bool = True
) -> bool:
    """Validate that required columns are present in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataframe.
    expected_columns : Optional[List[str]]
        List of required column names. If None, defaults to ONLINE_RETAIL_REQUIRED_COLUMNS.
    allow_extra_columns : bool
        If True (default), extra columns are permitted. If False, unexpected columns
        will trigger a validation failure.

    Returns
    -------
    bool
        True if schema meets expectations.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If required columns are missing, or unexpected columns are present when
        allow_extra_columns is False.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got: {type(df).__name__}")

    if expected_columns is None:
        expected_columns = ONLINE_RETAIL_REQUIRED_COLUMNS

    actual_columns = set(df.columns)
    missing_columns = [col for col in expected_columns if col not in actual_columns]

    if missing_columns:
        raise ValueError(
            f"Schema validation failed. Missing {len(missing_columns)} required column(s): "
            f"{missing_columns}. Found columns: {list(df.columns)}"
        )

    if not allow_extra_columns:
        extra_columns = [col for col in df.columns if col not in expected_columns]
        if extra_columns:
            raise ValueError(
                f"Schema validation failed. Disallowed extra column(s) encountered: "
                f"{extra_columns}. Allowed columns: {expected_columns}"
            )

    return True


def validate_non_empty(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to evaluate.

    Returns
    -------
    bool
        True if dataframe has at least 1 row and 1 column.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If dataframe has 0 rows or 0 columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got: {type(df).__name__}")

    if df.empty or len(df) == 0:
        raise ValueError("Empty data validation failed: DataFrame contains 0 rows.")

    if len(df.columns) == 0:
        raise ValueError("Empty data validation failed: DataFrame contains 0 columns.")

    return True


def check_missing_and_duplicates(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate diagnostic summary of missing values and duplicate rows.

    Does not modify or delete rows from the caller's DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataframe.
    key_columns : Optional[List[str]]
        Optional subset of columns forming a composite business key.

    Returns
    -------
    Dict[str, Any]
        Diagnostic dictionary containing:
        - total_rows: integer row count
        - missing_counts: Dict[col, count]
        - missing_percentages: Dict[col, float percentage]
        - total_duplicate_rows: int count of exact duplicate rows
        - key_duplicate_rows: int count of duplicates on key_columns (if provided)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got: {type(df).__name__}")

    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = (
        (df.isnull().sum() / total_rows * 100.0).round(2).to_dict()
        if total_rows > 0
        else {c: 0.0 for c in df.columns}
    )

    duplicate_rows_count = int(df.duplicated().sum())

    key_duplicates = 0
    if key_columns:
        valid_keys = [k for k in key_columns if k in df.columns]
        if valid_keys:
            key_duplicates = int(df.duplicated(subset=valid_keys).sum())

    return {
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "total_duplicate_rows": duplicate_rows_count,
        "key_duplicate_rows": key_duplicates,
    }


def validate_types_and_values(
    df: pd.DataFrame,
    strict: bool = False
) -> Dict[str, Any]:
    """Inspect and report field types and value plausibility for retail records.

    Evaluates:
    - Date parsing: Whether InvoiceDate can be interpreted as datetime.
    - Numeric parsing: Whether Quantity and UnitPrice can be interpreted as numeric.
    - Diagnostics (Warnings): Counts of negative quantities (returns/cancellations)
      and non-positive unit prices.

    Does not mutate the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input retail transaction dataframe.
    strict : bool
        If True, raises a ValueError upon discovering unparseable dates or non-numeric
        values in required numeric fields. If False (default), returns diagnostic counts.

    Returns
    -------
    Dict[str, Any]
        Diagnostic summary with error counts, warnings, and field status.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If strict is True and structural parsing errors are detected.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got: {type(df).__name__}")

    errors: List[str] = []
    warnings: List[str] = []

    invalid_date_count = 0
    invalid_quantity_count = 0
    invalid_unit_price_count = 0
    negative_quantity_count = 0
    non_positive_price_count = 0

    # 1. Date interpretability check
    if "InvoiceDate" in df.columns:
        date_series = df["InvoiceDate"].dropna()
        if len(date_series) > 0:
            parsed_dates = pd.to_datetime(date_series, errors="coerce")
            invalid_date_count = int(parsed_dates.isnull().sum())
            if invalid_date_count > 0:
                errors.append(
                    f"{invalid_date_count} records have unparseable 'InvoiceDate' values."
                )

    # 2. Quantity interpretability check
    if "Quantity" in df.columns:
        qty_series = df["Quantity"].dropna()
        if len(qty_series) > 0:
            parsed_qty = pd.to_numeric(qty_series, errors="coerce")
            invalid_quantity_count = int(parsed_qty.isnull().sum())
            if invalid_quantity_count > 0:
                errors.append(
                    f"{invalid_quantity_count} records have non-numeric 'Quantity' values."
                )
            else:
                # Plausibility check (Diagnostic warning, NOT a schema error)
                negative_quantity_count = int((parsed_qty < 0).sum())
                if negative_quantity_count > 0:
                    warnings.append(
                        f"{negative_quantity_count} records have negative 'Quantity' "
                        f"(represents cancellations/returns; deferred to preprocessing)."
                    )

    # 3. UnitPrice interpretability check
    if "UnitPrice" in df.columns:
        price_series = df["UnitPrice"].dropna()
        if len(price_series) > 0:
            parsed_price = pd.to_numeric(price_series, errors="coerce")
            invalid_unit_price_count = int(parsed_price.isnull().sum())
            if invalid_unit_price_count > 0:
                errors.append(
                    f"{invalid_unit_price_count} records have non-numeric 'UnitPrice' values."
                )
            else:
                # Plausibility check (Diagnostic warning, NOT a schema error)
                non_positive_price_count = int((parsed_price <= 0).sum())
                if non_positive_price_count > 0:
                    warnings.append(
                        f"{non_positive_price_count} records have non-positive 'UnitPrice' "
                        f"(zero/free or debt adjustments; deferred to preprocessing)."
                    )

    report = {
        "is_valid": len(errors) == 0,
        "invalid_date_count": invalid_date_count,
        "invalid_quantity_count": invalid_quantity_count,
        "invalid_unit_price_count": invalid_unit_price_count,
        "negative_quantity_count": negative_quantity_count,
        "non_positive_price_count": non_positive_price_count,
        "errors": errors,
        "warnings": warnings,
    }

    if strict and errors:
        raise ValueError(f"Type and value validation failed: {'; '.join(errors)}")

    return report


def validate_dataset(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    allow_extra_columns: bool = True,
    strict: bool = True
) -> Dict[str, Any]:
    """Comprehensive validation pipeline for incoming retail dataset.

    Performs non-empty check, schema check, missing/duplicate audits, and
    type/value plausibility checks.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded transaction dataframe.
    expected_columns : Optional[List[str]]
        Expected column headers.
    allow_extra_columns : bool
        Whether extra columns are accepted.
    strict : bool
        If True, raises exception on missing columns, empty data, or invalid types.

    Returns
    -------
    Dict[str, Any]
        Complete validation summary report.
    """
    validate_non_empty(df)
    validate_schema(df, expected_columns=expected_columns, allow_extra_columns=allow_extra_columns)
    missing_dup_report = check_missing_and_duplicates(df)
    type_report = validate_types_and_values(df, strict=strict)

    return {
        "is_valid": type_report["is_valid"],
        "schema_ok": True,
        "non_empty_ok": True,
        "missing_and_duplicates": missing_dup_report,
        "types_and_values": type_report,
    }


def validate_dataco_dataset(
    df: pd.DataFrame,
    strict: bool = True
) -> Dict[str, Any]:
    """Validation pipeline for DataCo Smart Supply Chain transaction records.

    Verifies the 53 required columns, checks for empty data, calculates missing/duplicate
    statistics, validates order and shipping date interpretability, and checks numeric fields.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataCo dataframe.
    strict : bool
        If True, raises exception on missing columns, empty data, or unparseable dates.

    Returns
    -------
    Dict[str, Any]
        Diagnostic summary report.
    """
    validate_non_empty(df)
    validate_schema(df, expected_columns=DATACO_REQUIRED_COLUMNS, allow_extra_columns=True)

    missing_dup_report = check_missing_and_duplicates(df)

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Date interpretability
    order_date_col = "order date (DateOrders)"
    shipping_date_col = "shipping date (DateOrders)"

    parsed_order_dates = pd.to_datetime(df[order_date_col], errors="coerce")
    invalid_order_dates = int(parsed_order_dates.isnull().sum())
    if invalid_order_dates > 0:
        errors.append(f"{invalid_order_dates} records have unparseable '{order_date_col}' values.")

    parsed_shipping_dates = pd.to_datetime(df[shipping_date_col], errors="coerce")
    invalid_shipping_dates = int(parsed_shipping_dates.isnull().sum())
    if invalid_shipping_dates > 0:
        errors.append(f"{invalid_shipping_dates} records have unparseable '{shipping_date_col}' values.")

    # Chronological consistency check
    if invalid_order_dates == 0 and invalid_shipping_dates == 0:
        chronological_inversions = int((parsed_shipping_dates < parsed_order_dates).sum())
        if chronological_inversions > 0:
            warnings.append(
                f"{chronological_inversions} records have shipping dates preceding order dates."
            )
    else:
        chronological_inversions = 0

    # 2. Quantity interpretability
    qty_col = "Order Item Quantity"
    parsed_qty = pd.to_numeric(df[qty_col], errors="coerce")
    invalid_qty_count = int(parsed_qty.isnull().sum())
    if invalid_qty_count > 0:
        errors.append(f"{invalid_qty_count} records have non-numeric '{qty_col}' values.")
    non_positive_qty_count = int((parsed_qty <= 0).sum())
    if non_positive_qty_count > 0:
        warnings.append(f"{non_positive_qty_count} records have non-positive '{qty_col}' values.")

    # 3. Product & Order identifiers
    distinct_orders = int(df["Order Id"].nunique()) if "Order Id" in df.columns else 0
    distinct_products = int(df["Product Card Id"].nunique()) if "Product Card Id" in df.columns else 0

    is_valid = len(errors) == 0
    if strict and errors:
        raise ValueError(f"DataCo validation failed: {'; '.join(errors)}")

    return {
        "is_valid": is_valid,
        "schema_ok": True,
        "non_empty_ok": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "distinct_orders": distinct_orders,
        "distinct_products": distinct_products,
        "chronological_inversions": chronological_inversions,
        "missing_and_duplicates": missing_dup_report,
        "errors": errors,
        "warnings": warnings,
    }
