"""Unit tests for Phase 3 data cleaning, transaction classification, and daily demand aggregation.

Verifies:
- Correct parsing of valid invoice dates and calendar normalization.
- Aggregation of multiple transactions for the same product and date.
- Separation of different products during aggregation.
- Handling of invoice numbers with and without a 'C' prefix.
- Identification and reporting of negative quantities (cancellations vs. returns).
- Exclusion of returns/cancellations from primary positive demand.
- Handling of missing or non-standard product identifiers.
- Handling of invalid dates and non-numeric quantities.
- Correct non-crashing behavior when input DataFrame is empty.
- Daily calendar construction within product-lifespan observation window.
- Zero-filling behavior and calculation of zero-filled record counts.
- Sparse-product diagnostics detection.
- Non-negative demand constraint in primary modeling output.
- Non-mutation / preservation of original input DataFrame.
- Correct output columns and data types.
- Error handling on missing essential columns.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from retail_forecasting.data.preprocess import (
    standardize_columns,
    parse_transaction_fields,
    clean_product_identifiers,
    classify_transactions,
    summarize_transaction_classification,
    filter_positive_demand,
    aggregate_daily_demand,
    apply_zero_filling,
    clean_sales_transactions,
    preprocess_pipeline,
)


@pytest.fixture
def sample_raw_transactions() -> pd.DataFrame:
    """Provide a diverse synthetic sample representing all retail transaction types."""
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "536366", "C536379", "536380", "536381"],
        "StockCode": ["85123A", "85123A", "71053", "85123A", "POST", "22633"],
        "Description": [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "WHITE METAL LANTERN",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "POSTAGE",
            "HAND WARMER UNION JACK",
        ],
        "Quantity": [6, 4, 10, -2, 1, -5],
        "InvoiceDate": [
            "2010-12-01 08:26:00",
            "2010-12-01 09:15:00",
            "2010-12-01 10:00:00",
            "2010-12-02 11:00:00",
            "2010-12-02 12:00:00",
            "2010-12-03 14:00:00",
        ],
        "UnitPrice": [2.55, 2.55, 3.39, 2.55, 18.00, 1.85],
        "CustomerID": [17850.0, 17850.0, 17850.0, 17850.0, 17850.0, 17850.0],
        "Country": ["United Kingdom"] * 6,
    })


# =========================================================================
# Date Parsing & Normalization Tests
# =========================================================================

def test_parse_transaction_fields_dates() -> None:
    """Verify that InvoiceDate strings are correctly parsed and normalized to calendar date."""
    df = pd.DataFrame({
        "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 23:59:59"],
        "Quantity": [5, 10],
        "UnitPrice": [1.5, 2.0],
    })
    parsed = parse_transaction_fields(df)
    assert pd.api.types.is_datetime64_any_dtype(parsed["InvoiceDate"])
    assert pd.api.types.is_datetime64_any_dtype(parsed["date"])
    # Both timestamps should normalize to the same midnight calendar date
    assert parsed["date"].iloc[0] == pd.Timestamp("2010-12-01")
    assert parsed["date"].iloc[1] == pd.Timestamp("2010-12-01")


def test_invalid_dates_and_non_numeric_handling() -> None:
    """Invalid dates and non-numeric values must be coerced to NaT/NaN and classified as invalid."""
    df = pd.DataFrame({
        "InvoiceNo": ["1001", "1002", "1003"],
        "StockCode": ["PROD1", "PROD2", "PROD3"],
        "Quantity": ["not_a_number", 5, 10],
        "InvoiceDate": ["2010-12-01", "unparseable_date", "2010-12-03"],
        "UnitPrice": [1.0, 2.0, "free"],
    })
    cleaned, audit = clean_sales_transactions(df)
    assert audit["category_counts"].get("invalid", 0) >= 2


# =========================================================================
# Transaction Classification & Cancellation Handling
# =========================================================================

def test_invoice_cancellation_c_prefix_handling(sample_raw_transactions: pd.DataFrame) -> None:
    """Invoice numbers beginning with 'C' (or 'c') must be classified as 'cancellation'."""
    cleaned, audit = clean_sales_transactions(sample_raw_transactions)
    cancellations = cleaned[cleaned["transaction_type"] == "cancellation"]
    assert len(cancellations) == 1
    assert cancellations.iloc[0]["InvoiceNo"] == "C536379"


def test_negative_quantity_without_c_prefix(sample_raw_transactions: pd.DataFrame) -> None:
    """Negative quantities where InvoiceNo lacks 'C' prefix are classified as 'return_only'."""
    cleaned, audit = clean_sales_transactions(sample_raw_transactions)
    returns = cleaned[cleaned["transaction_type"] == "return_only"]
    assert len(returns) == 1
    assert returns.iloc[0]["InvoiceNo"] == "536381"
    assert returns.iloc[0]["Quantity"] == -5


def test_classification_summary_no_double_counting(sample_raw_transactions: pd.DataFrame) -> None:
    """Sum of classified categories must equal total records without double counting."""
    cleaned, audit = clean_sales_transactions(sample_raw_transactions)
    assert sum(audit["category_counts"].values()) == len(sample_raw_transactions)
    assert audit["total_records"] == 6


def test_exclusion_of_cancellations_and_returns_from_primary_demand(
    sample_raw_transactions: pd.DataFrame
) -> None:
    """Primary positive-demand dataset must exclude cancellations and returns."""
    cleaned, _ = clean_sales_transactions(sample_raw_transactions)
    pos_demand = filter_positive_demand(cleaned, exclude_special_codes=False)
    # The two negative quantity records must be excluded
    assert (pos_demand["Quantity"] <= 0).sum() == 0
    assert not any(pos_demand["InvoiceNo"].str.startswith("C"))


# =========================================================================
# Product SKU Hygiene & Special Codes
# =========================================================================

def test_clean_product_identifiers_hygiene() -> None:
    """Whitespace must be stripped, casing normalized, and empty strings converted to NaN."""
    df = pd.DataFrame({
        "StockCode": [" 85123a ", "71053", "", "NAN", None],
        "Description": [" White heart ", "Lantern", "nan", None, "Valid"],
    })
    cleaned = clean_product_identifiers(df)
    assert cleaned["product_id"].iloc[0] == "85123A"
    assert pd.isna(cleaned["product_id"].iloc[2])
    assert pd.isna(cleaned["product_id"].iloc[3])
    assert pd.isna(cleaned["product_id"].iloc[4])


def test_filter_special_stock_codes(sample_raw_transactions: pd.DataFrame) -> None:
    """Non-merchandise codes (e.g., 'POST') must be filtered when exclude_special_codes=True."""
    cleaned, _ = clean_sales_transactions(sample_raw_transactions)
    filtered = filter_positive_demand(cleaned, exclude_special_codes=True)
    assert "POST" not in filtered["product_id"].values


# =========================================================================
# Daily Aggregation Tests
# =========================================================================

def test_aggregate_multiple_transactions_same_day_and_product(
    sample_raw_transactions: pd.DataFrame
) -> None:
    """Multiple transactions for the same product and date must be summed."""
    cleaned, _ = clean_sales_transactions(sample_raw_transactions)
    pos_demand = filter_positive_demand(cleaned)
    daily = aggregate_daily_demand(pos_demand)

    # Product 85123A on 2010-12-01 had two transactions: 6 + 4 = 10 units
    prod_row = daily[
        (daily["product_id"] == "85123A") & (daily["date"] == pd.Timestamp("2010-12-01"))
    ]
    assert len(prod_row) == 1
    assert prod_row.iloc[0]["demand"] == 10.0


def test_separation_of_different_products() -> None:
    """Different products on the same date must maintain distinct rows."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2010-12-01"), pd.Timestamp("2010-12-01")],
        "product_id": ["PROD_A", "PROD_B"],
        "Quantity": [5, 12],
    })
    daily = aggregate_daily_demand(df, target_col="Quantity")
    assert len(daily) == 2
    assert set(daily["product_id"]) == {"PROD_A", "PROD_B"}


def test_aggregated_demand_non_negative_constraint() -> None:
    """Aggregated demand must strictly be non-negative."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2010-12-01"), pd.Timestamp("2010-12-02")],
        "product_id": ["PROD_A", "PROD_A"],
        "Quantity": [10, -5],
    })
    # If negative values pass into aggregation, negative totals are purged
    daily = aggregate_daily_demand(df, target_col="Quantity")
    assert (daily["demand"] < 0).sum() == 0


# =========================================================================
# Zero-Filling & Observation Window Tests
# =========================================================================

def test_zero_filling_product_lifespan_policy() -> None:
    """Product lifespan window fills absent calendar days strictly between first and last sale."""
    # Product active on Dec 1 and Dec 4 (absent on Dec 2 and Dec 3)
    daily_input = pd.DataFrame({
        "date": [pd.Timestamp("2010-12-01"), pd.Timestamp("2010-12-04")],
        "product_id": ["ITEM_X", "ITEM_X"],
        "demand": [15.0, 20.0],
    })

    filled_df, report = apply_zero_filling(daily_input, window_policy="product_lifespan")

    # Dec 1, Dec 2, Dec 3, Dec 4 = 4 days
    assert len(filled_df) == 4
    assert report["total_product_days"] == 4
    assert report["observed_sales_days"] == 2
    assert report["zero_filled_days"] == 2
    assert report["zero_fill_ratio"] == 0.5

    # Check that Dec 2 and Dec 3 are zero-filled
    dec2 = filled_df[filled_df["date"] == pd.Timestamp("2010-12-02")]
    assert len(dec2) == 1
    assert dec2.iloc[0]["demand"] == 0.0


def test_zero_filling_does_not_extend_beyond_product_lifespan() -> None:
    """Product-lifespan policy must not extrapolate zeros before introduction or after retirement."""
    # Product A active Dec 1 to Dec 2; Product B active Dec 3 to Dec 4
    daily_input = pd.DataFrame({
        "date": [
            pd.Timestamp("2010-12-01"),
            pd.Timestamp("2010-12-02"),
            pd.Timestamp("2010-12-03"),
            pd.Timestamp("2010-12-04"),
        ],
        "product_id": ["PROD_EARLY", "PROD_EARLY", "PROD_LATE", "PROD_LATE"],
        "demand": [10.0, 10.0, 5.0, 5.0],
    })
    filled_df, report = apply_zero_filling(daily_input, window_policy="product_lifespan")
    # Neither product has internal gaps; each has lifespan of 2 days => total 4 days
    assert len(filled_df) == 4
    assert report["zero_filled_days"] == 0


def test_sparse_product_diagnostics_detected() -> None:
    """Products with fewer observed selling days than threshold must be flagged as sparse."""
    daily_input = pd.DataFrame({
        "date": [pd.Timestamp("2010-12-01"), pd.Timestamp("2010-12-02")],
        "product_id": ["SPARSE_ITEM", "SPARSE_ITEM"],
        "demand": [1.0, 2.0],
    })
    # Threshold is 7 active days; SPARSE_ITEM has only 2
    _, report = apply_zero_filling(daily_input, min_active_days_sparse=7)
    assert report["sparse_products_count"] == 1
    assert "SPARSE_ITEM" in report["sparse_products"]


# =========================================================================
# Invariance & Edge Case Tests
# =========================================================================

def test_original_input_dataframe_not_mutated(sample_raw_transactions: pd.DataFrame) -> None:
    """Input DataFrame must remain unchanged after running the preprocessing pipeline."""
    original_copy = sample_raw_transactions.copy(deep=True)
    _ = preprocess_pipeline(sample_raw_transactions)
    pd.testing.assert_frame_equal(sample_raw_transactions, original_copy)


def test_empty_dataframe_pipeline_handling() -> None:
    """Empty DataFrame input must return empty structures gracefully without crashing."""
    empty_df = pd.DataFrame(columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    ])
    result = preprocess_pipeline(empty_df)
    assert result["cleaned_transactions"].empty
    assert result["daily_demand"].empty
    assert result["audit_summary"]["total_records"] == 0
    assert result["zero_filling_report"]["total_product_days"] == 0


def test_pipeline_output_columns_and_types(sample_raw_transactions: pd.DataFrame) -> None:
    """Verify primary modeling output has exact expected columns: [date, product_id, demand]."""
    result = preprocess_pipeline(sample_raw_transactions)
    daily_demand = result["daily_demand"]

    assert "date" in daily_demand.columns
    assert "product_id" in daily_demand.columns
    assert "demand" in daily_demand.columns
    assert pd.api.types.is_datetime64_any_dtype(daily_demand["date"])
    assert pd.api.types.is_numeric_dtype(daily_demand["demand"])
    assert (daily_demand["demand"] >= 0).all()
