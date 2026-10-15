"""Unit tests for DataCo preprocessing and demand aggregation module."""

import pytest
import pandas as pd
import numpy as np

from retail_forecasting.data.validate_data import DATACO_REQUIRED_COLUMNS
from retail_forecasting.data.preprocess_dataco import (
    load_raw_dataco_data,
    validate_dataco_raw_columns,
    parse_dataco_dates_and_keys,
    apply_demand_inclusion_policy,
    aggregate_dataco_daily_demand,
    extract_dataco_product_catalog,
    verify_dataco_preprocessing_reconciliation,
    DEFAULT_DATACO_EXCLUDED_STATUSES,
)


@pytest.fixture
def synthetic_dataco_raw() -> pd.DataFrame:
    """Fixture providing a minimal, controlled synthetic DataCo transaction DataFrame."""
    rows = []
    # Base template for all required columns
    base_record = {col: "sample" for col in DATACO_REQUIRED_COLUMNS}

    # Item 1: Complete sale on 2016-01-01
    r1 = base_record.copy()
    r1.update({
        "Order Id": 101,
        "Order Item Id": 1001,
        "Product Card Id": 501,
        "Product Name": "Test Cleats",
        "Product Price": 50.0,
        "Category Id": 10,
        "Category Name": "Soccer",
        "Department Id": 2,
        "Department Name": "Fitness",
        "Order Item Quantity": 2,
        "Order Status": "COMPLETE",
        "Delivery Status": "Advance shipping",
        "order date (DateOrders)": "2016-01-01 09:30:00",
        "shipping date (DateOrders)": "2016-01-03 10:00:00",
        "Customer Email": "XXXXXXXXX",
        "Customer Password": "XXXXXXXXX",
        "Customer Fname": "John",
        "Customer Lname": "Doe",
        "Customer Street": "123 Main St",
    })
    rows.append(r1)

    # Item 2: Another complete sale on 2016-01-01 for same product
    r2 = base_record.copy()
    r2.update({
        "Order Id": 102,
        "Order Item Id": 1002,
        "Product Card Id": 501,
        "Product Name": "Test Cleats",
        "Product Price": 50.0,
        "Category Id": 10,
        "Category Name": "Soccer",
        "Department Id": 2,
        "Department Name": "Fitness",
        "Order Item Quantity": 3,
        "Order Status": "PENDING_PAYMENT",
        "Delivery Status": "Late delivery",
        "order date (DateOrders)": "2016-01-01 14:00:00",
        "shipping date (DateOrders)": "2016-01-04 10:00:00",
        "Customer Email": "XXXXXXXXX",
        "Customer Password": "XXXXXXXXX",
        "Customer Fname": "Jane",
        "Customer Lname": "Smith",
        "Customer Street": "456 Elm St",
    })
    rows.append(r2)

    # Item 3: Canceled order on 2016-01-01 for same product
    r3 = base_record.copy()
    r3.update({
        "Order Id": 103,
        "Order Item Id": 1003,
        "Product Card Id": 501,
        "Product Name": "Test Cleats",
        "Product Price": 50.0,
        "Category Id": 10,
        "Category Name": "Soccer",
        "Department Id": 2,
        "Department Name": "Fitness",
        "Order Item Quantity": 5,
        "Order Status": "CANCELED",
        "Delivery Status": "Shipping canceled",
        "order date (DateOrders)": "2016-01-01 16:00:00",
        "shipping date (DateOrders)": "2016-01-01 16:00:00",
        "Customer Email": "XXXXXXXXX",
        "Customer Password": "XXXXXXXXX",
        "Customer Fname": "Bob",
        "Customer Lname": "Brown",
        "Customer Street": "789 Pine St",
    })
    rows.append(r3)

    # Item 4: Suspected Fraud on 2016-01-02
    r4 = base_record.copy()
    r4.update({
        "Order Id": 104,
        "Order Item Id": 1004,
        "Product Card Id": 502,
        "Product Name": "Test Jersey",
        "Product Price": 80.0,
        "Category Id": 11,
        "Category Name": "Apparel",
        "Department Id": 3,
        "Department Name": "Outdoors",
        "Order Item Quantity": 4,
        "Order Status": "SUSPECTED_FRAUD",
        "Delivery Status": "Shipping canceled",
        "order date (DateOrders)": "2016-01-02 11:00:00",
        "shipping date (DateOrders)": "2016-01-02 11:00:00",
        "Customer Email": "XXXXXXXXX",
        "Customer Password": "XXXXXXXXX",
        "Customer Fname": "Alice",
        "Customer Lname": "White",
        "Customer Street": "101 Maple St",
    })
    rows.append(r4)

    # Item 5: Complete sale for product 502 on 2016-01-02
    r5 = base_record.copy()
    r5.update({
        "Order Id": 105,
        "Order Item Id": 1005,
        "Product Card Id": 502,
        "Product Name": "Test Jersey",
        "Product Price": 80.0,
        "Category Id": 11,
        "Category Name": "Apparel",
        "Department Id": 3,
        "Department Name": "Outdoors",
        "Order Item Quantity": 1,
        "Order Status": "COMPLETE",
        "Delivery Status": "Shipping on time",
        "order date (DateOrders)": "2016-01-02 12:00:00",
        "shipping date (DateOrders)": "2016-01-05 12:00:00",
        "Customer Email": "XXXXXXXXX",
        "Customer Password": "XXXXXXXXX",
        "Customer Fname": "Charlie",
        "Customer Lname": "Green",
        "Customer Street": "202 Cedar St",
    })
    rows.append(r5)

    return pd.DataFrame(rows)


def test_validate_dataco_raw_columns(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies that all 53 canonical columns are required."""
    assert validate_dataco_raw_columns(synthetic_dataco_raw) is True
    dropped = synthetic_dataco_raw.drop(columns=["Product Card Id"])
    with pytest.raises(ValueError, match="missing 1 required columns"):
        validate_dataco_raw_columns(dropped)


def test_parse_dataco_dates_and_keys(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies safe date parsing and string product ID formatting."""
    parsed = parse_dataco_dates_and_keys(synthetic_dataco_raw)
    assert "date" in parsed.columns
    assert "product_id" in parsed.columns
    assert "quantity" in parsed.columns
    assert parsed["product_id"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(parsed["date"])
    assert parsed["date"].iloc[0] == pd.Timestamp("2016-01-01")


def test_apply_demand_inclusion_policy(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies that CANCELED and SUSPECTED_FRAUD are excluded while other statuses are included."""
    parsed = parse_dataco_dates_and_keys(synthetic_dataco_raw)
    included, excluded, audit = apply_demand_inclusion_policy(parsed)

    # Rows 1, 2, 5 included (total 3 rows, total quantity 2 + 3 + 1 = 6)
    assert len(included) == 3
    assert audit["included_rows"] == 3
    assert audit["included_quantity"] == 6.0

    # Rows 3 (CANCELED, qty 5) and 4 (SUSPECTED_FRAUD, qty 4) excluded (total 2 rows, qty 9)
    assert len(excluded) == 2
    assert audit["excluded_rows"] == 2
    assert audit["excluded_quantity"] == 9.0

    assert set(excluded["Order Status"]) == {"CANCELED", "SUSPECTED_FRAUD"}


def test_aggregate_dataco_daily_demand(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies daily aggregation sums quantities correctly and prevents duplicate (date, product_id)."""
    parsed = parse_dataco_dates_and_keys(synthetic_dataco_raw)
    included, _, _ = apply_demand_inclusion_policy(parsed)
    agg = aggregate_dataco_daily_demand(included)

    # Product 501 on 2016-01-01 had two sales: 2 + 3 = 5
    # Product 502 on 2016-01-02 had one sale: 1
    assert len(agg) == 2
    assert list(agg.columns) == ["date", "product_id", "demand"]

    row_501 = agg[(agg["product_id"] == "501") & (agg["date"] == "2016-01-01")]
    assert len(row_501) == 1
    assert row_501["demand"].iloc[0] == 5.0

    row_502 = agg[(agg["product_id"] == "502") & (agg["date"] == "2016-01-02")]
    assert len(row_502) == 1
    assert row_502["demand"].iloc[0] == 1.0

    # Duplicate check
    assert agg.duplicated(subset=["date", "product_id"]).sum() == 0


def test_extract_dataco_product_catalog_privacy(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies product catalog extraction and confirms absence of any customer PII."""
    catalog = extract_dataco_product_catalog(synthetic_dataco_raw)
    assert len(catalog) == 2
    assert set(catalog.columns) == {
        "product_id",
        "product_name",
        "product_price",
        "category_id",
        "category_name",
        "department_id",
        "department_name",
    }
    # No PII columns
    for col in catalog.columns:
        assert "customer" not in col.lower()
        assert "email" not in col.lower()
        assert "password" not in col.lower()
        assert "street" not in col.lower()


def test_verify_dataco_preprocessing_reconciliation(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies that mathematical accounting checks pass on valid runs and catch mismatches."""
    parsed = parse_dataco_dates_and_keys(synthetic_dataco_raw)
    included, excluded, audit = apply_demand_inclusion_policy(parsed)
    agg = aggregate_dataco_daily_demand(included)

    recon = verify_dataco_preprocessing_reconciliation(
        synthetic_dataco_raw, included, excluded, agg, audit
    )
    assert all(recon.values())

    # Tampered aggregated quantity should trigger ValueError
    tampered_agg = agg.copy()
    tampered_agg.loc[0, "demand"] += 10.0
    with pytest.raises(ValueError, match="Aggregated demand mismatch"):
        verify_dataco_preprocessing_reconciliation(
            synthetic_dataco_raw, included, excluded, tampered_agg, audit
        )


def test_audit_dataco_demand_sensitivity(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies that sensitivity auditing computes metrics accurately across policies."""
    from retail_forecasting.data.preprocess_dataco import audit_dataco_demand_sensitivity
    
    sens_df = audit_dataco_demand_sensitivity(synthetic_dataco_raw)
    assert len(sens_df) == 4
    assert "policy_name" in sens_df.columns
    assert "included_rows" in sens_df.columns
    assert "included_units" in sens_df.columns
    
    # Baseline policy (Policy A) should have 3 included rows, 6 units
    row_a = sens_df[sens_df["policy_name"].str.contains("Policy A")].iloc[0]
    assert row_a["included_rows"] == 3
    assert row_a["included_units"] == 6.0

    # Policy C (COMPLETE only) should have 2 included rows (r1, r5), 3 units
    row_c = sens_df[sens_df["policy_name"].str.contains("Policy C")].iloc[0]
    assert row_c["included_rows"] == 2
    assert row_c["included_units"] == 3.0


def test_audit_dataco_timeseries_readiness(synthetic_dataco_raw: pd.DataFrame) -> None:
    """Verifies that timeseries readiness audit correctly calculates span, missing days, and sparsity."""
    from retail_forecasting.data.preprocess_dataco import (
        parse_dataco_dates_and_keys,
        apply_demand_inclusion_policy,
        aggregate_dataco_daily_demand,
        audit_dataco_timeseries_readiness,
    )
    
    parsed = parse_dataco_dates_and_keys(synthetic_dataco_raw)
    included, _, _ = apply_demand_inclusion_policy(parsed)
    agg = aggregate_dataco_daily_demand(included)
    
    readiness_df = audit_dataco_timeseries_readiness(agg)
    assert len(readiness_df) == 2
    assert set(readiness_df.columns) == {
        "product_id", "first_date", "last_date", "obs_sale_days",
        "span_days", "missing_days", "sparsity_pct", "total_demand"
    }
    
    # Product 501 was active on 2016-01-01 (1 day, 0 missing days)
    row_501 = readiness_df[readiness_df["product_id"] == "501"].iloc[0]
    assert row_501["obs_sale_days"] == 1
    assert row_501["span_days"] == 1
    assert row_501["missing_days"] == 0
    assert row_501["sparsity_pct"] == 0.0
