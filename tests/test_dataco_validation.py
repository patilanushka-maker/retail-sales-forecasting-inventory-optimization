"""Unit tests for DataCo dataset validation utilities."""

import pytest
import pandas as pd
import numpy as np

from retail_forecasting.data.validate_data import (
    DATACO_REQUIRED_COLUMNS,
    validate_dataco_dataset,
    validate_schema,
)


@pytest.fixture
def valid_dataco_sample() -> pd.DataFrame:
    """Fixture providing a valid synthetic sample of DataCo Smart Supply Chain records."""
    data = {col: ["sample"] * 3 for col in DATACO_REQUIRED_COLUMNS}
    # Provide proper types for key columns
    data["Days for shipping (real)"] = [3, 2, 5]
    data["Days for shipment (scheduled)"] = [4, 2, 4]
    data["Benefit per order"] = [12.5, -5.0, 45.2]
    data["Sales per customer"] = [100.0, 50.0, 200.0]
    data["Late_delivery_risk"] = [0, 0, 1]
    data["Category Id"] = [17, 17, 24]
    data["Customer Id"] = [101, 102, 103]
    data["Department Id"] = [2, 2, 3]
    data["Latitude"] = [18.2, 18.3, 18.4]
    data["Longitude"] = [-66.1, -66.2, -66.3]
    data["Order Customer Id"] = [101, 102, 103]
    data["order date (DateOrders)"] = ["2016-01-01 10:00:00", "2016-01-02 12:00:00", "2016-01-03 14:00:00"]
    data["Order Id"] = [1001, 1002, 1003]
    data["Order Item Cardprod Id"] = [501, 502, 501]
    data["Order Item Discount"] = [5.0, 0.0, 10.0]
    data["Order Item Discount Rate"] = [0.05, 0.0, 0.10]
    data["Order Item Id"] = [2001, 2002, 2003]
    data["Order Item Product Price"] = [100.0, 50.0, 100.0]
    data["Order Item Profit Ratio"] = [0.125, -0.10, 0.226]
    data["Order Item Quantity"] = [1, 2, 1]
    data["Sales"] = [100.0, 100.0, 100.0]
    data["Order Item Total"] = [95.0, 100.0, 90.0]
    data["Order Profit Per Order"] = [12.5, -5.0, 45.2]
    data["Product Card Id"] = [501, 502, 501]
    data["Product Category Id"] = [17, 17, 24]
    data["Product Price"] = [100.0, 50.0, 100.0]
    data["Product Status"] = [0, 0, 0]
    data["shipping date (DateOrders)"] = ["2016-01-04 10:00:00", "2016-01-04 12:00:00", "2016-01-08 14:00:00"]
    return pd.DataFrame(data)


def test_dataco_schema_has_53_columns() -> None:
    """Verify DATACO_REQUIRED_COLUMNS has exactly 53 unique fields."""
    assert len(DATACO_REQUIRED_COLUMNS) == 53
    assert len(set(DATACO_REQUIRED_COLUMNS)) == 53


def test_valid_dataco_sample_passes_validation(valid_dataco_sample: pd.DataFrame) -> None:
    """A valid synthetic DataCo dataframe should pass validation."""
    report = validate_dataco_dataset(valid_dataco_sample, strict=True)
    assert report["is_valid"] is True
    assert report["schema_ok"] is True
    assert report["distinct_orders"] == 3
    assert report["distinct_products"] == 2
    assert report["chronological_inversions"] == 0
    assert len(report["errors"]) == 0


def test_dataco_missing_column_raises(valid_dataco_sample: pd.DataFrame) -> None:
    """Dropping a required DataCo column should raise ValueError in schema validation."""
    invalid_df = valid_dataco_sample.drop(columns=["Product Card Id"])
    with pytest.raises(ValueError, match="Missing 1 required column"):
        validate_dataco_dataset(invalid_df, strict=True)


def test_dataco_unparseable_date_raises(valid_dataco_sample: pd.DataFrame) -> None:
    """Corrupting order date should trigger an error under strict validation."""
    invalid_df = valid_dataco_sample.copy()
    invalid_df.loc[0, "order date (DateOrders)"] = "not-a-valid-date"
    with pytest.raises(ValueError, match="unparseable 'order date"):
        validate_dataco_dataset(invalid_df, strict=True)


def test_dataco_chronological_inversion_warns(valid_dataco_sample: pd.DataFrame) -> None:
    """If shipping date precedes order date, a warning is recorded without failing."""
    warn_df = valid_dataco_sample.copy()
    # Shipping date before order date
    warn_df.loc[0, "shipping date (DateOrders)"] = "2015-12-31 10:00:00"
    report = validate_dataco_dataset(warn_df, strict=False)
    assert report["is_valid"] is True
    assert report["chronological_inversions"] == 1
    assert any("shipping dates preceding order dates" in w for w in report["warnings"])


def test_dataco_non_positive_quantity_warns(valid_dataco_sample: pd.DataFrame) -> None:
    """Non-positive quantities produce diagnostic warnings."""
    warn_df = valid_dataco_sample.copy()
    warn_df.loc[0, "Order Item Quantity"] = 0
    report = validate_dataco_dataset(warn_df, strict=False)
    assert any("non-positive 'Order Item Quantity'" in w for w in report["warnings"])
