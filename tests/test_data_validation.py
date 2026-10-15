"""Unit tests for data loading and schema validation modules.

Validates that:
- A valid DataFrame passes schema validation.
- Missing required columns are detected with informative errors.
- Empty DataFrames are rejected.
- Missing values and duplicate rows are reported accurately without mutating data.
- Unparseable dates and non-numeric values are caught.
- Missing input files raise FileNotFoundError.
- Supported files (.csv, .parquet) load correctly.
- Unsupported file extensions raise ValueError.
- Extra columns are accepted when allow_extra_columns=True.
- Informative error is raised if Excel reading is attempted without openpyxl.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from retail_forecasting.data.load_data import (
    load_raw_sales_data,
    SUPPORTED_EXTENSIONS,
)
from retail_forecasting.data.validate_data import (
    ONLINE_RETAIL_REQUIRED_COLUMNS,
    validate_schema,
    validate_non_empty,
    check_missing_and_duplicates,
    validate_types_and_values,
    validate_dataset,
)


@pytest.fixture
def valid_retail_df() -> pd.DataFrame:
    """Fixture providing a valid synthetic sample of UCI Online Retail records."""
    return pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "536366"],
        "StockCode": ["85123A", "71053", "22633"],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN", "HAND WARMER UNION JACK"],
        "Quantity": [6, 6, 8],
        "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:26:00", "2010-12-01 08:28:00"],
        "UnitPrice": [2.55, 3.39, 1.85],
        "CustomerID": [17850.0, 17850.0, 17850.0],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom"],
    })


# =========================================================================
# Schema & Empty Validation Tests
# =========================================================================

def test_valid_dataframe_passes_schema(valid_retail_df: pd.DataFrame) -> None:
    """A DataFrame containing all required columns must pass schema validation."""
    assert validate_schema(valid_retail_df) is True


def test_missing_required_column_detected(valid_retail_df: pd.DataFrame) -> None:
    """Dropping a required column must raise ValueError specifying the missing column."""
    df_missing = valid_retail_df.drop(columns=["UnitPrice"])
    with pytest.raises(ValueError, match="Missing 1 required column.*UnitPrice"):
        validate_schema(df_missing)


def test_multiple_missing_required_columns(valid_retail_df: pd.DataFrame) -> None:
    """Dropping multiple required columns reports all missing columns in error."""
    df_missing = valid_retail_df.drop(columns=["StockCode", "Quantity"])
    with pytest.raises(ValueError, match="Missing 2 required column.*StockCode.*Quantity"):
        validate_schema(df_missing)


def test_extra_columns_accepted_by_default(valid_retail_df: pd.DataFrame) -> None:
    """Extra unexpected columns must be accepted by default."""
    df_extra = valid_retail_df.copy()
    df_extra["ExtraCol1"] = "Bonus"
    df_extra["ExtraCol2"] = 123
    assert validate_schema(df_extra, allow_extra_columns=True) is True


def test_extra_columns_rejected_when_forbidden(valid_retail_df: pd.DataFrame) -> None:
    """Extra columns must raise ValueError if allow_extra_columns=False."""
    df_extra = valid_retail_df.copy()
    df_extra["UnexpectedCol"] = 999
    with pytest.raises(ValueError, match="Disallowed extra column"):
        validate_schema(df_extra, allow_extra_columns=False)


def test_empty_dataframe_rejected() -> None:
    """Empty DataFrames must be rejected with an informative ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="contains 0 rows"):
        validate_non_empty(empty_df)

    empty_rows_df = pd.DataFrame(columns=["InvoiceNo", "StockCode"])
    with pytest.raises(ValueError, match="contains 0 rows"):
        validate_non_empty(empty_rows_df)


def test_non_empty_dataframe_accepted(valid_retail_df: pd.DataFrame) -> None:
    """Non-empty DataFrames must pass validate_non_empty."""
    assert validate_non_empty(valid_retail_df) is True


# =========================================================================
# Missing Values & Duplicates Diagnostics Tests
# =========================================================================

def test_missing_values_reported_accurately() -> None:
    """Missing value counts and percentages must be reported accurately without mutating data."""
    df = pd.DataFrame({
        "InvoiceNo": ["536365", "536366", "536367"],
        "StockCode": ["85123A", None, "22633"],
        "CustomerID": [17850.0, np.nan, np.nan],
    })
    original_copy = df.copy(deep=True)

    report = check_missing_and_duplicates(df, key_columns=["InvoiceNo", "StockCode"])

    assert report["total_rows"] == 3
    assert report["missing_counts"]["StockCode"] == 1
    assert report["missing_counts"]["CustomerID"] == 2
    assert report["missing_counts"]["InvoiceNo"] == 0
    assert report["missing_percentages"]["StockCode"] == pytest.approx(33.33, rel=1e-2)
    assert report["missing_percentages"]["CustomerID"] == pytest.approx(66.67, rel=1e-2)

    # Confirm original dataframe was not mutated
    pd.testing.assert_frame_equal(df, original_copy)


def test_duplicates_reported_accurately() -> None:
    """Duplicate rows must be detected and counted."""
    df = pd.DataFrame({
        "InvoiceNo": ["536365", "536365", "536366"],
        "StockCode": ["85123A", "85123A", "22633"],
    })
    report = check_missing_and_duplicates(df)
    assert report["total_duplicate_rows"] == 1


# =========================================================================
# Type & Value Validation Tests
# =========================================================================

def test_invalid_dates_detected() -> None:
    """Unparseable dates must be caught and reported."""
    df = pd.DataFrame({
        "InvoiceNo": ["101", "102"],
        "InvoiceDate": ["2010-12-01 08:26:00", "not-a-valid-date-string"],
        "Quantity": [5, 10],
        "UnitPrice": [1.5, 2.0],
    })
    report = validate_types_and_values(df, strict=False)
    assert report["is_valid"] is False
    assert report["invalid_date_count"] == 1
    assert any("InvoiceDate" in err for err in report["errors"])

    with pytest.raises(ValueError, match="unparseable 'InvoiceDate'"):
        validate_types_and_values(df, strict=True)


def test_invalid_numeric_values_detected() -> None:
    """Non-numeric quantities or unit prices must be caught and reported."""
    df = pd.DataFrame({
        "InvoiceNo": ["101", "102"],
        "InvoiceDate": ["2010-12-01", "2010-12-02"],
        "Quantity": ["invalid_qty", "10"],
        "UnitPrice": ["$12.50", "2.00"],
    })
    report = validate_types_and_values(df, strict=False)
    assert report["is_valid"] is False
    assert report["invalid_quantity_count"] == 1
    assert report["invalid_unit_price_count"] == 1

    with pytest.raises(ValueError, match="non-numeric"):
        validate_types_and_values(df, strict=True)


def test_negative_quantities_reported_as_warnings_not_errors(valid_retail_df: pd.DataFrame) -> None:
    """Negative quantities (cancellations/returns) must be reported as warnings without failing schema."""
    df = valid_retail_df.copy()
    df.loc[0, "Quantity"] = -5  # cancellation / return

    report = validate_types_and_values(df, strict=False)
    assert report["is_valid"] is True
    assert report["negative_quantity_count"] == 1
    assert any("cancellations/returns" in w for w in report["warnings"])


def test_validate_dataset_end_to_end(valid_retail_df: pd.DataFrame) -> None:
    """Complete validation pipeline must pass on clean synthetic data."""
    summary = validate_dataset(valid_retail_df, strict=True)
    assert summary["is_valid"] is True
    assert summary["schema_ok"] is True
    assert summary["non_empty_ok"] is True
    assert summary["missing_and_duplicates"]["total_rows"] == 3


# =========================================================================
# File Loading Tests
# =========================================================================

def test_missing_input_file_raises_error(tmp_path: Path) -> None:
    """Loading a non-existent file path must raise FileNotFoundError with clear message."""
    missing_file = tmp_path / "non_existent_sales_data.csv"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_raw_sales_data(missing_file)


def test_directory_as_filepath_raises_error(tmp_path: Path) -> None:
    """Passing a directory instead of a file must raise ValueError."""
    with pytest.raises(ValueError, match="is a directory, not a file"):
        load_raw_sales_data(tmp_path)


def test_unsupported_file_extension_raises_error(tmp_path: Path) -> None:
    """Passing an unsupported file extension must raise ValueError."""
    bad_file = tmp_path / "sales.txt"
    bad_file.write_text("sample content", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format '.txt'"):
        load_raw_sales_data(bad_file)


def test_load_supported_csv_file(tmp_path: Path, valid_retail_df: pd.DataFrame) -> None:
    """CSV file must be read into a DataFrame matching original data."""
    csv_file = tmp_path / "online_retail.csv"
    valid_retail_df.to_csv(csv_file, index=False)

    loaded_df = load_raw_sales_data(csv_file)
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == len(valid_retail_df)
    assert list(loaded_df.columns) == list(valid_retail_df.columns)


def test_load_supported_parquet_file(tmp_path: Path, valid_retail_df: pd.DataFrame) -> None:
    """Parquet file must be read into a DataFrame matching original data."""
    parquet_file = tmp_path / "online_retail.parquet"
    valid_retail_df.to_parquet(parquet_file, index=False)

    loaded_df = load_raw_sales_data(parquet_file)
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == len(valid_retail_df)
    assert list(loaded_df.columns) == list(valid_retail_df.columns)


def test_load_excel_without_openpyxl_raises_informative_error(tmp_path: Path) -> None:
    """Attempting to load an .xlsx file when openpyxl is not installed raises clear ImportError."""
    xlsx_file = tmp_path / "online_retail.xlsx"
    xlsx_file.write_bytes(b"dummy excel bytes")

    # If openpyxl is not installed in the environment, verify the helpful message
    try:
        import openpyxl  # type: ignore
        pytest.skip("openpyxl is installed; skipping missing-dependency test.")
    except ImportError:
        with pytest.raises(ImportError, match="requires the 'openpyxl' package"):
            load_raw_sales_data(xlsx_file)
