"""Data ingestion module for retail demand datasets.

Provides utility functions to safely load tabular sales records, product catalogs,
and transaction tables from disk supporting CSV, Parquet, and Excel (.xlsx/.xls) formats.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}


def load_raw_sales_data(
    filepath: Union[str, Path],
    storage_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """Load raw sales transactions from disk into a Pandas DataFrame.

    Supports CSV, Parquet, and Excel (.xlsx, .xls) files. Does not mutate
    the raw source file on disk.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the raw data file.
    storage_options : Optional[Dict[str, Any]]
        Optional storage options for remote or cloud paths.
    **kwargs : Any
        Additional keyword arguments forwarded to the underlying pandas reader
        (e.g., nrows, sheet_name, parse_dates, dtype).

    Returns
    -------
    pd.DataFrame
        Loaded raw transaction dataframe.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the file extension is unsupported or path is not a file.
    ImportError
        If an Excel file is provided but 'openpyxl' is not installed.
    IOError
        If reading fails due to file corruption or invalid contents.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"The specified raw data file does not exist: '{path.resolve()}'"
        )

    if not path.is_file():
        raise ValueError(
            f"The specified path is a directory, not a file: '{path.resolve()}'"
        )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported extensions are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if suffix == ".csv":
            df = pd.read_csv(path, storage_options=storage_options, **kwargs)
        elif suffix == ".parquet":
            df = pd.read_parquet(path, storage_options=storage_options, **kwargs)
        elif suffix in {".xlsx", ".xls"}:
            try:
                import openpyxl  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    "Reading Excel files (.xlsx/.xls) requires the 'openpyxl' package. "
                    "Please install it using 'pip install openpyxl' or install dependencies "
                    "from requirements.txt."
                ) from err
            df = pd.read_excel(path, storage_options=storage_options, **kwargs)
        else:
            # Defensive branch (already checked against SUPPORTED_EXTENSIONS)
            raise ValueError(f"Unhandled file extension: '{suffix}'")

    except (FileNotFoundError, ValueError, ImportError):
        raise
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"File '{path.name}' is empty and contains no readable data.") from err
    except Exception as err:
        raise IOError(
            f"Failed to read data file '{path.name}' ({type(err).__name__}): {err}"
        ) from err

    if df is None:
        raise ValueError(f"Reader returned None for file: '{path.name}'")

    return df
