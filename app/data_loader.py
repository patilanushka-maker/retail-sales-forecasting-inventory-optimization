"""Cached data loaders and schema validators for the Streamlit dashboard.

Loads and validates:
1. Processed daily product demand (Parquet).
2. Out-of-sample forecast comparisons (Parquet).
3. Forecast evaluation metrics (JSON).
4. Retrospective inventory recommendations (Parquet / CSV).
5. Filtered replenishment purchase orders (CSV).

Path Resolution & Invariants:
- All paths are resolved relative to the project root directory.
- Cached using @st.cache_data for instant interactive responses.
- Explicit schema, date, null, and duplicate key validations prevent opaque UI failures.
- Legitimate nulls in unavailable recommendations are preserved, not silently zero-filled.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import streamlit as st


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


# Default artifact paths relative to project root
DEFAULT_DEMAND_PATH = "data/processed/daily_product_demand.parquet"
DEFAULT_FORECASTS_PATH = "outputs/forecasts/forecast_comparison.parquet"
DEFAULT_METRICS_PATH = "outputs/metrics/forecast_metrics.json"
DEFAULT_RECOMMENDATIONS_PATH = "outputs/inventory/inventory_recommendations.parquet"
DEFAULT_ORDERS_PATH = "outputs/inventory/replenishment_orders.csv"

# Default DataCo artifact paths relative to project root
DEFAULT_DATACO_DEMAND_PATH = "data/processed/dataco_daily_product_demand.parquet"
DEFAULT_DATACO_CATALOG_PATH = "data/processed/dataco_product_catalog.parquet"
DEFAULT_DATACO_FORECASTS_PATH = "outputs/dataco/forecasts/dataco_forecast_comparison.parquet"
DEFAULT_DATACO_METRICS_PATH = "outputs/dataco/metrics/dataco_forecast_metrics.json"
DEFAULT_DATACO_VALIDATION_FORECASTS_PATH = "outputs/dataco/forecasts/dataco_validation_forecasts.parquet"
DEFAULT_DATACO_FIXED_FORECASTS_PATH = "outputs/dataco/inventory/dataco_fixed_origin_forecasts.parquet"
DEFAULT_DATACO_RECOMMENDATIONS_PATH = "outputs/dataco/inventory/dataco_inventory_recommendations.parquet"
DEFAULT_DATACO_ORDERS_PATH = "outputs/dataco/inventory/dataco_replenishment_orders.csv"



REQUIRED_DEMAND_COLS = ["date", "product_id", "demand"]
REQUIRED_FORECAST_COLS = [
    "date",
    "product_id",
    "actual_demand",
    "random_forest_pred",
    "seasonal_naive_pred",
]
REQUIRED_RECOMMENDATION_COLS = [
    "product_id",
    "as_of_date",
    "forecast_start_date",
    "forecast_end_date",
    "is_retrospective_demonstration",
    "recommendation_type",
    "forecast_model",
    "forecast_horizon_days",
    "service_level",
    "z_score",
    "lead_time_days",
    "review_period_days",
    "protection_period_days",
    "is_simulated_operational_input",
    "expected_daily_demand",
    "expected_lead_time_demand",
    "expected_protection_period_demand",
    "forecast_error_std",
    "error_history_count",
    "is_uncertainty_fallback",
    "safety_stock",
    "reorder_point",
    "simulated_on_hand",
    "simulated_on_order",
    "simulated_backorders",
    "inventory_position",
    "target_stock",
    "suggested_order_qty",
    "reorder_triggered",
    "economic_order_quantity",
    "recommendation_status",
]
REQUIRED_METRICS_KEYS = [
    "split_summary",
    "overall_metrics",
    "macro_averaged_metrics",
    "model_parameters",
]


def resolve_artifact_path(relative_or_absolute_path: Union[str, Path]) -> Path:
    """Resolve an artifact path relative to project root if not already absolute."""
    p = Path(relative_or_absolute_path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


@st.cache_data(show_spinner=False)
def load_daily_demand(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate processed daily product demand series.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]], optional
        Path to processed daily demand parquet.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path. Defaults to 'data/processed/daily_product_demand.parquet'.

    Returns
    -------
    pd.DataFrame
        Validated daily demand table with parsed dates.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing, dates cannot be parsed, or duplicate keys exist.
    """
    target = file_path or data_path or DEFAULT_DEMAND_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"Daily demand artifact not found at: '{path}'. "
            "Please run: python scripts/prepare_data.py --config config/config.yaml"
        )

    df = pd.read_parquet(path)

    # 1. Column validation
    missing_cols = [c for c in REQUIRED_DEMAND_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Daily demand dataset at '{path}' is missing required columns: {missing_cols}")

    # 2. Date parsing
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        if df["date"].isna().any():
            raise ValueError(f"Daily demand dataset at '{path}' contains unparseable date values.")

    # 3. Numeric validation
    if not pd.api.types.is_numeric_dtype(df["demand"]):
        df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
        if df["demand"].isna().any():
            raise ValueError(f"Daily demand dataset at '{path}' contains non-numeric demand values.")

    # 4. Check nulls
    null_counts = df[REQUIRED_DEMAND_COLS].isna().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Daily demand dataset contains unexpected null values: {bad_cols}")

    # 5. Duplicate key check
    dups = int(df.duplicated(subset=["product_id", "date"]).sum())
    if dups > 0:
        raise ValueError(f"Detected {dups} duplicate (product_id, date) records in daily demand dataset.")

    return df


@st.cache_data(show_spinner=False)
def load_daily_demand_summary(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load pre-aggregated daily demand summary across all catalog products.

    Avoids transferring 1M product-level rows to charts when rendering portfolio-level trends.

    Returns
    -------
    pd.DataFrame
        Aggregated table with columns ['date', 'total_demand', 'active_products_count', 'catalog_products_count'].
    """
    df = load_daily_demand(file_path=file_path, data_path=data_path)
    
    # Vectorized aggregation for maximum performance
    is_active = (df["demand"] > 0).astype(int)
    temp_df = pd.DataFrame({
        "date": df["date"],
        "demand": df["demand"],
        "is_active": is_active,
        "product_id": df["product_id"],
    })
    
    summary = (
        temp_df.groupby("date", as_index=False)
        .agg(
            total_demand=("demand", "sum"),
            active_products_count=("is_active", "sum"),
            catalog_products_count=("product_id", "count"),
        )
        .sort_values("date")
    )
    return summary


@st.cache_data(show_spinner=False)
def get_daily_demand_portfolio_stats(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Return high-level summary statistics of the daily demand dataset."""
    df = load_daily_demand(file_path=file_path, data_path=data_path)
    return {
        "row_count": len(df),
        "sku_count": int(df["product_id"].nunique()),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "total_days": int((df["date"].max() - df["date"].min()).days) + 1,
        "total_demand": float(df["demand"].sum()),
    }


@st.cache_data(show_spinner=False)
def load_top_products_demand(
    n: int = 10,
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Calculate and return top N products ranked by total recorded demand units.

    Rankings are derived strictly from the full processed daily demand artifact.

    Parameters
    ----------
    n : int, default 10
        Number of top products to return.
    file_path : Optional[Union[str, Path]], optional
        Path to processed daily demand parquet.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['product_id', 'total_demand'] sorted descending.
    """
    df = load_daily_demand(file_path=file_path, data_path=data_path)
    top_df = (
        df.groupby("product_id", as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "total_demand"})
        .sort_values("total_demand", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return top_df


@st.cache_data(show_spinner=False)
def get_available_product_ids(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Return sorted unique list of all valid product identifiers in the demand dataset."""
    source_df = df if df is not None else load_daily_demand(file_path=file_path, data_path=data_path)
    if source_df.empty or "product_id" not in source_df.columns:
        return []
    valid_ids = (
        source_df["product_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    valid_ids = valid_ids[valid_ids != ""]
    return sorted(valid_ids.unique().tolist())


@st.cache_data(show_spinner=False)
def load_product_daily_demand(
    product_id: str,
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load daily demand time series for a single product, preserving zero-demand days.

    Parameters
    ----------
    product_id : str
        Target product SKU.
    file_path : Optional[Union[str, Path]], optional
        Path to processed daily demand parquet.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path.
    df : Optional[pd.DataFrame], optional
        Optional preloaded daily demand DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'product_id', 'demand'] sorted chronologically.
    """
    source_df = df if df is not None else load_daily_demand(file_path=file_path, data_path=data_path)
    if source_df.empty or "product_id" not in source_df.columns or "date" not in source_df.columns:
        return pd.DataFrame(columns=REQUIRED_DEMAND_COLS)
    sub_df = source_df[source_df["product_id"].fillna("").astype(str).str.strip() == str(product_id).strip()]
    return sub_df.sort_values("date").reset_index(drop=True)


def compute_product_period_statistics(
    df_product: pd.DataFrame,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Dict[str, Any]:
    """Compute summary statistics for a product over a specified date range.

    Preserves zero-demand days and calculates positive vs zero day counts.

    Parameters
    ----------
    df_product : pd.DataFrame
        Daily demand series for a single product.
    start_date : Optional[Union[str, pd.Timestamp]], optional
        Start of selection window (inclusive).
    end_date : Optional[Union[str, pd.Timestamp]], optional
        End of selection window (inclusive).

    Returns
    -------
    Dict[str, Any]
        Dictionary with total_demand, mean_demand, total_days, positive_demand_days,
        zero_demand_days, and filtered_df.
    """
    if df_product.empty:
        return {
            "total_demand": 0.0,
            "mean_demand": 0.0,
            "total_days": 0,
            "positive_demand_days": 0,
            "zero_demand_days": 0,
            "filtered_df": df_product.copy(),
        }

    sub = df_product.copy()
    if start_date is not None:
        sub = sub[sub["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        sub = sub[sub["date"] <= pd.to_datetime(end_date)]

    total_days = len(sub)
    total_demand = float(sub["demand"].sum()) if total_days > 0 else 0.0
    mean_demand = float(sub["demand"].mean()) if total_days > 0 else 0.0
    positive_days = int((sub["demand"] > 0).sum()) if total_days > 0 else 0
    zero_days = int((sub["demand"] == 0).sum()) if total_days > 0 else 0

    return {
        "total_demand": total_demand,
        "mean_demand": mean_demand,
        "total_days": total_days,
        "positive_demand_days": positive_days,
        "zero_demand_days": zero_days,
        "filtered_df": sub.reset_index(drop=True),
    }


@st.cache_data(show_spinner=False)
def load_forecast_comparison(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate out-of-sample forecast comparison table.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]], optional
        Path to forecast comparison parquet. Defaults to 'outputs/forecasts/forecast_comparison.parquet'.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path.

    Returns
    -------
    pd.DataFrame
        Validated forecast comparisons table with parsed dates.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If required columns are missing, dates unparseable, or duplicate keys exist.
    """
    target = file_path or data_path or DEFAULT_FORECASTS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast comparison artifact not found at: '{path}'. "
            "Please run: python scripts/train_model.py --config config/config.yaml"
        )

    df = pd.read_parquet(path)

    missing_cols = [c for c in REQUIRED_FORECAST_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Forecast comparison dataset at '{path}' is missing required columns: {missing_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        if df["date"].isna().any():
            raise ValueError(f"Forecast comparison dataset at '{path}' contains unparseable date values.")

    # Check numeric types
    for col in ["actual_demand", "random_forest_pred", "seasonal_naive_pred"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                raise ValueError(f"Forecast comparison column '{col}' contains non-numeric values.")

    # Check nulls
    null_counts = df[REQUIRED_FORECAST_COLS].isna().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Forecast comparison dataset contains unexpected nulls in aligned evaluation rows: {bad_cols}")

    # Check duplicate keys
    dups = int(df.duplicated(subset=["product_id", "date"]).sum())
    if dups > 0:
        raise ValueError(f"Detected {dups} duplicate (product_id, date) rows in forecast comparison dataset.")

    return df


@st.cache_data(show_spinner=False)
def get_evaluated_product_ids(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Return sorted unique list of all product SKUs present in forecast evaluation artifact."""
    source_df = df if df is not None else load_forecast_comparison(file_path=file_path, data_path=data_path)
    if source_df.empty or "product_id" not in source_df.columns:
        return []
    valid_ids = (
        source_df["product_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    valid_ids = valid_ids[valid_ids != ""]
    return sorted(valid_ids.unique().tolist())


@st.cache_data(show_spinner=False)
def load_product_forecast_comparison(
    product_id: str,
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load out-of-sample forecast comparison series for a single product.

    Preserves chronological ordering and all available evaluation dates without
    substituting or fabricating data.

    Parameters
    ----------
    product_id : str
        Target product SKU.
    file_path : Optional[Union[str, Path]], optional
        Path to forecast comparison parquet.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path.
    df : Optional[pd.DataFrame], optional
        Optional preloaded forecast comparisons DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered, chronologically sorted DataFrame with columns:
        ['date', 'product_id', 'actual_demand', 'random_forest_pred', 'seasonal_naive_pred'].
    """
    source_df = df if df is not None else load_forecast_comparison(file_path=file_path, data_path=data_path)
    if source_df.empty or "product_id" not in source_df.columns or "date" not in source_df.columns:
        return pd.DataFrame(columns=REQUIRED_FORECAST_COLS)
    sub = source_df[source_df["product_id"].fillna("").astype(str).str.strip() == str(product_id).strip()]
    return sub.sort_values("date").reset_index(drop=True)


def compute_single_product_forecast_metrics(
    df_product: pd.DataFrame,
) -> Dict[str, Any]:
    """Calculate product-level accuracy metrics from evaluation rows.

    Calculates MAE, RMSE, and WAPE for Random Forest and Seasonal Naive baseline.
    Identifies missing values without substitution and handles zero WAPE denominator safely.

    Parameters
    ----------
    df_product : pd.DataFrame
        Evaluation slice for a single product containing 'actual_demand',
        'random_forest_pred', and 'seasonal_naive_pred'.

    Returns
    -------
    Dict[str, Any]
        Dictionary with sample_size, missing_count, total_actual, random_forest, and seasonal_naive.
    """
    if df_product is None or df_product.empty:
        return {
            "sample_size": 0,
            "missing_count": 0,
            "total_actual": 0.0,
            "random_forest": {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "wape_defined": False},
            "seasonal_naive": {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "wape_defined": False},
        }

    # Check for missing values across required evaluation columns
    cols = ["actual_demand", "random_forest_pred", "seasonal_naive_pred"]
    missing_cols = [c for c in cols if c in df_product.columns]
    if len(missing_cols) < len(cols):
        # Missing essential columns
        return {
            "sample_size": 0,
            "missing_count": len(df_product),
            "total_actual": 0.0,
            "random_forest": {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "wape_defined": False},
            "seasonal_naive": {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "wape_defined": False},
        }

    missing_mask = df_product[cols].isna().any(axis=1)
    missing_count = int(missing_mask.sum())

    # Do not substitute missing values; evaluate valid rows
    valid_df = df_product[~missing_mask].copy()
    sample_size = len(valid_df)

    if sample_size == 0:
        return {
            "sample_size": 0,
            "missing_count": missing_count,
            "total_actual": 0.0,
            "random_forest": {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "wape_defined": False},
            "seasonal_naive": {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "wape_defined": False},
        }

    y_true = valid_df["actual_demand"].values
    y_rf = valid_df["random_forest_pred"].values
    y_sn = valid_df["seasonal_naive_pred"].values

    total_actual = float(np.sum(np.abs(y_true)))

    # Random forest metrics
    rf_abs_err = np.abs(y_true - y_rf)
    rf_mae = float(np.mean(rf_abs_err))
    rf_rmse = float(np.sqrt(np.mean((y_true - y_rf) ** 2)))
    if total_actual > 0:
        rf_wape = float((np.sum(rf_abs_err) / total_actual) * 100.0)
        rf_wape_defined = True
    else:
        rf_wape = float("nan")
        rf_wape_defined = False

    # Seasonal naive metrics
    sn_abs_err = np.abs(y_true - y_sn)
    sn_mae = float(np.mean(sn_abs_err))
    sn_rmse = float(np.sqrt(np.mean((y_true - y_sn) ** 2)))
    if total_actual > 0:
        sn_wape = float((np.sum(sn_abs_err) / total_actual) * 100.0)
        sn_wape_defined = True
    else:
        sn_wape = float("nan")
        sn_wape_defined = False

    return {
        "sample_size": sample_size,
        "missing_count": missing_count,
        "total_actual": total_actual,
        "random_forest": {
            "mae": round(rf_mae, 4),
            "rmse": round(rf_rmse, 4),
            "wape": round(rf_wape, 4) if rf_wape_defined else float("nan"),
            "wape_defined": rf_wape_defined,
        },
        "seasonal_naive": {
            "mae": round(sn_mae, 4),
            "rmse": round(sn_rmse, 4),
            "wape": round(sn_wape, 4) if sn_wape_defined else float("nan"),
            "wape_defined": sn_wape_defined,
        },
    }


@st.cache_data(show_spinner=False)
def load_forecast_metrics(
    file_path: Optional[Union[str, Path]] = None,
    metrics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load and validate forecast evaluation metrics JSON.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]], optional
        Path to forecast metrics JSON. Defaults to 'outputs/metrics/forecast_metrics.json'.
    metrics_path : Optional[Union[str, Path]], optional
        Alias for file_path.

    Returns
    -------
    Dict[str, Any]
        Parsed and validated evaluation metrics dictionary.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If required keys or sub-metrics are missing.
    """
    target = file_path or metrics_path or DEFAULT_METRICS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast metrics artifact not found at: '{path}'. "
            "Please run: python scripts/train_model.py --config config/config.yaml"
        )

    with open(path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    for k in REQUIRED_METRICS_KEYS:
        if k not in metrics:
            raise ValueError(f"Forecast metrics JSON at '{path}' is missing required top-level key: '{k}'")

    overall = metrics.get("overall_metrics", {})
    for model_key in ["seasonal_naive", "random_forest"]:
        if model_key not in overall:
            raise ValueError(f"Forecast metrics missing overall metrics for '{model_key}'.")
        for m in ["mae", "rmse", "wape"]:
            if m not in overall[model_key]:
                raise ValueError(f"Model '{model_key}' missing required metric '{m}'.")

    return metrics


@st.cache_data(show_spinner=False)
def load_inventory_recommendations(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate inventory replenishment recommendations table.

    Preserves legitimate nulls in unavailable recommendations (where horizon is insufficient).
    Verifies that eligible recommendations are complete, finite, and non-negative.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]], optional
        Path to inventory recommendations parquet. Defaults to 'outputs/inventory/inventory_recommendations.parquet'.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path.

    Returns
    -------
    pd.DataFrame
        Complete 31-column inventory recommendations table.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If required columns are missing, product IDs are duplicated, or eligible rows have invalid values.
    """
    target = file_path or data_path or DEFAULT_RECOMMENDATIONS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory recommendations artifact not found at: '{path}'. "
            "Please run: python scripts/generate_recommendations.py --config config/config.yaml"
        )

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    missing_cols = [c for c in REQUIRED_RECOMMENDATION_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Inventory recommendations dataset at '{path}' is missing required columns: {missing_cols}")

    # Verify 1 recommendation per product_id
    dups = int(df.duplicated(subset=["product_id"]).sum())
    if dups > 0:
        raise ValueError(f"Detected {dups} duplicate product_id entries in inventory recommendations.")

    # Validate legitimate nulls vs unexpected nulls
    eligible_mask = df["recommendation_status"].isin(["REORDER_NOW", "STOCK_ADEQUATE"])
    required_eligible_numeric = [
        "expected_lead_time_demand",
        "expected_protection_period_demand",
        "safety_stock",
        "reorder_point",
        "inventory_position",
        "target_stock",
        "suggested_order_qty",
    ]

    for col in required_eligible_numeric:
        null_in_eligible = int(df.loc[eligible_mask, col].isna().sum())
        if null_in_eligible > 0:
            raise ValueError(
                f"Column '{col}' contains {null_in_eligible} unexpected null values among eligible recommendations."
            )

    # Validate non-negativity for physical quantities in eligible items
    if (df.loc[eligible_mask, "suggested_order_qty"] < 0).any():
        raise ValueError("suggested_order_qty contains negative values, which is physically impossible.")

    return df


@st.cache_data(show_spinner=False)
def get_inventory_recommendation_product_ids(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Return sorted unique list of product IDs present in inventory recommendations."""
    source_df = df if df is not None else load_inventory_recommendations(file_path=file_path, data_path=data_path)
    if source_df.empty or "product_id" not in source_df.columns:
        return []
    valid_ids = (
        source_df["product_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    valid_ids = valid_ids[valid_ids != ""]
    return sorted(valid_ids.unique().tolist())


def compute_inventory_summary_kpis(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Derive aggregate KPI statistics from an inventory recommendations DataFrame.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Recommendations DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of derived KPIs.
    """
    if df is None or df.empty:
        return {
            "total_products": 0,
            "reorder_now_count": 0,
            "stock_adequate_count": 0,
            "unavailable_count": 0,
            "total_suggested_order_qty": 0,
            "has_duplicates": False,
            "duplicate_count": 0,
            "is_valid": False,
        }

    # Verify column existence
    if "recommendation_status" not in df.columns:
        return {
            "total_products": len(df),
            "reorder_now_count": None,
            "stock_adequate_count": None,
            "unavailable_count": None,
            "total_suggested_order_qty": None,
            "has_duplicates": False,
            "duplicate_count": 0,
            "is_valid": False,
        }

    # Check duplicates in product_id
    dup_count = int(df.duplicated(subset=["product_id"]).sum()) if "product_id" in df.columns else 0
    has_dups = dup_count > 0

    statuses = df["recommendation_status"].fillna("UNKNOWN")
    reorder_now = int((statuses == "REORDER_NOW").sum())
    stock_adequate = int((statuses == "STOCK_ADEQUATE").sum())
    unavailable = int((statuses.str.startswith("UNAVAILABLE")).sum())

    # Calculate total suggested order quantity across eligible recommendations
    # Unavailable items are explicitly excluded (even if they have numeric values or nulls)
    total_suggested_order_qty = 0
    if "suggested_order_qty" in df.columns:
        eligible_mask = statuses.isin(["REORDER_NOW", "STOCK_ADEQUATE"])
        valid_qty = df.loc[eligible_mask, "suggested_order_qty"].dropna()
        total_suggested_order_qty = int(valid_qty.sum())

    return {
        "total_products": len(df),
        "reorder_now_count": reorder_now,
        "stock_adequate_count": stock_adequate,
        "unavailable_count": unavailable,
        "total_suggested_order_qty": total_suggested_order_qty,
        "has_duplicates": has_dups,
        "duplicate_count": dup_count,
        "is_valid": True,
    }


def filter_inventory_recommendations(
    df: Optional[pd.DataFrame],
    status_filter: Optional[str] = None,
    search_query: Optional[str] = None,
) -> pd.DataFrame:
    """Filter inventory recommendations safely by status and product ID search string.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Recommendations DataFrame.
    status_filter : Optional[str], optional
        Exact status to filter by (e.g. 'REORDER_NOW', 'STOCK_ADEQUATE', or 'All').
    search_query : Optional[str], optional
        Substring to search inside product_id.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    res = df.copy()

    if status_filter and status_filter != "All" and "recommendation_status" in res.columns:
        res = res[res["recommendation_status"].fillna("").astype(str) == status_filter]

    if search_query and search_query.strip() and "product_id" in res.columns:
        query = search_query.strip().lower()
        res = res[res["product_id"].fillna("").astype(str).str.lower().str.contains(query, regex=False)]

    return res


def get_single_product_inventory_details(
    product_id: str,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve detailed inventory recommendation fields for a single SKU.

    Identifies duplicates and reports whether records are simulated demonstrations.
    """
    rec_df = df if df is not None else load_inventory_recommendations()
    if rec_df is None or rec_df.empty or "product_id" not in rec_df.columns:
        return None

    matches = rec_df[rec_df["product_id"].fillna("").astype(str) == str(product_id)]
    if matches.empty:
        return None

    duplicate_count = len(matches)
    record = matches.iloc[0].to_dict()
    record["_has_duplicates"] = duplicate_count > 1
    record["_duplicate_count"] = duplicate_count
    return record


@st.cache_data(show_spinner=False)
def load_replenishment_orders(
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate filtered active purchase orders table.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]], optional
        Path to replenishment orders CSV. Defaults to 'outputs/inventory/replenishment_orders.csv'.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path.

    Returns
    -------
    pd.DataFrame
        Filtered active replenishment purchase orders.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If required columns are missing, order quantities are zero, or nulls exist.
    """
    target = file_path or data_path or DEFAULT_ORDERS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"Replenishment orders artifact not found at: '{path}'. "
            "Please run: python scripts/generate_recommendations.py --config config/config.yaml"
        )

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Must contain product_id, reorder_point, inventory_position, suggested_order_qty
    core_cols = ["product_id", "reorder_point", "inventory_position", "suggested_order_qty"]
    missing = [c for c in core_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Replenishment orders file missing essential columns: {missing}")

    if not df.empty:
        # All orders must have suggested_order_qty > 0
        if (df["suggested_order_qty"] <= 0).any():
            raise ValueError("Replenishment orders file contains non-positive suggested order quantities.")

        # In inventory theory, replenishment is triggered when IP <= ROP
        if (df["inventory_position"] > df["reorder_point"]).any():
            raise ValueError("Found order records where inventory_position exceeds reorder_point.")

    return df


def get_replenishment_orders_summary(
    df_orders: Optional[pd.DataFrame] = None,
    file_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Calculate summary metrics for replenishment purchase orders.

    Parameters
    ----------
    df_orders : Optional[pd.DataFrame], optional
        Preloaded replenishment orders DataFrame.
    file_path : Optional[Union[str, Path]], optional
        Path to replenishment orders CSV.
    data_path : Optional[Union[str, Path]], optional
        Alias for file_path.

    Returns
    -------
    Dict[str, Any]
        Dictionary with order_count, total_order_units, and is_available.
    """
    if df_orders is None:
        try:
            df_orders = load_replenishment_orders(file_path=file_path, data_path=data_path)
        except (FileNotFoundError, ValueError):
            return {
                "order_count": 0,
                "total_order_units": 0,
                "is_available": False,
            }

    if df_orders.empty:
        return {
            "order_count": 0,
            "total_order_units": 0,
            "is_available": True,
        }

    order_count = len(df_orders)
    total_units = int(df_orders["suggested_order_qty"].sum()) if "suggested_order_qty" in df_orders.columns else 0

    return {
        "order_count": order_count,
        "total_order_units": total_units,
        "is_available": True,
    }


def verify_all_dashboard_artifacts() -> Dict[str, Dict[str, Any]]:
    """Inspect and return the health status of all required pipeline artifacts.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Status dictionary for each required artifact including existence, path, and size.
    """
    artifacts = {
        "Daily Demand": DEFAULT_DEMAND_PATH,
        "Forecast Comparisons": DEFAULT_FORECASTS_PATH,
        "Forecast Metrics": DEFAULT_METRICS_PATH,
        "Inventory Recommendations": DEFAULT_RECOMMENDATIONS_PATH,
        "Replenishment Orders": DEFAULT_ORDERS_PATH,
    }

    status_report = {}
    for name, rel_path in artifacts.items():
        abs_path = resolve_artifact_path(rel_path)
        exists = abs_path.exists()
        size_bytes = abs_path.stat().st_size if exists else 0
        status_report[name] = {
            "relative_path": rel_path,
            "absolute_path": str(abs_path),
            "exists": exists,
            "size_bytes": size_bytes,
        }

    return status_report


def check_product_data_coverage(
    product_id: Union[str, int, None],
    demand_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Check availability of historical demand, forecast evaluation, and inventory recommendation data for an SKU.

    Distinguishes product populations across:
    1. Historical demand artifact (e.g. 3,804 catalog SKUs).
    2. Out-of-sample forecast evaluation artifact (e.g. 2,813 eligible SKUs).
    3. Retrospective inventory recommendation artifact (e.g. 2,813 eligible SKUs).

    Parameters
    ----------
    product_id : Union[str, int, None]
        Product SKU to inspect.
    demand_df : Optional[pd.DataFrame], optional
        Preloaded daily demand DataFrame.
    forecast_df : Optional[pd.DataFrame], optional
        Preloaded forecast comparisons DataFrame.
    inventory_df : Optional[pd.DataFrame], optional
        Preloaded inventory recommendations DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary reporting coverage status for each section, row counts, and date ranges.
    """
    if product_id is None or pd.isna(product_id) or str(product_id).strip() == "" or str(product_id).strip().lower() == "nan":
        return {
            "product_id": "",
            "is_valid_id": False,
            "has_demand": False,
            "demand_count": 0,
            "demand_start_date": None,
            "demand_end_date": None,
            "has_forecast": False,
            "forecast_count": 0,
            "forecast_start_date": None,
            "forecast_end_date": None,
            "has_inventory": False,
            "inventory_status": None,
            "inventory_has_duplicates": False,
            "inventory_duplicate_count": 0,
            "is_fully_covered": False,
        }

    pid_str = str(product_id).strip()

    # 1. Historical demand coverage
    has_demand = False
    demand_count = 0
    demand_start = None
    demand_end = None
    try:
        df_d = demand_df if demand_df is not None else load_daily_demand()
        if df_d is not None and not df_d.empty and "product_id" in df_d.columns:
            m_d = df_d[df_d["product_id"].fillna("").astype(str).str.strip() == pid_str]
            demand_count = len(m_d)
            if demand_count > 0:
                has_demand = True
                if "date" in m_d.columns and pd.api.types.is_datetime64_any_dtype(m_d["date"]):
                    demand_start = str(m_d["date"].min().date())
                    demand_end = str(m_d["date"].max().date())
                elif "date" in m_d.columns:
                    d_parsed = pd.to_datetime(m_d["date"], errors="coerce")
                    if not d_parsed.isna().all():
                        demand_start = str(d_parsed.min().date())
                        demand_end = str(d_parsed.max().date())
    except Exception:
        has_demand = False

    # 2. Forecast evaluation coverage
    has_forecast = False
    forecast_count = 0
    forecast_start = None
    forecast_end = None
    try:
        df_f = forecast_df if forecast_df is not None else load_forecast_comparison()
        if df_f is not None and not df_f.empty and "product_id" in df_f.columns:
            m_f = df_f[df_f["product_id"].fillna("").astype(str).str.strip() == pid_str]
            forecast_count = len(m_f)
            if forecast_count > 0:
                has_forecast = True
                if "date" in m_f.columns and pd.api.types.is_datetime64_any_dtype(m_f["date"]):
                    forecast_start = str(m_f["date"].min().date())
                    forecast_end = str(m_f["date"].max().date())
                elif "date" in m_f.columns:
                    d_parsed = pd.to_datetime(m_f["date"], errors="coerce")
                    if not d_parsed.isna().all():
                        forecast_start = str(d_parsed.min().date())
                        forecast_end = str(d_parsed.max().date())
    except Exception:
        has_forecast = False

    # 3. Inventory recommendation coverage
    has_inventory = False
    inventory_status = None
    inventory_has_dups = False
    inventory_dup_count = 0
    try:
        df_i = inventory_df if inventory_df is not None else load_inventory_recommendations()
        if df_i is not None and not df_i.empty and "product_id" in df_i.columns:
            m_i = df_i[df_i["product_id"].fillna("").astype(str).str.strip() == pid_str]
            inventory_dup_count = len(m_i)
            if inventory_dup_count > 0:
                has_inventory = True
                inventory_has_dups = inventory_dup_count > 1
                if inventory_has_dups:
                    inventory_status = "DUPLICATE_RECORDS"
                elif "recommendation_status" in m_i.columns:
                    inventory_status = str(m_i["recommendation_status"].iloc[0])
    except Exception:
        has_inventory = False

    is_fully_covered = has_demand and has_forecast and has_inventory and not inventory_has_dups

    return {
        "product_id": pid_str,
        "is_valid_id": True,
        "has_demand": has_demand,
        "demand_count": demand_count,
        "demand_start_date": demand_start,
        "demand_end_date": demand_end,
        "has_forecast": has_forecast,
        "forecast_count": forecast_count,
        "forecast_start_date": forecast_start,
        "forecast_end_date": forecast_end,
        "has_inventory": has_inventory,
        "inventory_status": inventory_status,
        "inventory_has_duplicates": inventory_has_dups,
        "inventory_duplicate_count": inventory_dup_count,
        "is_fully_covered": is_fully_covered,
    }


def generate_sku_synthesis(
    coverage: Dict[str, Any],
    demand_stats: Optional[Dict[str, Any]] = None,
    forecast_metrics: Optional[Dict[str, Any]] = None,
    inventory_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate an objective, evidence-based synthesis connecting demand, forecast, and inventory.

    Adheres strictly to methodological guidelines:
    - Does not claim universal model superiority based on single-product results.
    - Does not invent causal assertions unsupported by data.
    - Accurately explains missing coverage where applicable.
    - Clearly labels simulated operational parameters.

    Parameters
    ----------
    coverage : Dict[str, Any]
        Coverage report from check_product_data_coverage.
    demand_stats : Optional[Dict[str, Any]], optional
        Historical demand statistics from compute_product_period_statistics.
    forecast_metrics : Optional[Dict[str, Any]], optional
        Forecast evaluation metrics from compute_single_product_forecast_metrics.
    inventory_details : Optional[Dict[str, Any]], optional
        Recommendation details from get_single_product_inventory_details.

    Returns
    -------
    Dict[str, Any]
        Dictionary with demand_commentary, forecast_commentary, inventory_commentary, and overall_synthesis.
    """
    pid = coverage.get("product_id", "Unknown")

    # 1. Demand commentary
    if coverage.get("has_demand") and demand_stats:
        total_demand = demand_stats.get("total_demand", 0.0)
        mean_demand = demand_stats.get("mean_demand", 0.0)
        total_days = demand_stats.get("total_days", 0)
        pos_days = demand_stats.get("positive_demand_days", 0)
        zero_days = demand_stats.get("zero_demand_days", 0)
        active_pct = (pos_days / max(1, total_days)) * 100
        zero_pct = (zero_days / max(1, total_days)) * 100
        demand_commentary = (
            f"Over the selected historical period ({total_days:,} days), SKU **{pid}** recorded "
            f"**{total_demand:,.0f} units** total demand ({mean_demand:.2f} units/day average). "
            f"Transactions occurred on {pos_days:,} days ({active_pct:.1f}%), with {zero_days:,} zero-demand days ({zero_pct:.1f}%). "
            "Note that zero recorded demand indicates calendar days without purchase transactions; "
            "it does not prove stock availability (unobserved stockouts)."
        )
    else:
        demand_commentary = f"Historical demand data is not available for SKU **{pid}**."

    # 2. Forecast commentary
    if coverage.get("has_forecast") and forecast_metrics:
        rf = forecast_metrics.get("random_forest", {})
        sn = forecast_metrics.get("seasonal_naive", {})
        sample_size = forecast_metrics.get("sample_size", 0)
        rf_mae = rf.get("mae")
        sn_mae = sn.get("mae")
        rf_wape = rf.get("wape")
        sn_wape = sn.get("wape")
        wape_defined = rf.get("wape_defined", False)

        if wape_defined:
            forecast_commentary = (
                f"During the 28-day retrospective evaluation window (Nov 12 – Dec 9, 2011; {sample_size} days evaluated), "
                f"Random Forest recorded an MAE of **{rf_mae:.2f} units** and WAPE of **{rf_wape:.1f}%**, compared to "
                f"Seasonal Naive baseline MAE of **{sn_mae:.2f} units** and WAPE of **{sn_wape:.1f}%**. "
                "These out-of-sample metrics represent product-level performance for this specific SKU and backtest window; "
                "they do not imply that one model is universally superior across the entire retail catalog."
            )
        else:
            forecast_commentary = (
                f"During the 28-day retrospective evaluation window ({sample_size} days evaluated), total actual demand was zero. "
                f"Random Forest MAE was **{rf_mae:.2f} units** and Seasonal Naive MAE was **{sn_mae:.2f} units**. "
                "WAPE is undefined due to zero denominator. "
                "These retrospective rolling one-day-ahead predictions reflect historical backtest behavior."
            )
    else:
        forecast_commentary = (
            f"Forecast evaluation records are not available for SKU **{pid}**. "
            "This product did not meet the feature eligibility threshold (insufficient historical data prior to the Nov 11, 2011 cutoff)."
        )

    # 3. Inventory commentary
    if coverage.get("has_inventory") and inventory_details:
        if inventory_details.get("_has_duplicates"):
            inventory_commentary = (
                f"Inventory recommendation data for SKU **{pid}** contains duplicate records "
                f"({inventory_details.get('_duplicate_count')} rows). In accordance with data integrity safeguards, "
                "recommendations are not arbitrarily selected from conflicting rows."
            )
        else:
            status = inventory_details.get("recommendation_status", "UNKNOWN")
            ip = inventory_details.get("inventory_position")
            rop = inventory_details.get("reorder_point")
            order_qty = inventory_details.get("suggested_order_qty")
            ss = inventory_details.get("safety_stock")
            ltd = inventory_details.get("expected_lead_time_demand")

            if status == "UNAVAILABLE_INSUFFICIENT_HORIZON":
                inventory_commentary = (
                    f"Under the simulated periodic review policy (L=7, R=1, P=8 days), recommendation status is **{status}**. "
                    "Replenishment parameters are undefined because available forecast horizon was shorter than the required protection period."
                )
            elif status == "REORDER_NOW":
                ip_val = f"{ip:.0f}" if ip is not None and not pd.isna(ip) else "N/A"
                rop_val = f"{rop:.1f}" if rop is not None and not pd.isna(rop) else "N/A"
                ss_val = f"{ss:.1f}" if ss is not None and not pd.isna(ss) else "N/A"
                ltd_val = f"{ltd:.1f}" if ltd is not None and not pd.isna(ltd) else "N/A"
                qty_val = f"{order_qty:.0f}" if order_qty is not None and not pd.isna(order_qty) else "N/A"
                inventory_commentary = (
                    f"Under the simulated periodic review policy (L=7, R=1, P=8 days, 95% service level), status is **REORDER_NOW**. "
                    f"Effective inventory position ({ip_val} units) is at or below the reorder point ({rop_val} units; SS={ss_val}, LTD={ltd_val}), "
                    f"triggering a simulated replenishment order of **{qty_val} units**. "
                    "Note: Inventory position, holding costs, and order triggers reflect simulated demonstration assumptions, not real warehouse stock."
                )
            elif status == "STOCK_ADEQUATE":
                ip_val = f"{ip:.0f}" if ip is not None and not pd.isna(ip) else "N/A"
                rop_val = f"{rop:.1f}" if rop is not None and not pd.isna(rop) else "N/A"
                ss_val = f"{ss:.1f}" if ss is not None and not pd.isna(ss) else "N/A"
                ltd_val = f"{ltd:.1f}" if ltd is not None and not pd.isna(ltd) else "N/A"
                inventory_commentary = (
                    f"Under the simulated periodic review policy (L=7, R=1, P=8 days, 95% service level), status is **STOCK_ADEQUATE**. "
                    f"Effective inventory position ({ip_val} units) exceeds the reorder point ({rop_val} units; SS={ss_val}, LTD={ltd_val}), "
                    "so no replenishment is triggered (suggested order: 0 units). "
                    "Note: Operational stock inputs reflect simulated demonstration parameters."
                )
            else:
                inventory_commentary = f"Inventory recommendation status: **{status}**."
    else:
        inventory_commentary = f"Inventory recommendation data is not available for SKU **{pid}**."

    # 4. Overall synthesis
    if coverage.get("is_fully_covered"):
        overall_synthesis = (
            f"**Comprehensive Synthesis for SKU {pid}**: The product exhibits empirical demand that informed "
            "both rolling one-day-ahead backtest predictions and periodic review inventory parameters. "
            "Historical demand patterns, backtest errors, and simulated inventory levels interact to determine "
            "safety stock buffers and reorder thresholds without assuming unobserved stockouts or cross-catalog superiority."
        )
    else:
        missing_parts = []
        if not coverage.get("has_demand"):
            missing_parts.append("historical demand")
        if not coverage.get("has_forecast"):
            missing_parts.append("forecast evaluation")
        if not coverage.get("has_inventory"):
            missing_parts.append("inventory recommendations")
        elif coverage.get("inventory_has_duplicates"):
            missing_parts.append("valid non-duplicate inventory recommendations")
        missing_str = ", ".join(missing_parts)
        overall_synthesis = (
            f"**Partial Coverage Notice for SKU {pid}**: Data coverage is incomplete ({missing_str} unavailable). "
            "The system presents available empirical observations without fabricating or imputing missing records."
        )

    return {
        "demand_commentary": demand_commentary,
        "forecast_commentary": forecast_commentary,
        "inventory_commentary": inventory_commentary,
        "overall_synthesis": overall_synthesis,
    }


# =============================================================================
# DataCo Smart Supply Chain Loaders & Utilities
# =============================================================================

@st.cache_data(show_spinner=False)
def load_dataco_daily_demand(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate processed DataCo daily product demand series."""
    target = file_path or DEFAULT_DATACO_DEMAND_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"DataCo daily demand artifact not found at: '{path}'. "
            "Please run: python src/retail_forecasting/data/preprocess_dataco.py"
        )
    df = pd.read_parquet(path)
    required = ["date", "product_id", "demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataCo daily demand is missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0.0)
    df["product_id"] = df["product_id"].astype(str)
    return df.sort_values(by=["product_id", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_dataco_catalog(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate processed DataCo product catalog."""
    target = file_path or DEFAULT_DATACO_CATALOG_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(f"DataCo catalog artifact not found at: '{path}'.")
    df = pd.read_parquet(path)
    df["product_id"] = df["product_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_dataco_daily_demand_summary(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load aggregate daily demand summary across all catalog products."""
    df = load_dataco_daily_demand(file_path=file_path)
    is_active = (df["demand"] > 0).astype(int)
    temp = pd.DataFrame({
        "date": df["date"],
        "demand": df["demand"],
        "is_active": is_active,
        "product_id": df["product_id"],
    })
    summary = (
        temp.groupby("date", as_index=False)
        .agg(
            total_demand=("demand", "sum"),
            active_products_count=("is_active", "sum"),
            catalog_products_count=("product_id", "count"),
        )
        .sort_values("date")
    )
    return summary


@st.cache_data(show_spinner=False)
def get_dataco_portfolio_stats(
    demand_path: Optional[Union[str, Path]] = None,
    catalog_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Return dictionary of verified portfolio statistics for DataCo."""
    demand_df = load_dataco_daily_demand(file_path=demand_path)
    catalog_df = load_dataco_catalog(file_path=catalog_path)
    min_date = demand_df["date"].min().strftime("%Y-%m-%d")
    max_date = demand_df["date"].max().strftime("%Y-%m-%d")
    total_days = int((demand_df["date"].max() - demand_df["date"].min()).days) + 1
    total_demand = float(demand_df["demand"].sum())
    sku_count = int(catalog_df["product_id"].nunique())
    row_count = len(demand_df)

    return {
        "start_date": min_date,
        "end_date": max_date,
        "total_days": total_days,
        "sku_count": sku_count,
        "row_count": row_count,
        "total_demand": total_demand,
        "raw_orders_count": 180519,
        "core_cohort_count": 9,
    }


@st.cache_data(show_spinner=False)
def load_dataco_top_products_demand(
    top_n: int = 10,
    demand_path: Optional[Union[str, Path]] = None,
    catalog_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Return top N products ranked by total gross placed demand."""
    demand_df = load_dataco_daily_demand(file_path=demand_path)
    catalog_df = load_dataco_catalog(file_path=catalog_path)

    totals = (
        demand_df.groupby("product_id", as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "total_demand"})
        .sort_values("total_demand", ascending=False)
        .head(top_n)
    )
    merged = pd.merge(
        totals,
        catalog_df[["product_id", "product_name", "department_name", "category_name", "product_price"]],
        on="product_id",
        how="left"
    )
    return merged


@st.cache_data(show_spinner=False)
def load_dataco_department_summary(
    demand_path: Optional[Union[str, Path]] = None,
    catalog_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Return demand summary aggregated by department."""
    demand_df = load_dataco_daily_demand(file_path=demand_path)
    catalog_df = load_dataco_catalog(file_path=catalog_path)

    merged = pd.merge(demand_df, catalog_df, on="product_id", how="left")
    dept_summary = (
        merged.groupby("department_name", as_index=False)
        .agg(
            total_demand=("demand", "sum"),
            product_count=("product_id", "nunique"),
        )
        .sort_values("total_demand", ascending=False)
    )
    return dept_summary


@st.cache_data(show_spinner=False)
def load_dataco_forecast_comparison(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load out-of-sample forecast comparison artifact for DataCo."""
    target = file_path or DEFAULT_DATACO_FORECASTS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(f"DataCo forecast comparison not found at: '{path}'.")
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_dataco_forecast_metrics(
    file_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load forecast evaluation metrics JSON for DataCo."""
    target = file_path or DEFAULT_DATACO_METRICS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(f"DataCo forecast metrics not found at: '{path}'.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_dataco_fixed_origin_forecasts(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load fixed-origin multi-step recursive forecasts for DataCo."""
    target = file_path or DEFAULT_DATACO_FIXED_FORECASTS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(f"DataCo fixed-origin forecasts not found at: '{path}'.")
    df = pd.read_parquet(path)
    df["product_id"] = df["product_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_dataco_inventory_recommendations(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load DataCo inventory recommendations table."""
    target = file_path or DEFAULT_DATACO_RECOMMENDATIONS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(f"DataCo inventory recommendations not found at: '{path}'.")
    df = pd.read_parquet(path)
    df["product_id"] = df["product_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_dataco_replenishment_orders(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load DataCo replenishment purchase orders CSV."""
    target = file_path or DEFAULT_DATACO_ORDERS_PATH
    path = resolve_artifact_path(target)
    if not path.exists():
        raise FileNotFoundError(f"DataCo replenishment orders not found at: '{path}'.")
    df = pd.read_csv(path)
    df["product_id"] = df["product_id"].astype(str)
    return df

