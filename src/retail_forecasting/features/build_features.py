"""Feature engineering pipeline for retail demand forecasting.

Constructs leak-free time-series features including:
- Calendar attributes derived strictly from transaction dates.
- Autoregressive lag features (lag_1, lag_7, lag_14, lag_28) calculated per product independently.
- Rolling window statistics (mean, std) calculated strictly on target values shifted by 1 row.
- Model eligibility indicators identifying rows with complete historical feature coverage.
- Product history diagnostics auditing calendar history, sales velocity, and sparsity.

Target & Feature Contract:
- Target column: 'demand' at date t (the value to be forecasted).
- Feature columns: Calendar indicators at date t and historical statistics computed strictly
  from observations prior to date t (t-1, t-7, etc.).
- Missing historical values are preserved as NaN and never silently replaced with 0.0.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


DEFAULT_LAGS = [1, 7, 14, 28]
DEFAULT_WINDOWS = [7, 14, 28]


def create_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date"
) -> pd.DataFrame:
    """Extract calendar-derived temporal features from date column.

    Features derived:
    - day_of_week : int (0 = Monday, 6 = Sunday)
    - month       : int (1 to 12)
    - day_of_month: int (1 to 31)
    - is_weekend  : int (1 if Saturday or Sunday, else 0)

    Does not use the target demand column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a date column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe augmented with calendar columns.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If date_col is missing from df.
    ValueError
        If date_col cannot be parsed as datetime.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in dataframe columns: {list(df.columns)}")

    out = df.copy()

    # Parse date if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        converted_dates = pd.to_datetime(out[date_col], errors="coerce")
        if converted_dates.isna().any() and not out[date_col].isna().all():
            unparseable_cnt = int(converted_dates.isna().sum() - out[date_col].isna().sum())
            if unparseable_cnt > 0:
                raise ValueError(
                    f"Found {unparseable_cnt} unparseable date values in column '{date_col}'."
                )
        out[date_col] = converted_dates

    dt_series = out[date_col].dt
    out["day_of_week"] = dt_series.dayofweek.astype("Int64")
    out["month"] = dt_series.month.astype("Int64")
    out["day_of_month"] = dt_series.day.astype("Int64")
    out["is_weekend"] = dt_series.dayofweek.isin([5, 6]).astype("Int64")

    return out


def create_lag_features(
    df: pd.DataFrame,
    group_cols: Optional[Union[str, List[str]]] = None,
    target_col: str = "demand",
    lags: Optional[List[int]] = None,
    date_col: str = "date"
) -> pd.DataFrame:
    """Create autoregressive lag features for each product entity independently.

    For prediction at date t:
    - lag_k equals target demand at date t-k.
    - Missing historical observations are retained as NaN (never silently zero-filled).

    Parameters
    ----------
    df : pd.DataFrame
        Daily product demand dataframe.
    group_cols : Optional[Union[str, List[str]]]
        Entity identifier column(s) (e.g. 'product_id'). Defaults to 'product_id'.
    target_col : str
        Demand column name to lag. Defaults to 'demand'.
    lags : Optional[List[int]]
        List of integer lag offsets (e.g. [1, 7, 14, 28]). Defaults to [1, 7, 14, 28].
    date_col : str
        Date column name used for chronological sorting.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with added lag feature columns.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If required columns are missing.
    ValueError
        If duplicate (group_cols, date_col) rows are detected.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if lags is None:
        lags = DEFAULT_LAGS

    if group_cols is None:
        group_cols = ["product_id"] if "product_id" in df.columns else []
    elif isinstance(group_cols, str):
        group_cols = [group_cols]

    required = [date_col, target_col] + group_cols
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe.")

    if df.empty:
        out = df.copy()
        for k in lags:
            out[f"lag_{k}"] = pd.Series(dtype="float64")
        return out

    # Check for duplicate product-date entries
    if group_cols:
        check_cols = group_cols + [date_col]
        duplicates = df.duplicated(subset=check_cols).sum()
        if duplicates > 0:
            raise ValueError(
                f"Detected {duplicates} duplicate entries for {check_cols}. "
                "Dataframe must have at most one observation per entity and date."
            )

    out = df.copy()

    # Sort strictly chronologically within group
    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    out = out.sort_values(by=sort_cols).reset_index(drop=True)

    if group_cols:
        for k in lags:
            col_name = f"lag_{k}"
            out[col_name] = out.groupby(group_cols)[target_col].shift(k)
    else:
        for k in lags:
            col_name = f"lag_{k}"
            out[col_name] = out[target_col].shift(k)

    return out


def create_rolling_features(
    df: pd.DataFrame,
    group_cols: Optional[Union[str, List[str]]] = None,
    target_col: str = "demand",
    windows: Optional[List[int]] = None,
    date_col: str = "date",
    min_periods: Optional[Dict[int, int]] = None
) -> pd.DataFrame:
    """Create rolling window statistics (mean, std) without lookahead leakage.

    Strict Leakage Prevention:
    - Target demand at date t is EXCLUDED from rolling calculations at date t.
    - Implemented by shifting the target series by 1 row (.shift(1)) before
      computing .rolling(w).
    - Therefore, rolling_mean_w at date t represents the mean of demand from
      t-w up to t-1.

    Parameters
    ----------
    df : pd.DataFrame
        Daily product demand dataframe.
    group_cols : Optional[Union[str, List[str]]]
        Entity identifier column(s). Defaults to 'product_id'.
    target_col : str
        Demand column name.
    windows : Optional[List[int]]
        Rolling window sizes in days (e.g. [7, 14, 28]). Defaults to [7, 14, 28].
    date_col : str
        Date column name used for sorting.
    min_periods : Optional[Dict[int, int]]
        Mapping of window size to minimum required historical observations.
        If None, defaults to the window size w (requiring full window history).

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with rolling mean and std columns added.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If required columns are missing.
    ValueError
        If duplicate product-date rows exist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if windows is None:
        windows = DEFAULT_WINDOWS

    if group_cols is None:
        group_cols = ["product_id"] if "product_id" in df.columns else []
    elif isinstance(group_cols, str):
        group_cols = [group_cols]

    required = [date_col, target_col] + group_cols
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe.")

    if df.empty:
        out = df.copy()
        for w in windows:
            out[f"rolling_mean_{w}"] = pd.Series(dtype="float64")
            out[f"rolling_std_{w}"] = pd.Series(dtype="float64")
        return out

    # Check for duplicate entity-date entries
    if group_cols:
        check_cols = group_cols + [date_col]
        duplicates = df.duplicated(subset=check_cols).sum()
        if duplicates > 0:
            raise ValueError(
                f"Detected {duplicates} duplicate entries for {check_cols}. "
                "Dataframe must have at most one observation per entity and date."
            )

    out = df.copy()

    # Sort strictly chronologically
    sort_cols = group_cols + [date_col] if group_cols else [date_col]
    out = out.sort_values(by=sort_cols).reset_index(drop=True)

    for w in windows:
        min_p = min_periods.get(w, w) if min_periods else w
        mean_col = f"rolling_mean_{w}"
        std_col = f"rolling_std_{w}"

        if group_cols:
            # Shift by 1 row first, then compute rolling window per group
            shifted_series = out.groupby(group_cols)[target_col].shift(1)
            # Group again on the shifted series to calculate rolling window
            rolling_obj = (
                shifted_series.groupby(out[group_cols].apply(tuple, axis=1) if len(group_cols) > 1 else out[group_cols[0]])
                .rolling(window=w, min_periods=min_p)
            )
            out[mean_col] = rolling_obj.mean().reset_index(level=0, drop=True)
            out[std_col] = rolling_obj.std().reset_index(level=0, drop=True)
        else:
            shifted = out[target_col].shift(1)
            rolling_obj = shifted.rolling(window=w, min_periods=min_p)
            out[mean_col] = rolling_obj.mean()
            out[std_col] = rolling_obj.std()

    return out


def build_feature_pipeline(
    df: pd.DataFrame,
    target_col: str = "demand",
    date_col: str = "date",
    group_col: str = "product_id",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
    min_periods: Optional[Dict[int, int]] = None
) -> pd.DataFrame:
    """Execute end-to-end feature engineering pipeline.

    Orchestrates:
    1. Validation of required columns and duplicate checking.
    2. Chronological sorting.
    3. Calendar feature generation.
    4. Independent per-product autoregressive lag calculation.
    5. Independent per-product shifted rolling statistics.
    6. Identification of model eligibility ('is_model_eligible').

    Parameters
    ----------
    df : pd.DataFrame
        Daily product demand DataFrame.
    target_col : str
        Target demand column name.
    date_col : str
        Date column name.
    group_col : str
        Product SKU identifier column name.
    lags : Optional[List[int]]
        Lag periods (default [1, 7, 14, 28]).
    windows : Optional[List[int]]
        Rolling windows (default [7, 14, 28]).
    min_periods : Optional[Dict[int, int]]
        Minimum observations for rolling windows.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame with all features and 'is_model_eligible' flag.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if lags is None:
        lags = DEFAULT_LAGS
    if windows is None:
        windows = DEFAULT_WINDOWS

    # 1. Calendar features
    with_cal = create_calendar_features(df, date_col=date_col)

    # 2. Lag features
    with_lags = create_lag_features(
        with_cal,
        group_cols=[group_col],
        target_col=target_col,
        lags=lags,
        date_col=date_col,
    )

    # 3. Rolling features
    with_rolling = create_rolling_features(
        with_lags,
        group_cols=[group_col],
        target_col=target_col,
        windows=windows,
        date_col=date_col,
        min_periods=min_periods,
    )

    # 4. Model eligibility determination
    # A row is eligible for training/inference if all required primary historical features are non-NaN
    lag_cols = [f"lag_{k}" for k in lags]
    max_lag = max(lags) if lags else 0
    max_window = max(windows) if windows else 0
    primary_req_cols = lag_cols + [f"rolling_mean_{max_window}"]

    with_rolling["is_model_eligible"] = with_rolling[primary_req_cols].notna().all(axis=1)

    return with_rolling


def analyze_product_histories(
    df: pd.DataFrame,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand",
    min_history_days: int = 28
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Produce comprehensive diagnostics on product observation history and sales sparsity.

    Distinguishes clearly between:
    - Calendar history length: Days between first observed sale and last observed sale.
    - Positive-demand days: Actual days where demand > 0 (sales velocity).
    - Zero-demand days: Days with zero sales (unobserved or non-selling).

    Parameters
    ----------
    df : pd.DataFrame
        Daily product demand DataFrame.
    date_col : str
        Date column name.
    product_col : str
        Product identifier column name.
    demand_col : str
        Demand column name.
    min_history_days : int
        Threshold of calendar observations required to compute max lag (e.g. 28 days).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (product_diagnostics_df, summary_report)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        empty_diag = pd.DataFrame(columns=[
            product_col, "first_observed_date", "last_observed_date",
            "calendar_observations", "positive_demand_days", "zero_demand_days",
            "zero_demand_pct", "has_enough_history_for_lag28", "eligible_modeling_rows"
        ])
        return empty_diag, {"total_products": 0, "history_tiers": {}}

    grouped = df.groupby(product_col)

    records = []
    for pid, group in grouped:
        cal_obs = len(group)
        pos_days = int((group[demand_col] > 0).sum())
        zero_days = cal_obs - pos_days
        zero_pct = round(zero_days / cal_obs * 100.0, 2) if cal_obs > 0 else 0.0
        first_dt = group[date_col].min()
        last_dt = group[date_col].max()
        has_lag28 = bool(cal_obs > 28)
        eligible_rows = max(0, cal_obs - min_history_days)

        records.append({
            product_col: pid,
            "first_observed_date": first_dt,
            "last_observed_date": last_dt,
            "calendar_observations": cal_obs,
            "positive_demand_days": pos_days,
            "zero_demand_days": zero_days,
            "zero_demand_pct": zero_pct,
            "has_enough_history_for_lag28": has_lag28,
            "eligible_modeling_rows": eligible_rows,
        })

    diag_df = pd.DataFrame(records)

    # Group into analytical history length tiers
    total_products = len(diag_df)
    tier_less_7 = int((diag_df["calendar_observations"] < 7).sum())
    tier_7_to_28 = int(((diag_df["calendar_observations"] >= 7) & (diag_df["calendar_observations"] <= 28)).sum())
    tier_29_to_90 = int(((diag_df["calendar_observations"] >= 29) & (diag_df["calendar_observations"] <= 90)).sum())
    tier_91_to_180 = int(((diag_df["calendar_observations"] >= 91) & (diag_df["calendar_observations"] <= 180)).sum())
    tier_over_180 = int((diag_df["calendar_observations"] > 180).sum())

    summary_report = {
        "total_products": total_products,
        "history_tiers": {
            "< 7 days (ultra-sparse)": tier_less_7,
            "7 to 28 days (short, insufficient for lag_28)": tier_7_to_28,
            "29 to 90 days (moderate history)": tier_29_to_90,
            "91 to 180 days (long history)": tier_91_to_180,
            "> 180 days (mature catalog)": tier_over_180,
        },
        "products_with_lag28_capacity": int(diag_df["has_enough_history_for_lag28"].sum()),
        "products_without_lag28_capacity": int((~diag_df["has_enough_history_for_lag28"]).sum()),
        "average_calendar_observations": round(float(diag_df["calendar_observations"].mean()), 1),
        "average_positive_demand_days": round(float(diag_df["positive_demand_days"].mean()), 1),
        "average_zero_demand_pct": round(float(diag_df["zero_demand_pct"].mean()), 2),
    }

    return diag_df, summary_report
