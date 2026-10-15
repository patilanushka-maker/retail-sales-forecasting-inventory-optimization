"""Benchmarking engine and model pipelines for DataCo daily retail demand.

Provides leak-free, rolling one-day-ahead time series forecasting for the Core 9
DataCo products comparing:
1. Baseline A: Seasonal Naive (7-calendar-day seasonal lag: y_hat(t) = y(t - 7d))
2. Baseline B: 7-Day Rolling Mean (mean of previous 7 calendar days strictly shifted by 1 day)
3. Benchmark C: Random Forest Regressor (tabular autoregressive features using lags and shifted rolling stats)

Temporal Evaluation Protocols:
- Pre-cutoffs / Training: 2015-01-01 to 2017-06-30 (912 calendar days)
- Validation Window: 2017-07-01 to 2017-08-31 (62 calendar days)
- Test Window: 2017-09-01 to 2017-10-02 (32 calendar days, through observed product lifespans)
- Alternative Horizon Audit: Documented reporting for Nov 2017 - Jan 2018 where the Core 9
  are observed inactive (discontinued/zero sales).

Zero-Filling & Leakage Protections:
- Reindexing to contiguous calendar days is applied strictly within each product's active lifespan.
- Rolling statistics strictly shift by 1 day to exclude date t actual demand.
- Models are evaluated on identical product-date forecast records.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from retail_forecasting.forecasting.evaluate import compute_metrics, compute_per_product_metrics


# Approved Core 9 Product IDs (verified continuous 2015-01-01 to 2017-10-02)
DATACO_CORE_9_PRODUCT_IDS: List[str] = [
    "365",
    "502",
    "1014",
    "191",
    "627",
    "403",
    "1004",
    "1073",
    "957",
]

# Standard evaluation date boundaries for the Core 9 cohort
SPLIT_TRAIN_START = "2015-01-01"
SPLIT_TRAIN_END = "2017-06-30"
SPLIT_VAL_START = "2017-07-01"
SPLIT_VAL_END = "2017-08-31"
SPLIT_TEST_START = "2017-09-01"
SPLIT_TEST_END = "2017-10-02"


def select_dataco_top_products(
    demand_df: pd.DataFrame,
    pretest_cutoff: str = "2017-07-01",
    top_n: int = 9,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand"
) -> List[str]:
    """Select the top N highest-volume products strictly using training-period demand.

    Prevents lookahead leakage by never inspecting validation or test-period demand.
    By default, pretest_cutoff='2017-07-01' restricts data strictly to observations
    through 2017-06-30.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Daily product demand DataFrame.
    pretest_cutoff : str
        Upper bound date (exclusive) defining the historical pre-test period.
    top_n : int
        Number of products to select (default 9).
    date_col : str
        Date column header.
    product_col : str
        Product ID column header.
    demand_col : str
        Demand column header.

    Returns
    -------
    List[str]
        Sorted list of selected product_id strings in descending order of pre-test demand.
    """
    if not isinstance(demand_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(demand_df).__name__}")

    pretest_df = demand_df[demand_df[date_col] < pretest_cutoff]
    if pretest_df.empty:
        raise ValueError(f"No records found strictly prior to pretest cutoff '{pretest_cutoff}'.")

    vol = (
        pretest_df.groupby(product_col)[demand_col]
        .sum()
        .sort_values(ascending=False)
    )
    return list(vol.head(top_n).index.astype(str))


def build_contiguous_daily_series(
    demand_df: pd.DataFrame,
    product_ids: Optional[List[str]] = None,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand"
) -> pd.DataFrame:
    """Construct contiguous daily calendar series per product with lifespan zero-filling.

    Fills missing dates with 0.0 strictly between each product's first and last observed
    order dates. Does not create artificial zeros before introduction or after discontinuation.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Input daily demand DataFrame.
    product_ids : Optional[List[str]]
        Subset of product IDs to process. If None, processes all products in demand_df.
    date_col : str
        Date column header.
    product_col : str
        Product ID column header.
    demand_col : str
        Demand column header.

    Returns
    -------
    pd.DataFrame
        Continuous daily DataFrame with columns [date, product_id, demand], sorted by
        [product_id, date].
    """
    if not isinstance(demand_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(demand_df).__name__}")

    df_sub = demand_df.copy()
    df_sub[date_col] = pd.to_datetime(df_sub[date_col]).dt.floor("D")
    df_sub[product_col] = df_sub[product_col].astype(str)

    if product_ids is not None:
        target_pids = set(str(p) for p in product_ids)
        df_sub = df_sub[df_sub[product_col].isin(target_pids)]

    reindexed_frames = []
    for pid, group in df_sub.groupby(product_col):
        min_date = group[date_col].min()
        max_date = group[date_col].max()

        full_calendar = pd.date_range(start=min_date, end=max_date, freq="D", name=date_col)
        cal_df = pd.DataFrame({date_col: full_calendar, product_col: pid})

        merged = pd.merge(cal_df, group[[date_col, product_col, demand_col]], on=[date_col, product_col], how="left")
        merged[demand_col] = merged[demand_col].fillna(0.0).astype(float)
        reindexed_frames.append(merged)

    if not reindexed_frames:
        return pd.DataFrame(columns=[date_col, product_col, demand_col])

    out = pd.concat(reindexed_frames, ignore_index=True)
    return out.sort_values(by=[product_col, date_col]).reset_index(drop=True)


def generate_seasonal_naive_forecasts(
    contiguous_df: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    seasonal_days: int = 7,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand"
) -> pd.DataFrame:
    """Generate 7-calendar-day seasonal naive forecasts for specified evaluation dates.

    Forecast rule: prediction(p, t) = actual_demand(p, t - 7 days).
    Requires the exact calendar date t - 7 days to exist in the history; returns NaN otherwise.

    Parameters
    ----------
    contiguous_df : pd.DataFrame
        Contiguous daily demand DataFrame with history and evaluation records.
    eval_dates : pd.DatetimeIndex
        Array of target forecast dates to evaluate.
    seasonal_days : int
        Number of calendar days in the seasonal shift (default 7).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [date, product_id, actual_demand, prediction].
    """
    lookup = {
        (str(row[product_col]), row[date_col]): float(row[demand_col])
        for _, row in contiguous_df.iterrows()
    }

    eval_dates_set = set(pd.to_datetime(eval_dates).floor("D"))
    eval_sub = contiguous_df[contiguous_df[date_col].isin(eval_dates_set)].copy()

    delta = pd.Timedelta(days=seasonal_days)
    preds = []
    for _, row in eval_sub.iterrows():
        pid = str(row[product_col])
        dt = row[date_col]
        prior_dt = dt - delta
        val = lookup.get((pid, prior_dt), np.nan)
        preds.append(val)

    eval_sub["actual_demand"] = eval_sub[demand_col].astype(float)
    eval_sub["prediction"] = preds
    return eval_sub[[date_col, product_col, "actual_demand", "prediction"]].reset_index(drop=True)


def generate_rolling_mean_forecasts(
    contiguous_df: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    window_days: int = 7,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand"
) -> pd.DataFrame:
    """Generate 7-day rolling mean forecasts for specified evaluation dates.

    Forecast rule: prediction(p, t) = mean(demand(p, t-7) ... demand(p, t-1)).
    Strictly excludes demand at date t (uses rolling window shifted by 1 day).

    Parameters
    ----------
    contiguous_df : pd.DataFrame
        Contiguous daily demand DataFrame.
    eval_dates : pd.DatetimeIndex
        Array of target forecast dates to evaluate.
    window_days : int
        Duration in calendar days of the historical rolling window (default 7).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [date, product_id, actual_demand, prediction].
    """
    eval_dates_set = set(pd.to_datetime(eval_dates).floor("D"))

    # Compute shifted rolling mean per product
    records = []
    for pid, group in contiguous_df.groupby(product_col):
        grp = group.sort_values(by=date_col).copy()
        # Shift target demand by 1 to exclude date t, then apply rolling window
        rolled = grp[demand_col].shift(1).rolling(window=window_days, min_periods=window_days).mean()
        grp["prediction"] = rolled
        grp["actual_demand"] = grp[demand_col].astype(float)

        sub = grp[grp[date_col].isin(eval_dates_set)]
        records.append(sub[[date_col, product_col, "actual_demand", "prediction"]])

    if not records:
        return pd.DataFrame(columns=[date_col, product_col, "actual_demand", "prediction"])

    out = pd.concat(records, ignore_index=True)
    return out.sort_values(by=[product_col, date_col]).reset_index(drop=True)


def create_dataco_tabular_features(
    contiguous_df: pd.DataFrame,
    lags: List[int] = [1, 7, 14, 28],
    windows: List[int] = [7, 14, 28],
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand"
) -> pd.DataFrame:
    """Generate calendar features and leak-free lag/rolling statistics for Random Forest.

    All lags and rolling statistics are computed strictly on past observations (shifted by 1).
    """
    df = contiguous_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Calendar indicators
    dt = df[date_col].dt
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["day_of_month"] = dt.day
    df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)

    # Autoregressive lags and rolling statistics per product
    feat_dfs = []
    for pid, group in df.groupby(product_col):
        grp = group.sort_values(by=date_col).copy()

        # Lag features
        for l in lags:
            grp[f"lag_{l}"] = grp[demand_col].shift(l)

        # Shifted rolling mean and std
        shifted_demand = grp[demand_col].shift(1)
        for w in windows:
            grp[f"rolling_mean_{w}"] = shifted_demand.rolling(window=w, min_periods=w).mean()
            grp[f"rolling_std_{w}"] = shifted_demand.rolling(window=w, min_periods=w).std()

        feat_dfs.append(grp)

    out = pd.concat(feat_dfs, ignore_index=True)
    return out.sort_values(by=[product_col, date_col]).reset_index(drop=True)


def train_and_predict_random_forest(
    features_df: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    eval_dates: pd.DatetimeIndex,
    feature_cols: List[str],
    date_col: str = "date",
    product_col: str = "product_id",
    target_col: str = "demand",
    random_state: int = 42,
    n_estimators: int = 100
) -> Tuple[pd.DataFrame, RandomForestRegressor]:
    """Train Random Forest strictly on training observations and generate forecasts.

    Parameters
    ----------
    features_df : pd.DataFrame
        Complete DataFrame with features and target.
    train_dates : pd.DatetimeIndex
        Dates reserved strictly for model fitting.
    eval_dates : pd.DatetimeIndex
        Dates reserved for rolling out-of-sample forecast generation.
    feature_cols : List[str]
        List of feature column names.
    random_state : int
        Seed for reproducibility (default 42).
    n_estimators : int
        Number of trees (default 100).

    Returns
    -------
    Tuple[pd.DataFrame, RandomForestRegressor]
        - Predictions DataFrame with columns [date, product_id, actual_demand, prediction]
        - Fitted RandomForestRegressor instance
    """
    train_set = set(pd.to_datetime(train_dates).floor("D"))
    eval_set = set(pd.to_datetime(eval_dates).floor("D"))

    # Train data: non-null features and targets strictly within training dates
    train_mask = features_df[date_col].isin(train_set)
    train_clean = features_df[train_mask].dropna(subset=feature_cols + [target_col])

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col].astype(float)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Eval data: predict on eval dates
    eval_sub = features_df[features_df[date_col].isin(eval_set)].copy()
    
    # Rows with missing feature history receive NaN prediction rather than imputed zero
    eval_clean_idx = eval_sub.dropna(subset=feature_cols).index
    
    preds_series = pd.Series(np.nan, index=eval_sub.index, dtype=float)
    if len(eval_clean_idx) > 0:
        preds_series.loc[eval_clean_idx] = rf.predict(eval_sub.loc[eval_clean_idx, feature_cols])

    eval_sub["actual_demand"] = eval_sub[target_col].astype(float)
    eval_sub["prediction"] = preds_series

    out = eval_sub[[date_col, product_col, "actual_demand", "prediction"]].reset_index(drop=True)
    return out, rf


def generate_dataco_fixed_origin_forecasts(
    contiguous_df: pd.DataFrame,
    forecast_origin: str = "2017-08-31",
    horizon_days: int = 8,
    product_ids: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    fitted_model: Optional[Any] = None,
    train_dates: Optional[pd.DatetimeIndex] = None,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand",
    random_state: int = 42,
    n_estimators: int = 100,
    model_name: str = "Random Forest",
) -> pd.DataFrame:
    """Generate multi-step recursive forecasts from a fixed decision origin date.

    Guarantees strict leakage prevention:
    - History is truncated to date <= forecast_origin.
    - Day h=1 uses historical actuals through forecast_origin.
    - Days h=2..horizon_days recursively use previous model predictions as lag inputs.
    - Zero actual demand from dates > forecast_origin is ever accessed or consumed.

    Parameters
    ----------
    contiguous_df : pd.DataFrame
        Full contiguous daily demand DataFrame.
    forecast_origin : str
        Fixed decision cutoff date (e.g. '2017-08-31').
    horizon_days : int
        Number of forward forecast steps (default 8 days for L=7 + R=1).
    product_ids : Optional[List[str]]
        Cohort of product IDs to forecast. Defaults to DATACO_CORE_9_PRODUCT_IDS.
    feature_cols : Optional[List[str]]
        Feature column names for the model.
    fitted_model : Optional[Any]
        Pre-trained regressor. If None, trains a Random Forest on historical data.
    train_dates : Optional[pd.DatetimeIndex]
        Training dates if fitting a new model.
    date_col : str
        Date column header.
    product_col : str
        Product SKU ID column header.
    demand_col : str
        Demand quantity column header.
    random_state : int
        Random seed for reproducibility.
    n_estimators : int
        Number of trees if fitting a new model.
    model_name : str
        Name of the forecasting model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        [product_id, forecast_origin, forecast_date, forecast_horizon_step,
         predicted_demand, model_name, forecast_method]
    """
    if feature_cols is None:
        feature_cols = [
            "day_of_week", "month", "day_of_month", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_std_7",
            "rolling_mean_14", "rolling_std_14",
            "rolling_mean_28", "rolling_std_28",
        ]

    origin_dt = pd.to_datetime(forecast_origin).floor("D")

    if product_ids is None:
        product_ids = [str(p) for p in DATACO_CORE_9_PRODUCT_IDS]
    else:
        product_ids = [str(p) for p in product_ids]

    # LEAKAGE SAFEGUARD: Strictly truncate historical data to date <= origin_dt
    hist_clean = contiguous_df[
        (contiguous_df[product_col].astype(str).isin(product_ids)) &
        (pd.to_datetime(contiguous_df[date_col]).dt.floor("D") <= origin_dt)
    ].copy()
    hist_clean[date_col] = pd.to_datetime(hist_clean[date_col]).dt.floor("D")
    hist_clean[product_col] = hist_clean[product_col].astype(str)

    # Train model if not provided
    rf = fitted_model
    if rf is None:
        features_df = create_dataco_tabular_features(
            hist_clean, date_col=date_col, product_col=product_col, demand_col=demand_col
        )
        if train_dates is not None:
            train_set = set(pd.to_datetime(train_dates).floor("D"))
            train_mask = features_df[date_col].isin(train_set)
        else:
            train_mask = features_df[date_col] <= origin_dt

        train_clean = features_df[train_mask].dropna(subset=feature_cols + [demand_col])
        if train_clean.empty:
            raise ValueError(f"No valid training records found on or before origin '{forecast_origin}'.")

        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        rf.fit(train_clean[feature_cols], train_clean[demand_col].astype(float))

    forecast_records = []
    for pid in product_ids:
        p_sub = hist_clean[hist_clean[product_col] == pid].sort_values(by=date_col)
        if len(p_sub) < 28:
            # Insufficient history to compute lags and rolling windows
            for h in range(1, horizon_days + 1):
                tgt_dt = origin_dt + pd.Timedelta(days=h)
                forecast_records.append({
                    product_col: pid,
                    "forecast_origin": origin_dt.strftime("%Y-%m-%d"),
                    "forecast_date": tgt_dt.strftime("%Y-%m-%d"),
                    "forecast_horizon_step": h,
                    "predicted_demand": np.nan,
                    "model_name": model_name,
                    "forecast_method": "recursive_multi_step_insufficient_history",
                })
            continue

        # Build dynamic time series dictionary: date -> demand
        series_dict = {
            row[date_col]: float(row[demand_col]) for _, row in p_sub.iterrows()
        }

        # Recursive multi-step rollout
        for h in range(1, horizon_days + 1):
            tgt_dt = origin_dt + pd.Timedelta(days=h)
            dow = tgt_dt.dayofweek
            month = tgt_dt.month
            dom = tgt_dt.day
            is_we = 1 if dow in [5, 6] else 0

            # Lags: references series_dict (which contains historical actuals for t <= origin
            # and recursive predictions for t > origin)
            lag1 = series_dict.get(tgt_dt - pd.Timedelta(days=1), np.nan)
            lag7 = series_dict.get(tgt_dt - pd.Timedelta(days=7), np.nan)
            lag14 = series_dict.get(tgt_dt - pd.Timedelta(days=14), np.nan)
            lag28 = series_dict.get(tgt_dt - pd.Timedelta(days=28), np.nan)

            # Rolling statistics (strictly preceding target date)
            w7_vals = [series_dict.get(tgt_dt - pd.Timedelta(days=k), np.nan) for k in range(1, 8)]
            rm7 = float(np.nanmean(w7_vals)) if not any(np.isnan(w7_vals)) else np.nan
            rs7 = float(np.nanstd(w7_vals, ddof=1)) if not any(np.isnan(w7_vals)) and len(w7_vals) > 1 else 0.0

            w14_vals = [series_dict.get(tgt_dt - pd.Timedelta(days=k), np.nan) for k in range(1, 15)]
            rm14 = float(np.nanmean(w14_vals)) if not any(np.isnan(w14_vals)) else np.nan
            rs14 = float(np.nanstd(w14_vals, ddof=1)) if not any(np.isnan(w14_vals)) and len(w14_vals) > 1 else 0.0

            w28_vals = [series_dict.get(tgt_dt - pd.Timedelta(days=k), np.nan) for k in range(1, 29)]
            rm28 = float(np.nanmean(w28_vals)) if not any(np.isnan(w28_vals)) else np.nan
            rs28 = float(np.nanstd(w28_vals, ddof=1)) if not any(np.isnan(w28_vals)) and len(w28_vals) > 1 else 0.0

            feat_dict = {
                "day_of_week": dow,
                "month": month,
                "day_of_month": dom,
                "is_weekend": is_we,
                "lag_1": lag1,
                "lag_7": lag7,
                "lag_14": lag14,
                "lag_28": lag28,
                "rolling_mean_7": rm7,
                "rolling_std_7": rs7,
                "rolling_mean_14": rm14,
                "rolling_std_14": rs14,
                "rolling_mean_28": rm28,
                "rolling_std_28": rs28,
            }

            if any(np.isnan(v) for v in feat_dict.values()):
                pred_val = np.nan
            else:
                feat_df = pd.DataFrame([feat_dict])[feature_cols]
                raw_pred = float(rf.predict(feat_df)[0])
                pred_val = max(0.0, raw_pred)

            # Store in series_dict for subsequent recursive lag evaluation
            series_dict[tgt_dt] = pred_val

            forecast_records.append({
                product_col: pid,
                "forecast_origin": origin_dt.strftime("%Y-%m-%d"),
                "forecast_date": tgt_dt.strftime("%Y-%m-%d"),
                "forecast_horizon_step": h,
                "predicted_demand": round(pred_val, 4) if np.isfinite(pred_val) else np.nan,
                "model_name": model_name,
                "forecast_method": "recursive_multi_step",
            })

    out_df = pd.DataFrame(forecast_records)
    return out_df.sort_values(by=[product_col, "forecast_horizon_step"]).reset_index(drop=True)


def generate_dataco_validation_forecasts(
    contiguous_df: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    feature_cols: Optional[List[str]] = None,
    product_ids: Optional[List[str]] = None,
    date_col: str = "date",
    product_col: str = "product_id",
    demand_col: str = "demand",
    random_state: int = 42,
    n_estimators: int = 100
) -> pd.DataFrame:
    """Generate out-of-sample forecasts on the pre-decision validation window.

    Fits Random Forest strictly on train_dates and predicts on val_dates.
    Guarantees all actual demands and predictions occur strictly on or before
    val_dates.max(), serving as leak-free data for uncertainty estimation.
    """
    if product_ids is None:
        product_ids = [str(p) for p in DATACO_CORE_9_PRODUCT_IDS]
    else:
        product_ids = [str(p) for p in product_ids]

    if feature_cols is None:
        feature_cols = [
            "day_of_week", "month", "day_of_month", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_std_7",
            "rolling_mean_14", "rolling_std_14",
            "rolling_mean_28", "rolling_std_28",
        ]

    val_max = pd.to_datetime(val_dates).max().floor("D")

    # Strictly filter contiguous_df to records <= val_max
    df_sub = contiguous_df[
        (contiguous_df[product_col].astype(str).isin(product_ids)) &
        (pd.to_datetime(contiguous_df[date_col]).dt.floor("D") <= val_max)
    ].copy()

    features_df = create_dataco_tabular_features(
        df_sub, date_col=date_col, product_col=product_col, demand_col=demand_col
    )

    preds_df, _ = train_and_predict_random_forest(
        features_df=features_df,
        train_dates=train_dates,
        eval_dates=val_dates,
        feature_cols=feature_cols,
        date_col=date_col,
        product_col=product_col,
        target_col=demand_col,
        random_state=random_state,
        n_estimators=n_estimators
    )
    return preds_df

