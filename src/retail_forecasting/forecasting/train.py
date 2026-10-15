"""Model training and chronological evaluation pipelines.

Implements:
1. Strict chronological temporal split across all products with common cutoff.
2. Scikit-learn RandomForestRegressor training on eligible historical features.
3. Rolling one-day-ahead out-of-sample forecast generation over the 28-day test period.
4. Paired model evaluation aligning Random Forest and Seasonal Naive on identical product-date observations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from retail_forecasting.forecasting.baseline import SeasonalNaiveForecaster
from retail_forecasting.forecasting.predict import generate_forecasts
from retail_forecasting.forecasting.evaluate import compute_metrics, compute_per_product_metrics


DEFAULT_FEATURE_COLS = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
]


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    test_days: int = 28,
    product_col: str = "product_id"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split time-series dataset chronologically using a common cutoff across all products.

    Never uses random row splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date and product identifier.
    date_col : str
        Name of the date column.
    test_days : int
        Duration in calendar days reserved for the final test period (default 28).
    product_col : str
        Product SKU identifier column name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        (train_df, test_df, split_report)

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    KeyError
        If date_col is not present.
    ValueError
        If duplicate (product_id, date) rows exist or dataset history is shorter than test_days.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in dataframe.")

    if df.empty:
        return df.copy(), df.copy(), {"train_rows": 0, "test_rows": 0}

    # Ensure date column is datetime
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[date_col]):
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

    # Check for duplicate product-date entries
    if product_col in df_sorted.columns:
        dups = int(df_sorted.duplicated(subset=[product_col, date_col]).sum())
        if dups > 0:
            raise ValueError(
                f"Detected {dups} duplicate entries for ({product_col}, {date_col}). "
                "Each product must have at most one observation per date."
            )

    # Sort strictly chronologically
    sort_cols = [date_col, product_col] if product_col in df_sorted.columns else [date_col]
    df_sorted = df_sorted.sort_values(by=sort_cols).reset_index(drop=True)

    max_date = df_sorted[date_col].max()
    min_date = df_sorted[date_col].min()
    total_calendar_days = (max_date - min_date).days + 1

    if total_calendar_days <= test_days:
        raise ValueError(
            f"Dataset total span ({total_calendar_days} days) is less than or equal to "
            f"requested test window ({test_days} days). Cannot construct train/test split."
        )

    # Cutoff date: the last day of training
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    train_df = df_sorted[df_sorted[date_col] <= cutoff_date].copy().reset_index(drop=True)
    test_df = df_sorted[df_sorted[date_col] > cutoff_date].copy().reset_index(drop=True)

    split_report = {
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()),
        "cutoff_date": str(cutoff_date.date()),
        "test_start_date": str((cutoff_date + pd.Timedelta(days=1)).date()),
        "test_end_date": str(max_date.date()),
        "test_horizon_days": test_days,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_products_count": int(train_df[product_col].nunique()) if product_col in train_df.columns else 0,
        "test_products_count": int(test_df[product_col].nunique()) if product_col in test_df.columns else 0,
    }

    return train_df, test_df, split_report


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> RandomForestRegressor:
    """Train Random Forest Regressor on feature matrix.

    Requirements:
    - Target column is strictly excluded from X_train.
    - No non-finite values (NaN / Inf) in training features or target.
    - Uses reproducible random_state.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target series.
    model_params : Optional[Dict[str, Any]]
        Hyperparameters for RandomForestRegressor.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    RandomForestRegressor
        Fitted Scikit-Learn regressor.

    Raises
    ------
    ValueError
        If X_train contains target column, non-numeric strings, or non-finite values.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for X_train, got {type(X_train).__name__}")

    if len(X_train) == 0:
        raise ValueError("Cannot train model on empty feature matrix X_train.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but y_train has {len(y_train)} rows."
        )

    # Convert features and target to float arrays for clean validation and sklearn fitting
    try:
        X_mat = np.asarray(X_train, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"X_train contains non-numeric values that cannot be converted to float: {e}")

    if not np.isfinite(X_mat).all():
        raise ValueError("X_train contains non-finite values (NaN or Inf). Filter to model-eligible rows before training.")

    try:
        y_vec = np.asarray(y_train, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"y_train contains non-numeric values that cannot be converted to float: {e}")

    if not np.isfinite(y_vec).all():
        raise ValueError("y_train contains non-finite values (NaN or Inf).")

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if model_params:
        params.update(model_params)

    model = RandomForestRegressor(**params)
    X_train_df = pd.DataFrame(X_mat, columns=X_train.columns, index=X_train.index)
    model.fit(X_train_df, y_vec)
    return model


def run_rolling_evaluation(
    feature_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "demand",
    date_col: str = "date",
    product_col: str = "product_id",
    test_days: int = 28,
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute end-to-end Phase 5 rolling one-day-ahead forecast evaluation.

    Workflow:
    1. Chronological split of feature_df into train (before cutoff) and test (final 28 days).
    2. Filter training set to model-eligible rows (where is_model_eligible == True).
    3. Fit Random Forest on train partition features only.
    4. Fit Seasonal Naive benchmark on training historical actuals.
    5. Generate rolling one-day-ahead predictions for both models over the test period.
    6. Align predictions: evaluate both models on the EXACT same product-date observations
       where both have valid (non-NaN) predictions.
    7. Compute MAE, RMSE, and WAPE across overall and per-product segments.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature-engineered dataframe containing target, lag, rolling, calendar, and eligibility columns.
    feature_cols : Optional[List[str]]
        List of feature column names to feed to Random Forest.
    target_col : str
        Demand target column.
    date_col : str
        Date column.
    product_col : str
        Product SKU column.
    test_days : int
        Number of test days (default 28).
    model_params : Optional[Dict[str, Any]]
        Hyperparameters for Random Forest.
    random_state : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'split_report': summary of train/test dates and row counts
        - 'evaluation_table': pd.DataFrame with aligned predictions and actuals
        - 'metrics': Dict comparing Random Forest vs. Seasonal Naive
        - 'per_product_metrics': Dict with macro summaries and per-product tables
        - 'exclusions': Dict detailing excluded rows and rationale
        - 'model': fitted RandomForestRegressor
    """
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in feature_df.columns]

    # Verify target is excluded from features
    if target_col in feature_cols:
        raise ValueError(f"Target column '{target_col}' must not be included in feature_cols.")

    # 1. Chronological split
    train_df, test_df, split_report = temporal_train_test_split(
        feature_df, date_col=date_col, test_days=test_days, product_col=product_col
    )

    # 2. Filter training set to model-eligible rows with complete feature coverage
    if "is_model_eligible" in train_df.columns:
        train_eligible = train_df[train_df["is_model_eligible"] == True].copy()
    else:
        train_eligible = train_df.copy()
    train_eligible = train_eligible.dropna(subset=feature_cols + [target_col]).copy()

    X_train = train_eligible[feature_cols]
    y_train = train_eligible[target_col]

    # 3. Fit Random Forest
    rf_model = train_random_forest(
        X_train, y_train, model_params=model_params, random_state=random_state
    )

    # 4. Fit Seasonal Naive baseline on historical data
    sn_forecaster = SeasonalNaiveForecaster(seasonal_days=7)
    sn_forecaster.fit(
        train_df, date_col=date_col, product_col=product_col, target_col=target_col
    )

    # 5. Generate rolling one-day-ahead predictions on test set
    # Note: In our feature pipeline, lag and rolling features for row t strictly use data <= t-1.
    # Therefore, test_df features already reflect rolling 1-day-ahead information available at t-1!
    test_eval = test_df[[date_col, product_col, target_col]].copy()
    test_eval.rename(columns={target_col: "actual_demand"}, inplace=True)

    # Random Forest predictions (only for test rows with complete features)
    rf_eligible_mask = test_df[feature_cols].notna().all(axis=1)
    rf_preds_full = np.full(len(test_df), np.nan)
    if rf_eligible_mask.any():
        rf_preds = generate_forecasts(rf_model, test_df.loc[rf_eligible_mask, feature_cols])
        rf_preds_full[rf_eligible_mask.values] = rf_preds

    test_eval["random_forest_pred"] = rf_preds_full

    # Seasonal Naive predictions: using train + test actuals for rolling lookup of t - 7 days
    # (strictly lookup prior dates, never future test dates!)
    combined_history = pd.concat([train_df, test_df], ignore_index=True)
    sn_preds = sn_forecaster.predict(
        test_df,
        date_col=date_col,
        product_col=product_col,
        history_df=combined_history,
    )
    test_eval["seasonal_naive_pred"] = sn_preds

    # 6. Align evaluation population (identically valid observations)
    valid_mask = test_eval["random_forest_pred"].notna() & test_eval["seasonal_naive_pred"].notna()
    aligned_eval = test_eval[valid_mask].copy().reset_index(drop=True)

    # Audit exclusions
    total_test_rows = len(test_df)
    evaluated_rows = len(aligned_eval)
    excluded_rows = total_test_rows - evaluated_rows
    excluded_rf_nan = int((~rf_eligible_mask).sum())
    excluded_sn_nan = int(pd.isna(test_eval["seasonal_naive_pred"]).sum())

    exclusions_report = {
        "total_test_observations": total_test_rows,
        "evaluated_observations": evaluated_rows,
        "excluded_observations": excluded_rows,
        "excluded_percentage": round(excluded_rows / total_test_rows * 100.0, 2) if total_test_rows > 0 else 0.0,
        "reasons": {
            "insufficient_history_for_random_forest": excluded_rf_nan,
            "missing_7_day_prior_observation_for_seasonal_naive": excluded_sn_nan,
        },
    }

    # 7. Compute metrics on aligned evaluation observations
    rf_metrics = compute_metrics(aligned_eval["actual_demand"], aligned_eval["random_forest_pred"])
    sn_metrics = compute_metrics(aligned_eval["actual_demand"], aligned_eval["seasonal_naive_pred"])

    # Per-product metrics
    rf_prod_df, rf_macro = compute_per_product_metrics(
        aligned_eval, actual_col="actual_demand", pred_col="random_forest_pred", product_col=product_col
    )
    sn_prod_df, sn_macro = compute_per_product_metrics(
        aligned_eval, actual_col="actual_demand", pred_col="seasonal_naive_pred", product_col=product_col
    )

    metrics_summary = {
        "evaluation_sample_size": evaluated_rows,
        "seasonal_naive": sn_metrics,
        "random_forest": rf_metrics,
        "macro_averaged": {
            "seasonal_naive": sn_macro,
            "random_forest": rf_macro,
        },
    }

    return {
        "split_report": split_report,
        "evaluation_table": aligned_eval,
        "metrics": metrics_summary,
        "per_product": {
            "random_forest": rf_prod_df,
            "seasonal_naive": sn_prod_df,
        },
        "exclusions": exclusions_report,
        "model": rf_model,
        "feature_cols": feature_cols,
    }
