"""Seasonal naive benchmark forecasting model.

Provides a 7-calendar-day seasonal persistence baseline:
    y_hat(p, t) = y(p, t - 7 days)
where p is a product and t is the target forecast date.

Requirements & Boundaries:
- Evaluated independently per product.
- Keyed strictly by calendar date (exactly 7 calendar days prior, not merely 7 rows prior).
- Missing historical observations are retained as NaN (never silently zero-filled).
- Target values strictly precede the forecast date (no lookahead leakage).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SeasonalNaiveForecaster:
    """Seasonal Naive benchmark model using 7-calendar-day historical lookup.

    Projects demand at date t as the observed demand for the same product exactly
    7 calendar days prior (t - 7 days), capturing weekly retail buying cycles.
    """

    def __init__(self, seasonal_days: int = 7) -> None:
        """Initialize Seasonal Naive forecaster.

        Parameters
        ----------
        seasonal_days : int
            Number of calendar days in the seasonal cycle (default 7 for weekly seasonality).
        """
        self.seasonal_days = seasonal_days
        self.history_: Optional[pd.DataFrame] = None
        self._history_lookup: Dict[Tuple[str, pd.Timestamp], float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        product_col: str = "product_id",
        target_col: str = "demand"
    ) -> "SeasonalNaiveForecaster":
        """Store historical observations for seasonal replication.

        Parameters
        ----------
        df : pd.DataFrame
            Historical training dataframe.
        date_col : str
            Name of the date column.
        product_col : str
            Name of the product ID column.
        target_col : str
            Name of the demand target column.

        Returns
        -------
        SeasonalNaiveForecaster
            Fitted forecaster instance.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        required = [date_col, product_col, target_col]
        for c in required:
            if c not in df.columns:
                raise KeyError(f"Required column '{c}' not found in dataframe: {list(df.columns)}")

        # Store a clean copy of history with normalized timestamps
        hist = df[[date_col, product_col, target_col]].copy()
        hist[date_col] = pd.to_datetime(hist[date_col]).dt.normalize()
        self.history_ = hist

        # Build fast (product_id, date) lookup dictionary using vectorized zip
        valid_hist = hist.dropna(subset=[target_col])
        keys = zip(valid_hist[product_col], valid_hist[date_col])
        values = valid_hist[target_col].astype(float)
        self._history_lookup = dict(zip(keys, values))

        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        date_col: str = "date",
        product_col: str = "product_id",
        history_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Generate 7-calendar-day seasonal naive predictions for test rows.

        For rolling evaluation, if history_df is provided (e.g. combined train + elapsed test history),
        it incorporates those observations for historical lookup without lookahead.

        Parameters
        ----------
        test_df : pd.DataFrame
            Dataframe of test dates and products to forecast.
        date_col : str
            Name of the date column.
        product_col : str
            Name of the product ID column.
        history_df : Optional[pd.DataFrame]
            Optional extended history dataframe for rolling one-day-ahead lookups.

        Returns
        -------
        np.ndarray
            Array of predicted values. Returns NaN if the exact observation 7 days
            prior is absent from historical data.
        """
        if not isinstance(test_df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(test_df).__name__}")

        # Update lookup if additional history provided
        lookup = dict(self._history_lookup)
        if history_df is not None:
            hist_ext = history_df[[date_col, product_col, "demand" if "demand" in history_df.columns else history_df.columns[-1]]].copy()
            hist_ext[date_col] = pd.to_datetime(hist_ext[date_col]).dt.normalize()
            target_name = hist_ext.columns[-1]
            valid_ext = hist_ext.dropna(subset=[target_name])
            keys = zip(valid_ext[product_col], valid_ext[date_col])
            vals = valid_ext[target_name].astype(float)
            lookup.update(zip(keys, vals))

        predictions = []
        delta = pd.Timedelta(days=self.seasonal_days)

        test_dates = pd.to_datetime(test_df[date_col]).dt.normalize()
        product_ids = test_df[product_col].values

        for dt, pid in zip(test_dates, product_ids):
            target_prior_dt = dt - delta
            # Lookup exact 7-calendar-day prior demand
            val = lookup.get((pid, target_prior_dt), np.nan)
            predictions.append(val)

        return np.array(predictions, dtype="float64")


def generate_seasonal_naive_predictions(
    df: pd.DataFrame,
    date_col: str = "date",
    product_col: str = "product_id",
    target_col: str = "demand",
    seasonal_days: int = 7
) -> pd.Series:
    """Convenience function generating 7-calendar-day seasonal naive forecasts.

    Performs exact calendar-day lookup: y_hat(p, t) = y(p, t - 7 days).
    Retains NaN if t - 7 days is not present in the historical record.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing date, product_id, and demand.
    date_col : str
        Date column name.
    product_col : str
        Product ID column name.
    target_col : str
        Target demand column name.
    seasonal_days : int
        Number of calendar days prior (default 7).

    Returns
    -------
    pd.Series
        Series of seasonal naive predictions indexed to match df.
    """
    forecaster = SeasonalNaiveForecaster(seasonal_days=seasonal_days)
    forecaster.fit(df, date_col=date_col, product_col=product_col, target_col=target_col)
    preds = forecaster.predict(df, date_col=date_col, product_col=product_col)
    return pd.Series(preds, index=df.index, name="seasonal_naive_pred")
