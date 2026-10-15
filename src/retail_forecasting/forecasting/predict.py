"""Forecasting inference and prediction pipeline.

Generates point predictions from trained estimators with configurable non-negative
clipping constraints to enforce physical reality (retail sales demand cannot be negative).
"""

from typing import Any, Union
import numpy as np
import pandas as pd


def generate_forecasts(
    model: Any,
    X: pd.DataFrame,
    clip_non_negative: bool = True
) -> np.ndarray:
    """Generate sales demand predictions on feature matrix.

    Parameters
    ----------
    model : Any
        Fitted model instance with a .predict() method.
    X : pd.DataFrame
        Feature matrix for inference.
    clip_non_negative : bool
        If True (default), clips negative predictions to 0.0 because retail demand
        cannot physically be negative.

    Returns
    -------
    np.ndarray
        Array of predicted sales quantities.

    Raises
    ------
    TypeError
        If X is not a DataFrame or model lacks predict method.
    ValueError
        If X is empty or contains non-finite values.
    """
    if not hasattr(model, "predict"):
        raise TypeError(f"Model object of type {type(model).__name__} does not have a 'predict' method.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for features X, got {type(X).__name__}")

    if X.empty:
        return np.array([], dtype="float64")

    # Convert features to numeric float array
    try:
        X_mat = np.asarray(X, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Feature matrix X contains non-numeric values: {e}")

    # Check for non-finite values
    if not np.isfinite(X_mat).all():
        raise ValueError("Feature matrix X contains non-finite values (NaN or Inf). Only model-eligible rows can be predicted.")

    X_df = pd.DataFrame(X_mat, columns=X.columns, index=X.index)
    predictions = model.predict(X_df)
    predictions = np.asarray(predictions, dtype="float64")

    if clip_non_negative:
        predictions = np.clip(predictions, a_min=0.0, a_max=None)

    return predictions
