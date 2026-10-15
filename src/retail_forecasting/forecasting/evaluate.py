"""Forecast evaluation metrics module.

Computes standard supply chain demand forecasting accuracy metrics:
- MAE  (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error, expressed as a percentage)

Quality & Correctness Invariants:
- Rejects non-finite inputs (NaN / Inf) with descriptive errors.
- Handles zero-denominator WAPE explicitly (returns NaN when total actual demand is 0).
- Reports evaluation sample size alongside each metric.
- Computes both global pooled metrics and macro-averaged per-product metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]]
) -> Dict[str, Any]:
    """Compute MAE, RMSE, and WAPE between ground-truth and predicted demand.

    Parameters
    ----------
    y_true : Union[np.ndarray, pd.Series, List[float]]
        Observed actual demand values.
    y_pred : Union[np.ndarray, pd.Series, List[float]]
        Model forecasted demand values.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'sample_size': int count of evaluated observations
        - 'mae': float Mean Absolute Error
        - 'rmse': float Root Mean Squared Error
        - 'wape': float Weighted Absolute Percentage Error (%)
        - 'wape_defined': bool indicating whether total actual demand > 0

    Raises
    ------
    ValueError
        If lengths do not match or inputs contain NaN / Inf values.
    """
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.asarray(y_pred, dtype="float64")

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true has length {len(y_t)}, but y_pred has length {len(y_p)}."
        )

    if len(y_t) == 0:
        return {
            "sample_size": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "wape": float("nan"),
            "wape_defined": False,
        }

    # Reject non-finite inputs
    if not np.isfinite(y_t).all() or not np.isfinite(y_p).all():
        raise ValueError(
            "Evaluation inputs contain non-finite values (NaN or Inf). "
            "Filter or resolve missing predictions before metric calculation."
        )

    abs_errors = np.abs(y_t - y_p)
    sq_errors = (y_t - y_p) ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    total_actual = float(np.sum(np.abs(y_t)))
    if total_actual > 0:
        wape = float((np.sum(abs_errors) / total_actual) * 100.0)
        wape_defined = True
    else:
        # Undefined when total actual sales is 0 across the entire evaluation horizon
        wape = float("nan")
        wape_defined = False

    return {
        "sample_size": int(len(y_t)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4) if wape_defined else float("nan"),
        "wape_defined": wape_defined,
    }


def compute_per_product_metrics(
    eval_df: pd.DataFrame,
    actual_col: str = "actual_demand",
    pred_col: str = "prediction",
    product_col: str = "product_id"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute accuracy metrics per product SKU and macro-averaged summary.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Dataframe containing actuals and predictions per product.
    actual_col : str
        Column name for ground-truth actual demand.
    pred_col : str
        Column name for model predictions.
    product_col : str
        Column name for product SKU identifier.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (per_product_metrics_df, macro_summary_dict)
    """
    if not isinstance(eval_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(eval_df).__name__}")

    required = [actual_col, pred_col, product_col]
    for c in required:
        if c not in eval_df.columns:
            raise KeyError(f"Required column '{c}' not found in evaluation dataframe.")

    records = []
    for pid, group in eval_df.groupby(product_col):
        m = compute_metrics(group[actual_col], group[pred_col])
        records.append({
            product_col: pid,
            "sample_size": m["sample_size"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "wape": m["wape"],
            "wape_defined": m["wape_defined"],
        })

    product_metrics_df = pd.DataFrame(records)

    # Macro averages (average of per-product metrics)
    valid_wapes = product_metrics_df[product_metrics_df["wape_defined"]]["wape"]
    macro_summary = {
        "evaluated_products_count": len(product_metrics_df),
        "macro_mae": round(float(product_metrics_df["mae"].mean()), 4),
        "macro_rmse": round(float(product_metrics_df["rmse"].mean()), 4),
        "macro_wape": round(float(valid_wapes.mean()), 4) if not valid_wapes.empty else float("nan"),
    }

    return product_metrics_df, macro_summary
