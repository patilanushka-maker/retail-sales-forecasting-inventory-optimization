# Model Architecture, Feature Contracts & Learning Mechanics

## 1. Feature & Target Contract

The forecasting system models consumer product demand at daily frequency.

### Target Definition
- **Target ($y_t$)**: `demand` at date $t$ for product $p$.
- **Strict Invariance**: Features for predicting $y_t$ use information available **strictly prior to $t$** ($t-1, t-2, \dots$).

### Feature Set Breakdown
1. **Calendar Features (Derived from Date at $t$)**:
   - `day_of_week`: Integer day index (0 = Monday, 6 = Sunday).
   - `month`: Integer calendar month (1 to 12).
   - `day_of_month`: Day of the month (1 to 31).
   - `is_weekend`: Binary flag (1 if Saturday or Sunday, else 0).
2. **Autoregressive Lags**:
   - `lag_1`: Demand at $t-1$.
   - `lag_7`: Demand at $t-7$ (capturing weekly seasonal cycle).
   - `lag_14`: Demand at $t-14$ (two-week persistence).
   - `lag_28`: Demand at $t-28$ (four-week persistence).
3. **Shifted Rolling Window Statistics**:
   - **Convention**: The demand series is shifted by 1 row (`.shift(1)`) before calculating rolling windows:
     $$\text{rolling\_mean}_w(t) = \frac{1}{w} \sum_{k=1}^{w} y_{t-k}$$
     $$\text{rolling\_std}_w(t) = \sqrt{\frac{1}{w-1} \sum_{k=1}^{w} (y_{t-k} - \text{rolling\_mean}_w(t))^2}$$
   - Window sizes: 7 days, 14 days, 28 days.
   - All rolling metrics require at least $w$ valid historical days; unobserved rows remain `NaN`.

---

## 2. Model Eligibility & Missing History Policy
- **No Silent Zero-Filling**: Unobserved historical lags and rolling windows are retained as `NaN`. A missing historical observation is not equivalent to zero demand.
- **Eligibility Indicator (`is_model_eligible`)**:
  - A row is flagged as `is_model_eligible = True` if and only if all primary historical features (`lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_28`) are populated.
  - Rows during the initial 28-day warm-up window for each product are flagged as ineligible for training/testing.

---

## 3. Product Catalog History Diagnostics (UCI Online Retail)

Analysis of 1,053,334 product-days across 3,804 distinct catalog products:

| Catalog History Tier | Calendar Span | Product Count | % of Catalog | Modeling Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| **Ultra-Sparse** | $< 7$ days | 155 | 4.1% | Ineligible for 7d/28d lags; heuristic fallback |
| **Short History** | $7 - 28$ days | 75 | 2.0% | Ineligible for 28d lag; heuristic fallback |
| **Moderate History** | $29 - 90$ days | 291 | 7.6% | Eligible for 28d lag; shorter training history |
| **Long History** | $91 - 180$ days | 407 | 10.7% | Fully eligible |
| **Mature Catalog** | $> 180$ days | 2,876 | 75.6% | Primary modeling cohort |

---

## 4. Chronological Splitting Protocol

- **Single Global Cutoff**: All products are partitioned using the same chronological cutoff date.
  - Cutoff Date: $T_{\text{cutoff}} = T_{\text{max}} - 28\text{ days}$.
  - **Training Partition**: Observations with $\text{date} \le T_{\text{cutoff}}$.
  - **Testing Partition**: Observations with $\text{date} > T_{\text{cutoff}}$ (the final 28 calendar days).
- **Leakage Prevention**:
  - No random row splitting (`train_test_split` with shuffling is strictly prohibited).
  - Train and test date ranges are strictly disjoint ($\text{train\_dates} \cap \text{test\_dates} = \emptyset$).
  - Target values from the test period are never included in the training partition or used to fit estimators.

---

## 5. Seasonal-Naive Benchmark Model

### Mathematical Formulation
$$\hat{y}_{p,t} = y_{p, t - 7\text{ days}}$$

where $p$ is the product SKU and $t$ is the forecast target date.

### Operational Characteristics
- Evaluated independently per product.
- **Calendar Alignment**: Uses exact 7-calendar-day historical lookup ($t - 7\text{ days}$), not simply the 7th prior transaction row.
- **Missing Value Handling**: If the exact 7-calendar-day historical observation is missing from the record, the model outputs `NaN`. It **never** silently replaces missing history with zero.
- **Limitations**:
  - Assumes exact weekly recurrence of demand volumes.
  - Cannot adapt to long-term trend, promotional shocks, day-of-month effects, or cross-product dynamics.
  - Fails on products introduced fewer than 7 days ago or products with gaps in historical sales.

---

## 6. Random Forest Regression Model

### Algorithm & Hyperparameters
Implemented using Scikit-Learn `RandomForestRegressor`:
- `n_estimators`: 100 decision trees.
- `max_depth`: 10 (controls tree depth, mitigating overfitting on noisy daily retail demand).
- `random_state`: 42 (ensures deterministic reproducibility).
- `n_jobs`: -1 (parallel CPU tree construction).

### Feature Matrix (14 Features)
- Calendar (4): `day_of_week`, `month`, `day_of_month`, `is_weekend`.
- Autoregressive Lags (4): `lag_1`, `lag_7`, `lag_14`, `lag_28`.
- Shifted Rolling Windows (6): `rolling_mean_7`, `rolling_std_7`, `rolling_mean_14`, `rolling_std_14`, `rolling_mean_28`, `rolling_std_28`.

### Non-Negative Prediction Constraint
Retail sales demand cannot physically be negative. Because regression trees or tree averaging might theoretically extrapolate or predict edge cases, predictions are clipped to non-negative values:
$$\hat{y}_{\text{final}} = \max(0.0, \hat{y})$$

### Product Identity Policy
`product_id` is excluded as a raw numerical feature to avoid introducing spurious ordinal ordering. Features strictly represent observable demand velocity and calendar context.

---

## 7. Rolling One-Day-Ahead Evaluation Protocol

### Rolling 1-Day-Ahead vs. Fixed-Origin Multi-Step
- **Rolling One-Day-Ahead Evaluation**: Evaluates out-of-sample prediction at date $t$ using historical actuals available through $t-1$. At date $t+1$, actuals from date $t$ become available for rolling features and baseline lookup. This simulates daily operational retail forecasting.
- **Fixed-Origin Multi-Day Forecast**: Generates predictions for $t+1 \dots t+28$ simultaneously from origin $t_0$ without observing intermediate actuals. This is reserved for multi-step replenishment simulations and is not evaluated here.

### Paired Comparison Alignment
The Seasonal Naive baseline and Random Forest regressor are evaluated on the **exact same product-date observations**. Any row where either model produces `NaN` (due to missing 7-day prior history or insufficient lag warm-up) is excluded from both models to ensure an unbiased, like-for-like comparison.

---

## 8. Forecast Accuracy Metrics

### Mean Absolute Error (MAE)
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
Measures average magnitude of absolute forecasting errors in unit volume.

### Root Mean Squared Error (RMSE)
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
Penalizes larger forecast errors more heavily, reflecting operational disruption caused by massive demand misestimates.

### Weighted Absolute Percentage Error (WAPE)
$$WAPE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i|} \times 100\%$$
- Preferred over MAPE for retail demand because it is robust to zero-demand days ($y_i = 0$).
- **Zero-Denominator Boundary**: When total actual demand $\sum |y_i| = 0$, WAPE is mathematically undefined; our metric implementation returns `NaN` with `wape_defined = False` rather than raising a division error.

---

## 9. Empirical Experiment Results (UCI Online Retail Dataset)

### Dataset & Partition Summary
- **Dataset History**: `2010-12-01` to `2011-12-09` (374 calendar days).
- **Cutoff Date**: `2011-11-11`.
- **Training Set**: 983,489 rows across 3,774 products (`2010-12-01` to `2011-11-11`).
- **Test Set**: 69,845 rows across 2,847 products (`2011-11-12` to `2011-12-09`, 28 days).
- **Evaluated Sample Size**: **68,907 observations** across 2,813 distinct products.
- **Excluded Observations**: 938 rows (1.34%) due to incomplete lag warm-up history (<28 days) or missing 7-day prior observations.

### Overall Pooled Performance (68,907 Product-Days)
| Model | MAE (Units) | RMSE (Units) | WAPE (%) | Performance Delta vs. Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Seasonal Naive Baseline** | 10.7714 | 42.5609 | 110.88% | Benchmark |
| **Random Forest Regressor** | **9.4424** | **34.7948** | **97.20%** | **-12.3% MAE, -18.2% RMSE, -13.68% WAPE** |

### Macro-Averaged Per-Product Performance (2,813 Products)
| Model | Macro MAE | Macro RMSE | Macro WAPE (%) |
| :--- | :--- | :--- | :--- |
| **Seasonal Naive** | 9.9543 | 18.6490 | 146.29% |
| **Random Forest** | 8.7747 | 14.7720 | 189.95% |

*Note on Macro WAPE*: Macro-averaged WAPE computes the arithmetic mean of individual product WAPEs. For ultra-slow-moving catalog items with only 1-2 total sales over the 28-day window, small unit forecast errors produce high individual WAPEs (e.g. predicting 2 units when actual is 0 yields undefined or inflated ratios), elevating macro WAPE. Global pooled WAPE (97.20% vs 110.88%) correctly weights unit volume across the business.

---

## 10. Model Limitations & Scope Boundaries
1. **Demand vs. Sales Truncation**: Models are trained on observed transaction demand. True unconstrained customer demand during historical stockout events is unobserved.
2. **Promotions & Price Elasticity**: The UCI dataset lacks promotional markdown calendars; unit prices fluctuate slightly but are not explicitly modeled as dynamic causal drivers in Phase 5.
3. **No Claim of Optimal Inventory**: Lower MAE/RMSE indicates superior statistical fit; it does not automatically guarantee optimal inventory levels without proper safety stock sizing and service level trade-off analysis (Phase 6).
