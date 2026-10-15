# Model Evaluation & Benchmark Report

> **Status:** Scaffolding initialized. Benchmarking results will be logged after experimental runs.

---

## 1. Evaluation Methodology

### Chronological Train-Test Split
- Models are strictly evaluated on unseen future horizons (e.g., final 28 days) to simulate real-world production forecasting.
- Random or shuffled k-fold cross-validation is avoided to prevent lookahead data leakage.

### Evaluation Metrics
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$
- **Weighted Absolute Percentage Error (WAPE)**:
  $$\text{WAPE} = \frac{\sum_{t=1}^{n} |y_t - \hat{y}_t|}{\sum_{t=1}^{n} y_t}$$

---

## 2. Benchmark Comparison (Pending Model Runs)

| Model | MAE | RMSE | WAPE | Training Runtime (s) | Inference Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seasonal Naive Baseline** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Random Forest Regressor** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 3. Residual & Error Analysis (Planned)
- Heteroskedasticity and residual autocorrelation assessment.
- Error breakdown by product category and sales velocity tier (Fast-moving vs. Slow-moving SKUs).
