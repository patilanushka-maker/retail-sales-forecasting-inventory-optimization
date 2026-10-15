# System Assumptions, Decisions & Technical Limitations

> **Overview**: Documents foundational assumptions, system boundaries, operational limitations, and open decisions requiring user confirmation.

---

## 1. Operational & Modeling Assumptions

1. **Stationarity of Lead Times**: Initial baseline assumes supplier lead times are normally distributed around a known mean $L$ and standard deviation $\sigma_L$.
2. **Backorder Behavior**: Demand that cannot be met immediately is assumed to either become a backorder (in wholesale/B2B models) or lost sales (typical grocery/retail model). The initial simulation models lost sales.
3. **Simulated Inventory Parameters**:
   - Holding cost rate ($20\%$ annual) and order setup costs ($50 per order) are representative operational assumptions for simulation purposes.
   - These are **distinct from historical retail transactional data** and must be calibrated to specific commercial contracts when deployed.

---

## 2. Technical Limitations

1. **Point Forecasts vs. Full Predictive Distributions**:
   - The initial Random Forest model yields point predictions. Safety stock calculations approximate demand volatility using empirical residual standard deviation $\sigma_{\text{residuals}}$.
   - Quantile regression or probabilistic forecasting (e.g., LightGBM Quantile or DeepAR) can be considered in future phases.
2. **Single-Echelon Scope**:
   - Replenishment logic models single-echelon (store or single warehouse) inventory. Multi-echelon distribution network balancing is out of scope for the MVP.
3. **Exogenous Feature Availability**:
   - Weather forecasts and competitor price scrapers are excluded from baseline modeling unless explicitly available in the chosen public dataset.

---

## 3. Decisions Requiring User Confirmation

| Decision Item | Current Proposed Default | Alternative Options | Status |
| :--- | :--- | :--- | :--- |
| **Public Dataset Selection** | Rossmann Store Sales or Online Retail II / Walmart M5 | Synthetic retail sample or custom enterprise dump | **Pending Decision** |
| **Lost Sales vs. Backorders** | Lost sales assumption (excess demand is forfeited) | Full backlogging (customer waits until stock arrives) | **Pending Decision** |
| **Replenishment Policy** | Continuous $(s, Q)$ review policy (reorder $EOQ$ when $\le ROP$) | Periodic $(R, S)$ review policy (order-up-to level every $R$ days) | **Pending Decision** |
| **Software License** | Decision deferred (marked in `LICENSE`) | MIT, Apache 2.0, or Proprietary | **Pending Decision** |
