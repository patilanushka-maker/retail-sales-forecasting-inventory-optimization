# DataCo Inventory Optimization & Replenishment Policy Documentation

> **Document Status**: Audit & Correction Report (Phase 9 Post-Audit)  
> **Target Dataset**: DataCo Smart Supply Chain (`data/raw/DataCoSupplyChainDataset.csv`)  
> **Input Forecast Artifact (Rolling Benchmark)**: `outputs/dataco/forecasts/dataco_forecast_comparison.parquet`  
> **Pre-Decision Validation Forecasts**: `outputs/dataco/forecasts/dataco_validation_forecasts.parquet`  
> **Fixed-Origin Forecast Artifact**: `outputs/dataco/inventory/dataco_fixed_origin_forecasts.parquet`  
> **Output Recommendations**: `outputs/dataco/inventory/dataco_inventory_recommendations.parquet`  
> **Actionable Purchase Orders**: `outputs/dataco/inventory/dataco_replenishment_orders.csv`  
> **Decision Origin Date**: August 31, 2017 (`2017-08-31`)  
> **Review Date**: October 15, 2026  
> **Author**: Senior Data Scientist & Inventory Optimization Reviewer  

---

## 1. Executive Summary

This document presents the audited, leak-free inventory optimization engine designed for the DataCo Smart Supply Chain Core 9 product cohort. 

Following an audit of the initial Phase 9 implementation, two critical methodological corrections were applied:
1. **Fixed-Origin Recursive Multi-Step Forecasting**: Rather than extracting slices from the rolling 1-day-ahead benchmark (which consumed actual demand from day $t$ to forecast day $t+1$), the inventory engine now generates true fixed-origin multi-step recursive forecasts starting strictly at decision origin $\tau = \text{2017-08-31}$ across an 8-day forward horizon (`2017-09-01` through `2017-09-08`). In recursive rollout, predictions from step $h-1$ are fed forward as autoregressive lag and rolling inputs, guaranteeing that **zero future actual demand** is ever accessed.
2. **Pre-Decision Residual Uncertainty Estimation**: Forecast error uncertainty ($\sigma_e$) is now estimated strictly from out-of-sample residuals observed **on or before the August 31, 2017 decision date**. Residuals from the pre-decision validation window (`2017-07-01` to `2017-08-31`, 62 observations per product) are used. No September 2017 test actuals or errors enter the August 31 inventory decision.

---

## 2. Rolling One-Day-Ahead Benchmark vs. Fixed-Origin Multi-Step Forecasting

Understanding the distinction between model benchmarking and operational replenishment planning is essential:

```
Timeline:
------------------ Train -----------------|---- Validation ----|------------- Test -------------
2015-01-01                     2017-06-30 | 2017-07-01  2017-08-31 | 2017-09-01        2017-10-02
                                          |                    |
                                          |           Decision Origin (tau = 2017-08-31)
                                          |                    |
                                          |             Fixed Horizon (8 days: 2017-09-01 to 2017-09-08)
```

| Dimension | Rolling 1-Day-Ahead Benchmark (`dataco_forecast_comparison.parquet`) | Fixed-Origin Multi-Step Recursive Forecast (`dataco_fixed_origin_forecasts.parquet`) |
| :--- | :--- | :--- |
| **Operational Purpose** | Model accuracy benchmarking under daily observed updates. | Replenishment ordering across an unobserved multi-day supplier lead time. |
| **Forecast Origin** | Shifts every calendar day $t$ ($t \in \text{Test Window}$). | **Fixed at decision date $\tau = \text{2017-08-31}$**. |
| **Input for Lag 1 at Day $t+2$** | Actual historical demand observed on day $t+1$. | **Recursive model prediction $\hat{y}_{t+1}$** generated at step 1. |
| **Future Actual Demand Access** | Consumes ground-truth actuals from previous test days. | **Strictly ZERO access to actual demand after August 31, 2017**. |
| **Artifact Location** | `outputs/dataco/forecasts/dataco_forecast_comparison.parquet` | `outputs/dataco/inventory/dataco_fixed_origin_forecasts.parquet` |

### 2.1 How Recursive Multi-Step Forecasting Works
For each product $p$ and target step $h \in \{1, 2, \dots, H\}$ (where $H = L + R = 8\text{ days}$):
1. **Target Date**: $t_{\text{target}} = \tau + h\text{ days}$.
2. **Feature Evaluation**:
   - Calendar features: `day_of_week`, `month`, `day_of_month`, `is_weekend` evaluated for $t_{\text{target}}$.
   - Autoregressive Lags ($k \in \{1, 7, 14, 28\}$):
     $$\text{lag}_k(t_{\text{target}}) = \begin{cases} y(t_{\text{target}} - k), & \text{if } t_{\text{target}} - k \le \tau \text{ (historical actual)} \\ \hat{y}(t_{\text{target}} - k), & \text{if } t_{\text{target}} - k > \tau \text{ (recursive model prediction)} \end{cases}$$
   - Shifted Rolling Statistics (windows $W \in \{7, 14, 28\}$):
     Computed across the sequence of preceding demands $\{d(t_{\text{target}} - W), \dots, d(t_{\text{target}} - 1)\}$, using actual demand for dates $\le \tau$ and previously predicted demand for dates $> \tau$.
3. **Model Prediction**: $\hat{y}(t_{\text{target}}) = \max(0.0, \text{RF.predict}(\mathbf{x}))$.
4. **Recursive Cache**: Store $\hat{y}(t_{\text{target}})$ into the product's time series map to serve as autoregressive inputs for steps $h+1 \dots H$.

---

## 3. Pre-Decision Forecast Uncertainty Estimation

### 3.1 Residual Date Cutoff & Observation Window
To determine the safety stock buffer at origin $\tau = \text{2017-08-31}$, forecast error uncertainty ($\sigma_e$) must only utilize residuals that would have been observed on or before August 31, 2017.

- **Model Used**: Random Forest fitted on historical training data (`2015-01-01` through `2017-06-30`).
- **Residual Observation Period**: Out-of-sample validation window (`2017-07-01` through `2017-08-31`).
- **Sample Size ($N_p$)**: Exactly 62 daily out-of-sample residual observations per product ($N_p = 62$).
- **Leakage Prevention**: All observations with date $> \text{2017-08-31}$ are strictly excluded.

### 3.2 Uncertainty Formulas: Dispersion ($s_e$) vs. Bias & RMSE
Daily forecast residuals are defined as $e_{p, t} = y_{p, t} - \hat{y}_{p, t}$.

1. **Sample Standard Deviation ($s_e$, Dispersion around mean error $\bar{e}$)**:
   $$s_e = \sqrt{\frac{1}{N - 1} \sum_{t=1}^{N} (e_t - \bar{e})^2}$$
   Measures the empirical variability/dispersion of forecast errors.
2. **Root Mean Squared Error ($\text{RMSE}$, Expected squared deviation from actuals)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} e_t^2} = \sqrt{\frac{N-1}{N} s_e^2 + \bar{e}^2}$$
   Combines residual dispersion and systematic bias $\bar{e}$.

In safety stock sizing under unbiased or near-unbiased forecasts, sample standard deviation $s_e$ serves as the primary uncertainty measure. For the Core 9 cohort, mean validation errors are small ($|\bar{e}| \in [0.31, 2.48]$), resulting in close alignment between $s_e$ and RMSE.

### 3.3 Core 9 Pre-Decision Validation Error Summary (62 Days per SKU)

| SKU | Product Name | Sample Size ($N$) | Mean Error ($\bar{e}$) | Error Std ($s_e$) | Error RMSE | Uncertainty Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `1004` | Field & Stream 16 Gun Safe | 62 | +0.75 | **3.7232** | 3.7686 | Reliable Sample |
| `1014` | O'Brien Neoprene Life Vest | 62 | -2.49 | **14.8727** | 14.9601 | Reliable Sample |
| `1073` | Pelican Sunstream 100 Kayak| 62 | -0.31 | **5.1027** | 5.0709 | Reliable Sample |
| `191` | Nike Men's Free 5.0+ Shoe | 62 | +1.83 | **13.1555** | 13.1763 | Reliable Sample |
| `365` | Perfect Fitness Rip Deck | 62 | +1.40 | **16.5449** | 16.4701 | Reliable Sample |
| `403` | Nike Men's CJ Elite 2 Cleat | 62 | +0.46 | **4.8496** | 4.8318 | Reliable Sample |
| `502` | Nike Men's Victory Golf Polo| 62 | -0.33 | **13.7608** | 13.6534 | Reliable Sample |
| `627` | UA Girls' Toddler Surge Shoe| 62 | -0.64 | **10.4344** | 10.3697 | Reliable Sample |
| `957` | Diamondback Serene Bike | 62 | -0.72 | **3.9019** | 3.9375 | Reliable Sample |

*Fallback Policy*: If a product has $0 < N < 14$, catalog-pooled error standard deviation across all pre-decision validation records is applied with an explicit `is_fallback: True` indicator. If a product has $N = 0$, the engine returns `status = "UNAVAILABLE"`, `uncertainty_value = NaN`, and refuses to invent precision.

---

## 4. Observed vs. Simulated Data Boundaries

| Supply Chain Parameter | Source in DataCo Dataset | Operational Nature in Engine | Technical Reason & Operational Impact |
| :--- | :---: | :---: | :--- |
| **Customer Order Timestamps** | `order date (DateOrders)` | **Observed** | Empirical transaction record of customer demand. |
| **Customer Order Quantities** | `Order Item Quantity` ($1 - 5$) | **Observed** | Aggregated into daily gross retail demand. |
| **Outbound Carrier Shipping Days** | `Days for shipping (real)` ($0 - 6$) | **Observed** | Downstream delivery to consumer; **NOT supplier lead time**. |
| **Inbound Supplier Lead Time ($L$)** | **Absent** | **Simulated** | Factory manufacturing and dock-to-stock transit are unrecorded. Default assumed $L = 7\text{ days}$. |
| **Warehouse Stock-on-Hand ($I_{\text{on-hand}}$)** | **Absent** | **Simulated** | Warehouse bin counts are unrecorded. Baseline assumed $50\text{ units}$. |
| **In-Transit Supplier Orders ($I_{\text{on-order}}$)** | **Absent** | **Simulated** | Upstream purchase orders are unrecorded. Baseline assumed $0\text{ units}$. |
| **Customer Backorders ($B_{\text{backorders}}$)** | **Absent** | **Simulated** | Unfulfilled queued demand is unrecorded. Baseline assumed $0\text{ units}$. |
| **Order Setup Cost ($K$)** | **Absent** | **Simulated** | Administrative procurement parameter assumed $\$50.00/\text{PO}$. |
| **Annual Unit Holding Cost ($h$)** | **Absent** | **Simulated** | Inventory capital carrying cost assumed $\$2.00/\text{unit/year}$. |
| **True Historical Stockouts** | **Absent** | **Unrecorded** | Unrecorded lost sales cannot be reconstructed. |

---

## 5. Mathematical Inventory Formulas

1. **Net Inventory Position ($IP$)**:
   $$IP = I_{\text{on-hand}} + I_{\text{on-order}} - B_{\text{backorders}}$$
2. **Safety Stock ($SS$)**:
   - Lead time safety stock: $SS_{\text{lead}} = z \cdot \sigma_e \cdot \sqrt{L}$
   - Protection period safety stock: $SS_{\text{protection}} = z \cdot \sigma_e \cdot \sqrt{L + R}$
   - Where $z = \Phi^{-1}(\text{CSL}) \approx 1.6449$ for $\text{CSL} = 0.95$.
3. **Reorder Point ($ROP$)**:
   $$ROP = D_L + SS_{\text{lead}} = \sum_{h=1}^{L} \hat{y}_h + z \cdot \sigma_e \cdot \sqrt{L}$$
4. **Target Stock Level ($S$, Order-Up-To Level)**:
   $$S = D_{L+R} + SS_{\text{protection}} = \sum_{h=1}^{L+R} \hat{y}_h + z \cdot \sigma_e \cdot \sqrt{L + R}$$
5. **Suggested Replenishment Quantity ($Q$)**:
   $$Q = \begin{cases} \lceil S - IP \rceil, & \text{if } IP \le ROP \\ 0, & \text{if } IP > ROP \end{cases}$$
6. **Optional Economic Order Quantity ($EOQ$) Benchmark**:
   $$EOQ = \sqrt{\frac{2 \cdot D_{\text{annual}} \cdot K}{h}}, \quad D_{\text{annual}} = 365 \cdot \frac{D_L}{L}$$

---

## 6. Generated Inventory Recommendations (Core 9 Cohort)

Generated using fixed-origin recursive forecasts ($\tau = \text{2017-08-31}$) and pre-decision validation uncertainty under policy: $\text{CSL} = 95\%$, $L = 7\text{ days}$, $R = 1\text{ day}$, $\text{Initial On-Hand} = 50\text{ units}$:

| SKU | Product Name | LTD (7d) | PPD (8d) | Daily $\sigma_e$ | Safety Stock | ROP | Target Stock ($S$) | Net $IP$ | Order Qty ($Q$) | Status | EOQ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1004` | Field & Stream 16 Gun Safe | 115.43 | 131.93 | 3.72 | **16.20** | **131.63** | 149.25 | 50.0 | **100** | REORDER | 548.6 |
| `1014` | O'Brien Neoprene Life Vest | 386.89 | 438.57 | 14.87 | **64.72** | **451.61** | 507.76 | 50.0 | **458** | REORDER | 1,004.3 |
| `1073` | Pelican Sunstream 100 Kayak| 110.48 | 127.30 | 5.10 | **22.21** | **132.69** | 151.04 | 50.0 | **102** | REORDER | 536.7 |
| `191` | Nike Men's Free 5.0+ Shoe | 222.36 | 254.77 | 13.16 | **57.25** | **279.61** | 315.97 | 50.0 | **266** | REORDER | 761.4 |
| `365` | Perfect Fitness Rip Deck | 484.79 | 560.46 | 16.54 | **72.00** | **556.79** | 637.43 | 50.0 | **588** | REORDER | 1,124.2 |
| `403` | Nike Men's CJ Elite 2 Cleat | 151.92 | 173.05 | 4.85 | **21.10** | **173.02** | 195.61 | 50.0 | **146** | REORDER | 629.3 |
| `502` | Nike Men's Victory Golf Polo| 395.77 | 447.39 | 13.76 | **59.89** | **455.66** | 511.41 | 50.0 | **462** | REORDER | 1,015.8 |
| `627` | UA Girls' Toddler Surge Shoe| 218.93 | 252.18 | 10.43 | **45.41** | **264.34** | 300.72 | 50.0 | **251** | REORDER | 755.5 |
| `957` | Diamondback Serene Bike | 96.77 | 108.41 | 3.90 | **16.98** | **113.75** | 126.56 | 50.0 | **77** | REORDER | 502.3 |

### 6.1 Updated Worked Example: `Product 365` (Perfect Fitness Perfect Rip Deck)
1. **Forecast Horizon Steps**:
   - $h=1$ (`2017-09-01`): $\hat{y}_1 = 67.42$
   - $h=2$ (`2017-09-02`): $\hat{y}_2 = 65.78$
   - $h=3$ (`2017-09-03`): $\hat{y}_3 = 69.65$
   - $h=4$ (`2017-09-04`): $\hat{y}_4 = 71.13$
   - $h=5$ (`2017-09-05`): $\hat{y}_5 = 67.29$
   - $h=6$ (`2017-09-06`): $\hat{y}_6 = 71.11$
   - $h=7$ (`2017-09-07`): $\hat{y}_7 = 72.41$
   - $h=8$ (`2017-09-08`): $\hat{y}_8 = 75.67$
2. **Expected Demands**:
   - $\text{Expected Lead-Time Demand } D_L = \sum_{h=1}^7 \hat{y}_h = 484.79\text{ units}$
   - $\text{Expected Protection-Period Demand } D_{L+R} = \sum_{h=1}^8 \hat{y}_h = 560.46\text{ units}$
3. **Safety Stock Buffer**:
   - Daily residual standard deviation from validation period: $\sigma_e = 16.5449\text{ units/day}$
   - Quantile: $z = \Phi^{-1}(0.95) \approx 1.64485$
   - Lead time buffer: $SS_{\text{lead}} = 1.64485 \times 16.5449 \times \sqrt{7} = 72.00\text{ units}$
   - Protection period buffer: $SS_{\text{protection}} = 1.64485 \times 16.5449 \times \sqrt{8} = 76.98\text{ units}$
4. **Decision Thresholds**:
   - $\text{Reorder Point } ROP = 484.79 + 72.00 = 556.79\text{ units}$
   - $\text{Target Stock Level } S = 560.46 + 76.98 = 637.43\text{ units}$
5. **Replenishment Decision**:
   - Current net position: $IP = 50.00\text{ units}$
   - Check condition: $IP = 50.00 \le ROP = 556.79 \implies$ **REORDER TRIGGERED**
   - Order quantity: $Q = \lceil 637.43 - 50.00 \rceil = \mathbf{588\text{ units}}$

---

## 7. Verification and Test Results

Executed complete test suite:
- **DataCo inventory audit & correction tests (`tests/test_inventory_dataco.py`)**: 11 passed out of 11.
- **DataCo benchmark tests (`tests/test_benchmark_dataco.py`)**: 9 passed out of 9.
- **DataCo validation tests (`tests/test_dataco_validation.py`)**: 22 passed out of 22.
- **Full repository test suite**: **200 passed, 1 skipped, 0 failures**. Zero regressions on UCI Online Retail.

---

## 8. Remaining Limitations & Boundary Disclaimers

1. **Unrecorded Inbound Supply Chain**: DataCo provides no supplier orders, factory lead times, purchase agreements, or freight shipping logs. Inbound replenishment lead times are assumed.
2. **Censored Historical Demand**: DataCo records placed customer transactions. If real stockouts occurred, unmet customer demand was lost and never captured in the logs.
3. **Multi-Step Recursive Error Accumulation**: In recursive multi-step forecasting, forecast errors at step 1 can compound into subsequent steps. Normal quantile approximation assumes error variance scales with $\sqrt{L}$.
4. **Decision-Support vs. Production Orders**: All generated recommendations are decision-support planning benchmarks for policy simulation, not actual operational purchase orders.
