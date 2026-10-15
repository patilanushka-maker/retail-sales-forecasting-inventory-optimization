# Forecasting & Inventory Optimization Methodology

> **Overview**: Comprehensive mathematical and operational formulation behind demand forecasting, feature engineering, and inventory policies.

---

## 1. Time-Series Data Preprocessing & Leakage Prevention

1. **Daily Aggregation**: Retail transactions are aggregated to $(store, item, date)$ triplets. 
2. **Zero-Filling**: Dates where a store was open but recorded zero sales for an item are explicitly imputed with 0 to prevent artificial demand spikes.
3. **Temporal Partitioning**:
   - An expanding temporal split is employed. The final 28 days are withheld as the strictly out-of-sample evaluation period.
   - Shuffled cross-validation is strictly forbidden to preserve temporal ordering and avoid lookahead bias.

---

## 2. Feature Engineering Mechanics

All rolling statistics and aggregations are strictly shifted by at least 1 day ($t-1$):
- **Lag Features**:
  $$\text{lag}_k(t) = y(t - k), \quad k \in \{1, 7, 14, 28\}$$
- **Shifted Rolling Window**:
  $$\text{rolling\_mean}_w(t) = \frac{1}{w} \sum_{i=1}^{w} y(t - i)$$
- **Temporal Encodings**: Calendar attributes capture regular weekly and seasonal retail buying cycles.

---

## 3. Modeling Strategy

### Baseline Model: Seasonal Naive Forecaster
Acts as the baseline standard. Projects demand by replicating the value from the matching day in the preceding weekly cycle:
$$\hat{y}(t + h) = y(t + h - m \cdot \lfloor h/m + 1 \rfloor), \quad m=7$$

### Machine Learning Model: Random Forest Regressor
- Non-linear ensemble of decision trees capturing interactions across lags, rolling statistics, and seasonal factors.
- Evaluated against the baseline using MAE, RMSE, and WAPE.

---

## 4. Inventory Replenishment Logic

1. **Cycle Service Level & Safety Stock**:
   $$SS = Z_{\alpha} \times \sqrt{L \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_L^2}$$
2. **Reorder Point (ROP)**:
   $$ROP = (\bar{d} \times L) + SS$$
3. **Economic Order Quantity (EOQ)**:
   $$EOQ = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$
4. **Replenishment Trigger**:
   An order of size $EOQ$ (or up-to-order-level quantity) is triggered whenever:
   $$\text{Inventory Position} = (\text{On-Hand} + \text{On-Order} - \text{Backorders}) \le ROP$$
