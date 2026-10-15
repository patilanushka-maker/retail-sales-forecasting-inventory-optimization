# DataCo Smart Supply Chain Dashboard Integration (Phase 10)

## Overview

In Phase 10, the **DataCo Smart Supply Chain** dataset was integrated into the existing interactive Streamlit dashboard (`app/streamlit_app.py`) alongside the established **UCI Online Retail** pipeline. The two datasets remain completely isolated: every metric, chart, model comparison, SKU explorer, and inventory recommendation is scoped strictly to the selected dataset.

---

## 🚀 How to Launch the Dashboard Locally

Ensure your Python virtual environment is activated and dependencies are installed:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

From the repository root, run:

```bash
streamlit run app/streamlit_app.py
```

The application will start and open in your default browser at `http://localhost:8501`.

---

## 🔀 Dataset Selection

The sidebar contains a dedicated **Dataset Selector**:
- **UCI Online Retail** (UK giftware e-commerce, 2010–2011, 2,813 products, 68,907 test observations, 28-day rolling evaluation).
- **DataCo Smart Supply Chain** (US/Latin America multi-channel supply chain, 2015–2017, 118 catalog products, Core 9 high-volume forecasting cohort, 8-day fixed-origin inventory horizon).

Switching between datasets immediately reconfigures navigation, metrics, titles, filters, and visualizations. No data from UCI is displayed when DataCo is active, and vice versa.

---

## 📊 DataCo Dashboard Pages & Features

When **DataCo Smart Supply Chain** is selected, five dedicated views are available:

### 1. 🌐 Overview (`app/views/dataco_overview.py`)
- **Key Metrics**: Date coverage (2015-01-01 to 2018-01-31, 1,127 calendar days), raw transactions (180,519 order-lines across 65,752 distinct order IDs), included transactions under demand policy (172,765 rows across 62,897 distinct order IDs), product catalog size (118 SKUs), and total historical gross placed retail demand (367,591 physical units across 21,787 daily product-date records).
- **Core 9 Benchmark Summary**: Highlights the 9 highest-volume products analyzed in the forecasting benchmark (159,183 units; 43.3% of total portfolio demand).
- **Demand Definition Guardrail**: Explains that target demand is *gross placed retail demand* (ordered units aggregated by order creation date) rather than fulfilled or dispatched sales.
- **Simulated Inventory Banner**: Explains that inventory levels, review periods, supplier lead times, and unit costs are simulated demonstration parameters.
- **Workflow Architecture**: Interactive mermaid diagram tracing raw data $\to$ preprocessing $\to$ forecasting benchmark $\to$ inventory optimization $\to$ dashboard views.

### 2. 📈 Sales Analytics (`app/views/dataco_sales_analytics.py`)
- **Portfolio Daily Trend**: 1,127-day time series chart showing daily gross placed retail demand alongside a 7-day rolling moving average.
- **Interactive Date Filter**: Allows slicing the historical demand time-series by start and end date (defaulting to the full span 2015-01-01 to 2018-01-31).
- **Department Breakdown**: Horizontal bar chart comparing demand distribution across departments (Fan Shop, Apparel, Golf, Footwear, Outdoors, Fitness).
- **Top Products Table**: Tabular summary of top-selling products by total units, daily mean demand, and standard deviation.
- **Single Product Demand Inspector**: Interactive dropdown to inspect the individual daily time-series profile of any catalog SKU.

### 3. 🎯 Forecast Performance (`app/views/dataco_forecast_performance.py`)
- **Model Scorecard**: Comparative performance table across Seasonal Naive ($t-7$), 7-Day Rolling Mean, and Random Forest Regressor evaluated out-of-sample on the 32-day test period (2017-09-01 to 2017-10-02; 288 observations).
- **Metric Definitions**:
  - $\text{MAE} = \frac{1}{N} \sum |y - \hat{y}|$: Mean Absolute Error in physical units.
  - $\text{RMSE} = \sqrt{\frac{1}{N} \sum (y - \hat{y})^2}$: Root Mean Squared Error, penalizing large outlier errors.
  - $\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$: Weighted Absolute Percentage Error, providing scale-independent portfolio error.
- **Visual Error Benchmark**: Grouped bar chart comparing MAE, RMSE, and WAPE across candidate models.
- **Per-Product Accuracy Table**: Granular breakdown of error metrics for each of the Core 9 benchmark products.
- **Actual vs Forecast Visualizer**: Overlay of test-period historical actuals against model predictions for any selected product.
- **Evaluation Protocol Note**: Distinguishes the 32-day rolling 1-day-ahead benchmark from fixed-origin multi-step forecasting.

### 4. 📦 Product Catalog & Explorer (`app/views/dataco_sku_explorer.py`)
- **118-Product Catalog**: Full catalog table showing Product ID, Name, Category, Department, Price, and Core 9 Cohort Status.
- **Search & Department Filtering**: Fast text search and departmental dropdown filters.
- **SKU Demand Deep-Dive**: Deep dive into individual product demand statistics (min, max, mean, zero-demand days, demand distribution).
- **Privacy & PII Protection**: ZERO customer personal information is included. Customer names, emails, street addresses, zip codes, and individual customer IDs are strictly omitted.

### 5. 🛡️ Inventory Optimization & ROP (`app/views/dataco_inventory_optimization.py`)
- **Fixed-Origin 8-Day Forecast Chart**: Multi-step recursive forecast horizon visualization ($h=1\dots 8$, origin strictly at $T_{\text{origin}} = 2017\text{-}08\text{-}31$, forecast dates 2017-09-01 through 2017-09-08) with visual boundary between lead time ($L=7$ days) and review period ($R=1$ day).
- **Daily Forecast Horizon Table**: Tabular breakdown of day-by-day predicted demand for each day in the protection window.
- **Core 9 Recommendation Table**: Product-level inventory recommendations with target stock levels ($S$), reorder points ($\text{ROP}$), safety stock ($\text{SS}$), simulated on-hand inventory ($I_t$), inventory positions, and suggested order quantities ($Q^*$).
- **Replenishment Orders CSV Export**: Downloadable purchase order list for items where inventory position falls below ROP.
- **Unavailable State Handling**: If fixed forecasts or inventory artifacts are missing, the view displays an explicit unavailable warning instead of silently imputing zeroes.

---

## 🔬 Methodological Differences & Safeguards

### Rolling 1-Day Benchmark vs. Fixed-Origin Multi-Step Forecasting
- **Rolling 1-Day-Ahead Evaluation**: Used in the benchmark (`outputs/dataco/forecasts/dataco_forecast_comparison.parquet`). Each day $t$, the models observe all actual demand history up to $t-1$ to predict day $t$. This evaluates rolling one-step predictive accuracy over the 32-day test period (Sept 1 – Oct 2, 2017).
- **Fixed-Origin 8-Day Forecasting**: Used in inventory optimization (`outputs/dataco/inventory/dataco_fixed_origin_forecasts.parquet`). The forecast origin is strictly fixed at $T_{\text{origin}} = 2017\text{-}08\text{-}31$. The model recursively forecasts all 8 days of the protection period (2017-09-01 through 2017-09-08; $L=7$ days lead time $+ R=1$ day review period) into the future without observing intermediate ground-truth demand. Residual uncertainty ($\sigma_e$) is derived strictly from pre-decision validation errors (2017-07-01 to 2017-08-31).

### Demand Definition Limitations
- **Gross Placed Retail Demand**: The target demand represents units placed by customers on order creation dates (`order_date`), excluding canceled orders.
- **Unfulfilled vs Fulfilled**: Because the dataset does not record backorders, stockout duration, lost sales, or true warehouse inventory availability, historical transactions reflect *placed orders*, which may not equal fulfilled demand or unconstrained customer desire.

### Simulated Inventory Parameters
- DataCo provides multi-channel shipping status and order transactions, but **no supplier replenishment records, internal warehouse stock levels, purchase costs, holding cost percentages, or actual vendor lead times**.
- All inventory parameters ($L=7$ days, $R=1$ day, SL=95%, holding cost=20%/yr, ordering cost=\$50, simulated on-hand inventory) are **demonstration parameters** clearly designated as operational simulations.

---

## 🧪 Verification & Automated Testing

The dashboard integration is covered by automated unit and integration tests:
- `tests/test_dataco_dashboard.py`:
  - `test_dataco_loaders_load_valid_data`: Verifies all cached DataCo loading functions return non-empty, expected structures.
  - `test_dataco_catalog_no_pii_exposed`: Verifies no customer PII columns (customer names, emails, addresses, passwords) exist in catalog data.
  - `test_dataco_missing_files_graceful_handling`: Asserts that missing artifact paths return empty DataFrames/dicts with warning logs rather than unhandled exceptions.
  - `test_dataco_plot_functions_return_figures`: Validates that portfolio, department, fixed-horizon, and model-comparison plotting functions return valid `plotly.graph_objects.Figure` instances.
  - `test_apptest_navigation_all_dataco_pages`: Uses `streamlit.testing.v1.AppTest` to mount the live app, switch to DataCo, navigate through all 5 pages, and assert error-free rendering.
  - `test_apptest_navigation_all_uci_pages`: Navigates through all 6 UCI pages to guarantee complete regression-free operation of the original pipeline.
- **Suite-wide Status**: 212 tests passed, 1 skipped (optional openpyxl test).
