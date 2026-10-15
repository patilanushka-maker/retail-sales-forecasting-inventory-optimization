# DataCo Demand Definition & Time-Series Readiness Audit Report

> **Audit Type**: Read-Only Senior Data Science & Engineering Audit  
> **Target Dataset**: DataCo Smart Supply Chain (`data/raw/DataCoSupplyChainDataset.csv`)  
> **Current Processed Artifacts**:  
> - `data/processed/dataco_daily_product_demand.parquet` (21,787 records across 118 SKUs)  
> - `data/processed/dataco_product_catalog.parquet` (118 SKUs)  
> **Review Date**: October 15, 2026  
> **Reviewer**: Senior Data Scientist & Data Engineering Reviewer  

---

## 1. Executive Summary

This read-only audit evaluates the operational semantics, policy sensitivities, and time-series structure of the DataCo dataset prior to feature engineering or model training.

### Key Conclusions:
1. **Demand Semantics Clarification**: The current baseline policy (excluding `CANCELED` and `SUSPECTED_FRAUD`) measures **Gross Placed Order Demand** (orders placed by customers for which fulfillment was intended). It does **not** represent verified delivered/realized sales, because intermediate statuses (`PENDING`, `PENDING_PAYMENT`, `ON_HOLD`, `PROCESSING`) reflect settlement workflows.
2. **Coupled Payment Mechanism**: An unpublicized architectural pattern was discovered: **`Order Status` is deterministically partitioned by payment method (`Type`)**. For example, 100% of `CLOSED` orders were paid via `CASH`, 100% of `COMPLETE` orders via `DEBIT`, and 100% of `PENDING_PAYMENT` via `PAYMENT`. Excluding `PENDING_PAYMENT` or `PENDING` unintentionally discards entire payment channels rather than unfulfilled orders.
3. **Severe Catalog Sparsity & Power-Law Concentration**:
   - The top **9 SKUs (7.6% of catalog) account for 86.4% of all demand volume**. These 9 SKUs have dense, continuous 1,006-day selling histories with **0% missing days**.
   - The remaining **109 SKUs are highly intermittent and sparse** (median sparsity: 70.1% missing days in active lifespan).
4. **Time-Series Readiness Blockers**: The current `dataco_daily_product_demand.parquet` file contains only observed sale days. **It cannot be directly fed into autoregressive lag pipelines (e.g. `lag_1`, `lag_7`, `rolling_mean_7`) without explicit, calendar-aligned zero-filling within active product lifespans.**

---

## 2. Part 1 — Order-Status Semantics & Cross-Tabulation Audit

### 2.1 Complete Frequency Distribution
All 180,519 order-line rows map into 9 distinct `Order Status` categories:

| Order Status | Row Count | Total Quantity | Row % | Qty % | Delivery Status Observed | Baseline Handling |
| :--- | :- | :- | :- | :- | :--- | :---: |
| `COMPLETE` | 59,491 | 126,624 | 32.96% | 32.97% | Advance (23.8%) / Late (57.5%) / On time (18.8%) | **INCLUDED** |
| `PENDING_PAYMENT` | 39,832 | 84,736 | 22.07% | 22.06% | Advance (24.1%) / Late (57.5%) / On time (18.4%) | **INCLUDED** |
| `PROCESSING` | 21,902 | 46,623 | 12.13% | 12.14% | Advance (24.5%) / Late (57.1%) / On time (18.4%) | **INCLUDED** |
| `PENDING` | 20,227 | 43,196 | 11.20% | 11.25% | Advance (24.0%) / Late (57.9%) / On time (18.1%) | **INCLUDED** |
| `CLOSED` | 19,616 | 41,458 | 10.87% | 10.79% | Advance (24.5%) / Late (56.6%) / On time (18.9%) | **INCLUDED** |
| `ON_HOLD` | 9,804 | 20,845 | 5.43% | 5.43% | Advance (24.6%) / Late (55.6%) / On time (19.8%) | **INCLUDED** |
| `SUSPECTED_FRAUD` | 4,062 | 8,613 | 2.25% | 2.24% | **Shipping canceled (100.0%)** | **EXCLUDED** |
| `CANCELED` | 3,692 | 7,875 | 2.05% | 2.05% | **Shipping canceled (100.0%)** | **EXCLUDED** |
| `PAYMENT_REVIEW` | 1,893 | 4,109 | 1.05% | 1.07% | Advance (22.2%) / Late (57.2%) / On time (20.7%) | **INCLUDED** |
| **Total** | **180,519** | **384,079** | **100.0%** | **100.0%** | | **7,754 Excluded (4.3%)** |

### 2.2 The Payment Method Coupling Finding
A cross-tabulation of `Order Status` against payment `Type` demonstrates that order statuses in DataCo reflect billing gateways rather than lifecycle progress:

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Order Status     │   CASH   │  DEBIT   │ PAYMENT  │ TRANSFER │  TOTAL   │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ CANCELED         │        0 │        0 │        0 │    3,692 │    3,692 │
│ CLOSED           │   19,616 │        0 │        0 │        0 │   19,616 │
│ COMPLETE         │        0 │   59,491 │        0 │        0 │   59,491 │
│ ON_HOLD          │        0 │    9,804 │        0 │        0 │    9,804 │
│ PAYMENT_REVIEW   │        0 │        0 │    1,893 │        0 │    1,893 │
│ PENDING          │        0 │        0 │        0 │   20,227 │   20,227 │
│ PENDING_PAYMENT  │        0 │        0 │   39,832 │        0 │   39,832 │
│ PROCESSING       │        0 │        0 │        0 │   21,902 │   21,902 │
│ SUSPECTED_FRAUD  │        0 │        0 │        0 │    4,062 │    4,062 │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ TOTAL            │   19,616 │   69,295 │   41,725 │   49,883 │  180,519 │
└──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

**Crucial Insight**:
- `CLOSED` was used exclusively for **CASH** payments that reached final completion.
- `COMPLETE` was used exclusively for **DEBIT** payments that settled.
- `PENDING_PAYMENT` was used for orders placed via credit/online **PAYMENT** gateways.
- `PENDING` and `PROCESSING` were used for **WIRE TRANSFER** orders.
- Furthermore, **all 7 included statuses have identical shipping delay distributions** (~57% late delivery, ~24% advance, ~19% on time).
- Therefore, discarding `PENDING_PAYMENT` or `PENDING` does **not** filter out unfulfilled orders; it biases the demand dataset by discarding entire customer payment segments.

### 2.3 What the Data Can vs. Cannot Establish
* **Can Establish**:
  - Exact customer order timestamps and placed item quantities.
  - Exactly which orders were canceled or flagged as fraud before shipping (`Delivery Status == 'Shipping canceled'`).
  - Courier transit duration (`Days for shipping (real)`).
* **Cannot Establish**:
  - Whether a customer initiated a return after receiving delivery (no RMA timestamps).
  - Whether zero demand on a day was caused by a warehouse stockout vs. absence of customer interest.
  - Upstream supplier purchase order dates or lead times.

---

## 3. Part 2 — Quantitative Sensitivity of Alternative Demand Definitions

To assess the impact of demand definition on volume and coverage, four alternative policies were computed:

| Policy Name | Included Statuses | Included Rows (%) | Included Units (%) | Daily Demand Records | SKUs Covered | Date Span |
| :--- | :--- | :- | :- | :- | :- | :--- |
| **Policy A (Current Baseline: Gross Placed Demand)** | `COMPLETE`, `PENDING_PAYMENT`, `PROCESSING`, `PENDING`, `CLOSED`, `ON_HOLD`, `PAYMENT_REVIEW` | 172,765 (**95.70%**) | 367,591 (**95.71%**) | 21,787 | 118 | 2015-01-01 to 2018-01-31 |
| **Policy B (Conservative Confirmed: COMPLETE & CLOSED)** | `COMPLETE`, `CLOSED` | 79,107 (**43.82%**) | 168,082 (**43.76%**) | 15,497 | 118 | 2015-01-01 to 2018-01-31 |
| **Policy C (Strictly Complete Only)** | `COMPLETE` | 59,491 (**32.96%**) | 126,624 (**32.97%**) | 14,028 | 118 | 2015-01-01 to 2018-01-31 |
| **Policy D (Settled & In-Flight: COMPLETE, CLOSED, PROCESSING)** | `COMPLETE`, `CLOSED`, `PROCESSING` | 101,009 (**55.95%**) | 214,705 (**55.90%**) | 17,142 | 118 | 2015-01-01 to 2018-01-31 |

### Policy Sensitivity Analysis
1. Moving from Policy A to Policy B cuts total demand volume by **56.2%** (loss of 199,509 units).
2. However, because `Order Status` is tightly linked to payment `Type`, Policy B effectively discards **all wire transfers and credit card payments**, keeping only cash and debit transactions.
3. **Recommendation**: Retain **Policy A** for primary retail demand modeling, but explicitly document its definition as **Gross Placed Retail Demand** (excluding canceled and fraudulent orders).

---

## 4. Part 3 — Daily Time-Series Readiness Audit

### 4.1 Per-Product Lifespan and Sparsity Profile
Analysis of `dataco_daily_product_demand.parquet` across all 118 SKUs:

| Metric | Minimum | 25th Percentile | Median (50th) | Mean | 75th Percentile | Maximum |
| :--- | :- | :- | :- | :- | :- | :- |
| **Observed Sale Days** | 3 days | 18.25 days | 56.0 days | 184.6 days | 241.0 days | 1,006 days |
| **Lifespan Span Days** | 3 days | 75.5 days | 141.0 days | 445.5 days | 839.8 days | 1,006 days |
| **Missing Days in Span** | 0 days | 46.5 days | 87.5 days | 260.9 days | 593.0 days | 631 days |
| **Sparsity % (Missing / Span)**| 0.0% | 60.4% | 70.1% | 58.8% | 72.6% | 93.2% |
| **Total Product Volume** | 8 units | 72.5 units | 489.5 units | 3,115.2 units | 848.8 units | 70,575 units |

### 4.2 Extreme Power-Law Volume Concentration
A critical structural characteristic of DataCo is extreme Pareto concentration:

* **Top 1 SKU (`365` - Perfect Fitness Rip Deck)**: 70,575 units (**19.2% of total dataset demand**).
* **Top 5 SKUs**: 251,600 units (**68.4% of total dataset demand**).
* **Top 9 SKUs**: 317,444 units (**86.4% of total dataset demand**).
* **Top 15 SKUs**: 323,016 units (**87.9% of total dataset demand**).
* **Bottom 103 SKUs combined**: Only **13.6% of total demand**.

#### Top 9 Dense Core SKUs (The "Core 9")
These 9 products represent 86.4% of all transaction volume and exhibit continuous daily sales throughout the entire 3-year observation window:

| Product ID | Product Name | Department | Price | Observed Days | Calendar Span | Sparsity % | Total Volume |
| :---: | :--- | :--- | :- | :- | :- | :- | :- |
| `365` | Perfect Fitness Perfect Rip Deck | Apparel | $59.99 | 1,006 | 1,006 | **0.0%** | 70,575 |
| `502` | Nike Men's Dri-FIT Victory Golf Polo | Golf | $50.00 | 1,006 | 1,006 | **0.0%** | 60,232 |
| `1014`| O'Brien Men's Neoprene Life Vest | Fan Shop | $49.98 | 1,006 | 1,006 | **0.0%** | 55,333 |
| `191` | Nike Men's Free 5.0+ Running Shoe | Footwear | $99.99 | 1,006 | 1,006 | **0.0%** | 35,079 |
| `627` | Under Armour Girls' Toddler Spine Surge | Golf | $39.99 | 1,006 | 1,006 | **0.0%** | 30,380 |
| `403` | Nike Men's CJ Elite 2 TD Football Cleat | Apparel | $129.99 | 1,006 | 1,006 | **0.0%** | 21,263 |
| `1004`| Field & Stream Sportsman 16 Gun Safe | Fan Shop | $399.98 | 1,006 | 1,006 | **0.0%** | 16,595 |
| `1073`| Pelican Sunstream 100 Kayak | Fan Shop | $199.99 | 1,006 | 1,006 | **0.0%** | 14,840 |
| `957` | Diamondback Women's Serene Comfort Bike | Fan Shop | $299.98 | 1,006 | 1,006 | **0.0%** | 13,157 |

### 4.3 Why Raw Output Cannot Be Fed Directly to Lag Features
The current `dataco_daily_product_demand.parquet` contains **only days with non-zero sales** (21,787 rows).
- If `lag_1` is computed via `df.groupby('product_id')['demand'].shift(1)` without zero-filling, the lag value for an intermittent product will point to **the previous transaction date rather than calendar day $t-1$** (which could be 5, 20, or 60 days in the past).
- This introduces severe temporal distortion and invalidates autoregressive models.

---

## 5. Part 4 — Recommended Time-Series Construction Policy for Next Phase

To transition DataCo into a leak-free forecasting benchmark, the following 4-step policy is recommended:

```
[Raw Processed Active Sales] (21,787 rows)
            │
            ▼
[Step 1: Segment into Core vs. Long-Tail Cohorts]
  ├── Cohort 1: Core 9 SKUs (86.4% volume, 0% sparsity, continuous 1,006 days)
  └── Cohort 2: Long-Tail Intermittent SKUs (109 SKUs, 13.6% volume, 70% sparsity)
            │
            ▼
[Step 2: Product-Lifespan Calendar Reindexing (Zero-Filling)]
  ├── For each SKU, construct complete daily index: [first_sale_date, last_sale_date]
  └── Fill unobserved days with demand = 0.0 (strictly within active lifespan)
            │
            ▼
[Step 3: Minimum History Filtering for Autoregressive Feasibility]
  ├── Exclude products with span < 90 days (31 SKUs) from lag-based feature modeling
  └── Retain 54 products with span >= 365 days for seasonal evaluations
            │
            ▼
[Step 4: Chronological Train / Validation / Test Splits]
  ├── Train Window: 2015-01-01 to 2017-06-30 (2.5 years, 912 days)
  ├── Validation Window: 2017-07-01 to 2017-10-31 (4 months, 123 days)
  └── Test Window: 2017-11-01 to 2018-01-31 (3 months, 92 days)
```

---

## 6. Decisions Requiring Human Approval Before Model Training

1. **Demand Definition Approval**:
   * *Recommended*: Confirm **Policy A** (`Gross Placed Retail Demand`, excluding cancellations and fraud).
   * *Alternative*: Select **Policy B** (`Confirmed Completed Sales`), acknowledging that it eliminates 56.2% of transactions and discards credit card/wire transfer payment channels.
2. **Catalog Scope for Model Benchmarking**:
   * *Option 1 (Full Catalog Zero-Filled)*: Zero-fill all 118 SKUs across their respective lifespans. (Requires Croston/intermittent models for the 109 sparse SKUs).
   * *Option 2 (Core 9 Benchmark)*: Focus the initial multi-model comparison (Rolling Mean, Exponential Smoothing, LightGBM) on the 9 high-volume continuous SKUs representing 86.4% of total demand.
3. **Train / Test Chronological Cutoff**:
   * Approve the recommended out-of-sample test split: **Last 92 days (November 1, 2017 – January 31, 2018)**.
