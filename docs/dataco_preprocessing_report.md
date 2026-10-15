# DataCo Smart Supply Chain Dataset: Preprocessing & Demand Definition Report (Phase 8B)

> **Phase**: 8B — Data Preparation, Policy Definition, and Daily Aggregation  
> **Source Artifact**: `data/raw/DataCoSupplyChainDataset.csv`  
> **Output Artifacts**:  
> - `data/processed/dataco_daily_product_demand.parquet`  
> - `data/processed/dataco_product_catalog.parquet`  
> **Review Date**: October 15, 2026  
> **Reviewer**: Senior Data Scientist & Data Engineering Reviewer  

---

## 1. Executive Summary

Phase 8B establishes a clean, auditable, and scientifically justified daily product demand time series from the raw DataCo dataset. In accordance with strict engineering guidelines, the pipeline:
1. **Protects Existing Workflows**: Leaves the UCI Online Retail raw dataset, interim files, processed artifacts, and models 100% untouched.
2. **Eliminates Customer PII**: Completely excludes personal identifiers (customer names, emails, passwords, and street addresses) from all processed artifacts and reports.
3. **Applies an Explicit Demand Policy**: Differentiates true retail consumer demand from non-fulfilled transactions (cancellations and suspected fraud).
4. **Maintains Full Accounting Reconciliation**: Verifies that every single row and unit of quantity is rigorously reconciled between raw, excluded, included, and aggregated datasets.

---

## 2. Raw Source & Provenance Verification

| Parameter | Specification |
| :--- | :--- |
| **Raw File Path** | `data/raw/DataCoSupplyChainDataset.csv` |
| **SHA-256 Checksum** | `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6` |
| **Authoritative Source** | Mendeley Data (DOI: `10.17632/8gx2fvg2k6.5`) |
| **Original Authors** | Fabián Constante, Fernando Silva, António Pereira (2019) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Raw Dimensions** | 180,519 rows × 53 columns |
| **Total Raw Units Sold**| 384,079 units (`Order Item Quantity`) |

---

## 3. Demand-Inclusion Policy & Order Status Audit

### 3.1 Order Status Audit & Cross-Tabulation
Every row in DataCo contains an `Order Status` and a `Delivery Status`. Cross-tabulation reveals an exact 1-to-1 correspondence with canceled shipping:

| Order Status | Row Count | Total Quantity | Delivery Status Observed | Policy Decision | Justification |
| :--- | :- | :- | :--- | :---: | :--- |
| `COMPLETE` | 59,491 | 126,624 | Advance / Late / On time | **INCLUDED** | Confirmed customer order, payment settled and fulfilled. |
| `PENDING_PAYMENT`| 39,832 | 84,736 | Advance / Late / On time | **INCLUDED** | Valid placed order undergoing standard payment clearing. |
| `PROCESSING` | 21,902 | 46,623 | Advance / Late / On time | **INCLUDED** | Order accepted and actively packed in warehouse. |
| `PENDING` | 20,227 | 43,196 | Advance / Late / On time | **INCLUDED** | Valid customer demand awaiting allocation. |
| `CLOSED` | 19,616 | 41,458 | Advance / Late / On time | **INCLUDED** | Successfully completed and archived transaction. |
| `ON_HOLD` | 9,804 | 20,845 | Advance / Late / On time | **INCLUDED** | Valid placed order under administrative or stock hold. |
| `PAYMENT_REVIEW` | 1,893 | 4,109 | Advance / Late / On time | **INCLUDED** | Placed order undergoing merchant verification. |
| `SUSPECTED_FRAUD`| 4,062 | 8,613 | **Shipping canceled (100%)** | **EXCLUDED** | Fraudulent attempt; transaction voided, no merchandise shipped. |
| `CANCELED` | 3,692 | 7,875 | **Shipping canceled (100%)** | **EXCLUDED** | Order canceled prior to fulfillment; goods never dispatched. |
| **Total** | **180,519** | **384,079** | | | **7,754 rows excluded (4.3%)** |

### 3.2 Key Demand-Definition Principles
1. **Demand Date Selection:** Demand is assigned to the calendar date when the customer placed the order (`order date (DateOrders)` truncated to `YYYY-MM-DD`). We do **not** use `shipping date (DateOrders)` because shipping reflects internal warehouse dispatch latency, whereas demand models consumer purchasing propensity.
2. **Exclusion of Non-Fulfilled Orders:** Canceled orders and suspected fraudulent attempts (totaling 7,754 rows and 16,488 units) are strictly excluded from downstream demand modeling. Counting these would artificially inflate baseline demand forecasts and cause excessive safety stock sizing for sales that never materialized.
3. **Treatment of Returns vs. Positive Quantities:** While `Order Item Quantity` is strictly positive in all rows ($[1, 5]$), positive quantities do **not** rule out post-delivery returns. DataCo does not log Return Merchandise Authorization (RMA) dates. Hence, post-delivery returns cannot be subtracted from sales on subsequent dates without inventing unobserved return timestamps.
4. **Zero-Demand vs. Stockout Boundary:** Non-selling days represent unobserved sales. They must **not** be interpreted as evidence of a warehouse stockout. In retail data, zero sales can be caused by natural zero demand, catalog non-listing, or stockouts.

---

## 4. Mathematical Accounting Reconciliation

The pipeline enforces rigorous checksum accounting before generating final outputs:

$$\text{Raw Rows} (180,519) = \text{Included Rows} (172,765) + \text{Excluded Rows} (7,754) \quad [\checkmark]$$
$$\text{Raw Units} (384,079) = \text{Included Units} (367,591) + \text{Excluded Units} (16,488) \quad [\checkmark]$$
$$\sum \text{Daily Aggregated Demand} = \sum \text{Included Transaction Quantities} = 367,591.0 \quad [\checkmark]$$

* **Invalid Rows Encountered**: 0 unparseable dates, 0 missing product IDs, 0 non-numeric quantities.
* **Duplicate Aggregate Pairs**: Exactly 0 duplicate `(date, product_id)` records in the aggregated output.
* **Demand Sign**: 100% of aggregated demand values are strictly positive ($\ge 1.0$).

---

## 5. Processed Dataset Profiles

### 5.1 Daily Product Demand (`dataco_daily_product_demand.parquet`)
- **Location**: `data/processed/dataco_daily_product_demand.parquet`
- **File Size**: 61,133 bytes
- **Row Count**: 21,787 daily product demand records
- **Schema**:
  - `date` (`datetime64[ns]`): Normalized midnight calendar date
  - `product_id` (`string`): Clean product identifier (e.g. `'19'`, `'1004'`)
  - `demand` (`float64`): Sum of included unit quantities ordered on that calendar day
- **Temporal Coverage**: 2015-01-01 to 2018-01-31 (1,127 distinct calendar dates across ~36 months)
- **Product Coverage**: Exactly 118 distinct SKUs
- **Daily Demand Statistics**:
  - Minimum: 1.0 units
  - 25th Percentile: 3.0 units
  - Median (50th): 5.0 units
  - Mean: 16.87 units
  - 75th Percentile: 22.0 units
  - Maximum: 133.0 units

### 5.2 Product Catalog Metadata (`dataco_product_catalog.parquet`)
- **Location**: `data/processed/dataco_product_catalog.parquet`
- **File Size**: 10,133 bytes
- **Row Count**: 118 unique products
- **Schema**:
  - `product_id` (`string`): Primary product key
  - `product_name` (`string`): Clean descriptive title (e.g. `Nike Men's CJ Elite 2 TD Football Cleat`)
  - `product_price` (`float64`): List catalog price ($[9.99, 1999.99]$)
  - `category_id` (`int64`): Category code
  - `category_name` (`string`): Category title (50 categories, e.g. `Cleats`, `Men's Footwear`)
  - `department_id` (`int64`): Department code
  - `department_name` (`string`): Department title (`Fitness`, `Apparel`, `Golf`, `Footwear`, `Outdoors`, `Fan Shop`)
- **PII Audit**: Zero customer-identifying fields (`Customer Id`, `Customer Email`, `Customer Password`, `Customer Name`, `Customer Street`, or location coordinates) are present in this artifact.

---

## 6. Strategic Limitations & Safeguards for Portfolio Presentation

1. **Downstream Delivery vs. Upstream Replenishment:**  
   The observed delivery metric (`Days for shipping (real)`) measures courier delivery duration from warehouse to consumer ($0 - 6$ days). It must never be described as vendor replenishment lead time.
2. **Simulated Inventory Levels:**  
   No physical warehouse inventory balances exist in DataCo. Any inventory simulation performed in subsequent phases must be clearly presented as an operational simulation over empirical customer demand.
3. **No Inference of Stockouts:**  
   Zero sales on any given day cannot be inferred as stockouts.
