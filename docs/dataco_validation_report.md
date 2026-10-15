# DataCo Smart Supply Chain Dataset: Validation & Verification Report (Phase 8A)

> **Document Status**: Complete Empirical Audit  
> **Target Artifact**: `data/raw/DataCoSupplyChainDataset.csv`  
> **Review Date**: October 15, 2026  
> **Reviewer**: Senior Data Scientist & Data Engineering Reviewer  

---

## 1. Executive Summary

As part of Phase 8A, the raw DataCo Smart Supply Chain dataset (`DataCoSupplyChainDataset.csv`) was acquired and comprehensively validated. The dataset contains **180,519 order-line item records** spanning **53 columns** across **3 calendar years (2015-01-01 to 2018-01-31)**.

Key validation conclusions:
1. **Integrity & Parseability**: The CSV parses cleanly without malformed rows or delimiter errors using standard `latin1` (ISO-8859-1) encoding.
2. **Zero Inversions**: There are **0 chronological inversions** (no shipping timestamp precedes its order timestamp).
3. **Zero Exact Duplicates**: There are **0 full duplicate rows**.
4. **Clean Quantities**: All quantities in `Order Item Quantity` are strictly positive integers in the range $[1, 5]$. No negative quantities or zero quantities exist.
5. **Exact Alignment**: The empirical data confirms the structural findings of the feasibility audit, with minor documented nuances regarding missing values in customer last names.

---

## 2. File Verification & Physical Properties

| Metric | Measured Value | Standard / Expectation |
| :--- | :--- | :--- |
| **File Path** | `data/raw/DataCoSupplyChainDataset.csv` | Under project raw directory |
| **File Size** | **95,910,149 bytes** (91.47 MB) | Standard uncompressed distribution |
| **Encoding** | `latin1` (ISO-8859-1) / `cp1252` | Characters in customer names/cities require latin1 |
| **SHA-256 Checksum** | `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6` | Matches authoritative distribution checksum |
| **Row Count** | **180,519 rows** | Exact match |
| **Column Count** | **53 columns** | Exact match |
| **Duplicate Rows** | **0 (0.0%)** | Exact match |

---

## 3. Schema & Column Catalog (All 53 Columns)

| # | Column Name | Storage Type | Null Count | Null % | Domain / Description |
| :- | :--- | :--- | :- | :- | :--- |
| 1 | `Type` | string (`object`) | 0 | 0.0% | Payment type (`DEBIT`, `TRANSFER`, `PAYMENT`, `CASH`) |
| 2 | `Days for shipping (real)` | integer (`int64`) | 0 | 0.0% | Empirical days taken for customer delivery $[0, 6]$ |
| 3 | `Days for shipment (scheduled)` | integer (`int64`) | 0 | 0.0% | Scheduled delivery promise $[0, 4]$ |
| 4 | `Benefit per order` | float (`float64`) | 0 | 0.0% | Net profit/loss per transaction line ($[-4274.98, 911.80]$) |
| 5 | `Sales per customer` | float (`float64`) | 0 | 0.0% | Net sales price after discount |
| 6 | `Delivery Status` | string (`object`) | 0 | 0.0% | `Late delivery`, `Advance shipping`, `Shipping on time`, `Shipping canceled` |
| 7 | `Late_delivery_risk` | integer (`int64`) | 0 | 0.0% | Binary flag: 1 if `Days (real) > Days (scheduled)`, else 0 |
| 8 | `Category Id` | integer (`int64`) | 0 | 0.0% | Product category numerical identifier |
| 9 | `Category Name` | string (`object`) | 0 | 0.0% | Nominal category (e.g. `Sporting Goods`, `Cleats`, `Women's Apparel`) |
| 10 | `Customer City` | string (`object`) | 0 | 0.0% | Customer residence city |
| 11 | `Customer Country` | string (`object`) | 0 | 0.0% | Customer residence country (`EE. UU.`, `Puerto Rico`) |
| 12 | `Customer Email` | string (`object`) | 0 | 0.0% | Masked customer email (`XXXXXXXXX`) |
| 13 | `Customer Fname` | string (`object`) | 0 | 0.0% | Customer first name |
| 14 | `Customer Id` | integer (`int64`) | 0 | 0.0% | Unique customer identifier |
| 15 | `Customer Lname` | string (`object`) | **8** | **0.004%** | Customer last name (8 records missing) |
| 16 | `Customer Password` | string (`object`) | 0 | 0.0% | Masked customer password (`XXXXXXXXX`) |
| 17 | `Customer Segment` | string (`object`) | 0 | 0.0% | `Consumer`, `Corporate`, `Home Office` |
| 18 | `Customer State` | string (`object`) | 0 | 0.0% | Customer residence state/territory (e.g., `PR`, `CA`, `NY`) |
| 19 | `Customer Street` | string (`object`) | 0 | 0.0% | Customer street address string |
| 20 | `Customer Zipcode` | float (`float64`) | **3** | **0.002%** | Customer postal code (3 records missing) |
| 21 | `Department Id` | integer (`int64`) | 0 | 0.0% | High-level department identifier |
| 22 | `Department Name` | string (`object`) | 0 | 0.0% | `Fitness`, `Apparel`, `Golf`, `Footwear`, `Outdoors`, `Fan Shop` |
| 23 | `Latitude` | float (`float64`) | 0 | 0.0% | Store/fulfillment center latitude |
| 24 | `Longitude` | float (`float64`) | 0 | 0.0% | Store/fulfillment center longitude |
| 25 | `Market` | string (`object`) | 0 | 0.0% | Global trading market (`LATAM`, `Europe`, `Pacific Asia`, `USCA`, `Africa`) |
| 26 | `Order City` | string (`object`) | 0 | 0.0% | Destination delivery city |
| 27 | `Order Country` | string (`object`) | 0 | 0.0% | Destination delivery country |
| 28 | `Order Customer Id` | integer (`int64`) | 0 | 0.0% | Customer ID tied to order |
| 29 | `order date (DateOrders)` | datetime/str (`object`)| 0 | 0.0% | Order creation timestamp (`M/d/yyyy H:mm`) |
| 30 | `Order Id` | integer (`int64`) | 0 | 0.0% | Unique order identifier |
| 31 | `Order Item Cardprod Id`| integer (`int64`) | 0 | 0.0% | Product card ID tied to order item |
| 32 | `Order Item Discount` | float (`float64`) | 0 | 0.0% | Discount amount in USD ($[0.00, 500.00]$) |
| 33 | `Order Item Discount Rate`| float (`float64`) | 0 | 0.0% | Discount rate $[0.00, 0.25]$ (up to 25%) |
| 34 | `Order Item Id` | integer (`int64`) | 0 | 0.0% | Unique order-item transaction line ID |
| 35 | `Order Item Product Price`| float (`float64`) | 0 | 0.0% | Unit item price at transaction |
| 36 | `Order Item Profit Ratio`| float (`float64`) | 0 | 0.0% | Profit margin percentage |
| 37 | `Order Item Quantity` | integer (`int64`) | 0 | 0.0% | Units ordered per line item ($[1, 5]$) |
| 38 | `Sales` | float (`float64`) | 0 | 0.0% | Gross sales price (`Order Item Product Price * Order Item Quantity`) |
| 39 | `Order Item Total` | float (`float64`) | 0 | 0.0% | Net transaction line total (`Sales - Order Item Discount`) |
| 40 | `Order Profit Per Order`| float (`float64`) | 0 | 0.0% | Net profit per order |
| 41 | `Order Region` | string (`object`) | 0 | 0.0% | Delivery region (e.g., `Western Europe`, `Central America`) |
| 42 | `Order State` | string (`object`) | 0 | 0.0% | Destination state / administrative region |
| 43 | `Order Status` | string (`object`) | 0 | 0.0% | Order execution status |
| 44 | `Order Zipcode` | float (`float64`) | **155,679** | **86.240%** | Destination zip code (populated primarily for domestic US orders) |
| 45 | `Product Card Id` | integer (`int64`) | 0 | 0.0% | Primary product identifier (SKU ID) |
| 46 | `Product Category Id` | integer (`int64`) | 0 | 0.0% | Category numerical identifier |
| 47 | `Product Description` | float (`float64`) | **180,519** | **100.000%**| Column is completely empty (all NaN) |
| 48 | `Product Image` | string (`object`) | 0 | 0.0% | Image URL link |
| 49 | `Product Name` | string (`object`) | 0 | 0.0% | Title of the product |
| 50 | `Product Price` | float (`float64`) | 0 | 0.0% | List catalog price |
| 51 | `Product Status` | integer (`int64`) | 0 | 0.0% | Catalog status binary flag (all rows are 0) |
| 52 | `shipping date (DateOrders)`| datetime/str (`object`)| 0 | 0.0% | Dispatch/shipping timestamp (`M/d/yyyy H:mm`) |
| 53 | `Shipping Mode` | string (`object`) | 0 | 0.0% | `Standard Class`, `First Class`, `Second Class`, `Same Day` |

---

## 4. Key Distributions & Diagnostic Findings

### 4.1 Order Status Distribution
| Order Status | Row Count | Percentage | Operational Meaning |
| :--- | :- | :- | :--- |
| `COMPLETE` | 59,491 | 32.96% | Order paid and fulfilled |
| `PENDING_PAYMENT` | 39,832 | 22.06% | Awaiting payment settlement |
| `PROCESSING` | 21,902 | 12.13% | In fulfillment process |
| `PENDING` | 20,227 | 11.20% | Awaiting fulfillment |
| `CLOSED` | 19,616 | 10.87% | Completed and archived |
| `ON_HOLD` | 9,804 | 5.43% | Under customer/inventory hold |
| `SUSPECTED_FRAUD` | 4,062 | 2.25% | Flagged fraudulent transaction |
| `CANCELED` | 3,692 | 2.05% | Order canceled by customer/system |
| `PAYMENT_REVIEW` | 1,893 | 1.05% | Payment verification check |

### 4.2 Delivery Status Distribution
| Delivery Status | Row Count | Percentage |
| :--- | :- | :- |
| `Late delivery` | 98,977 | 54.83% |
| `Advance shipping` | 41,592 | 23.04% |
| `Shipping on time` | 32,196 | 17.84% |
| `Shipping canceled` | 7,754 | 4.30% |

### 4.3 Outbound Shipping Durations vs. Scheduled Promises
| Metric | `Days for shipping (real)` | `Days for shipment (scheduled)` |
| :--- | :- | :- |
| **Mean** | 3.50 days | 2.93 days |
| **Standard Deviation** | 1.62 days | 1.37 days |
| **Min** | 0 days | 0 days |
| **25th Percentile** | 2.00 days | 2.00 days |
| **Median (50th)** | 3.00 days | 4.00 days |
| **75th Percentile** | 5.00 days | 4.00 days |
| **Max** | 6.00 days | 4.00 days |

---

## 5. Comparison: Actual Observations vs. Feasibility Report

| Parameter | Feasibility Report Hypothesis | Actual Empirical Inspection | Assessment & Flags |
| :--- | :--- | :--- | :--- |
| **Row Count** | 180,519 | 180,519 | **Exact Match** |
| **Column Count**| 53 | 53 | **Exact Match** |
| **Date Range** | 2015-01-01 to 2018-01-31 | 2015-01-01 to 2018-01-31 | **Exact Match** (1,127 distinct dates) |
| **Unique Orders**| 65,752 | 65,752 | **Exact Match** |
| **Unique SKUs** | 118 | 118 | **Exact Match** (`Product Card Id`) |
| **Missing Values**| Expected nulls in `Customer Zipcode`, `Order Zipcode`, `Product Description` | Confirmed those three, **plus 8 missing values in `Customer Lname` (0.004%)** | **Minor Discrepancy**: 8 rows lack a customer last name. Irrelevant to demand aggregation. |
| **Quantity Domain**| Assumed strictly positive integers | Confirmed: values are integers $\in [1, 5]$. No negatives or zeros | **Exact Match** |
| **Shipping Inversions**| Hypothesized to be consistent | Verified: 0 inversions (`shipping date >= order date` across all 180,519 rows) | **Exact Match** |
| **Supplier Lead Time**| Hypothesized unobserved | Confirmed: Zero inbound vendor/PO fields exist. `Days for shipping (real)` is purely customer outbound shipping | **Confirmed Constraint** |

---

## 6. Recommendations for Downstream Phase 8B Ingestion

1. **Encoding**: File ingestion must specify `encoding="latin1"` to prevent `UnicodeDecodeError` on Latin accented characters in names and cities.
2. **Date Parsing**: Timestamps in `order date (DateOrders)` must be parsed explicitly and truncated to date (`YYYY-MM-DD`).
3. **Canceled / Fraud Filtering**: When generating canonical daily product demand, records with `Order Status.isin(['CANCELED', 'SUSPECTED_FRAUD'])` should be excluded to reflect true net fulfilled retail demand.
4. **Demand Zero-Filling**: Cartesian product of `(1,127 calendar dates) × (118 SKUs)` must be constructed, filling unobserved daily sales with 0.
