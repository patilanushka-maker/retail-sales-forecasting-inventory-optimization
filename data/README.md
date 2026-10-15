# Data Management & Storage Guidelines

This directory houses all data artifacts utilized throughout the demand forecasting and inventory optimization pipeline across three structured tiers.

---

## Directory Hierarchy

- `raw/`: Unmodified, immutable source data files.
  - **Location**: `data/raw/Online Retail.xlsx` (23.7 MB, acquired from official UCI repository).
  - **Rule**: Never modify raw files in-place. Raw records remain byte-for-byte immutable.
- `interim/`: Intermediate transformed states.
  - **File**: `data/interim/cleaned_transactions.parquet`.
  - Transaction-level records with parsed dates, standardized SKU casing, and mutually exclusive transaction classification (`positive_sale`, `cancellation`, `return_only`, `zero_quantity`, `invalid`).
- `processed/`: Final, analysis-ready datasets.
  - **File**: `data/processed/daily_product_demand.parquet`.
  - Daily aggregated product-level demand series with product-lifespan zero-filling (`date`, `product_id`, `demand`).

---

## 🌐 Dataset Provenance & Source Details

- **Dataset Name**: UCI Online Retail
- **Official Source URL**: [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Source Format**: Microsoft Excel (`.xlsx`) workbook.
- **Dependency for Excel**: Reading `.xlsx` workbooks requires `openpyxl>=3.1.0` (declared in [`requirements.txt`](file:///d:/Anushka/IIT/DS/Retail%20Sales%20Forecasting%20&%20Inventory%20Optimization%20System/requirements.txt)).
- **Current Status**: Acquired and verified from official UCI source. Shape: 541,909 rows, 8 columns.

---

## 📋 Required Canonical Schema

| Column Name | Logical Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | String / Integer | Invoice identifier. 6-digit number; if prefixed with 'C', indicates a cancellation. |
| `StockCode` | String / Integer | Distinct product (item) code. |
| `Description` | String | Product name/title. |
| `Quantity` | Numeric (Integer/Float) | Quantity of items generated per transaction line. |
| `InvoiceDate` | Datetime / String | Transaction generation timestamp. |
| `UnitPrice` | Numeric (Float) | Price per unit sterling. |
| `CustomerID` | Numeric / Integer | Customer identifier (nullable for guest wholesale checkouts). |
| `Country` | String | Country name of the customer. |

---

## ⚙️ Implemented Preprocessing Policies (Phase 3)

1. **Transaction Classification**:
   - `cancellation`: Invoice numbers beginning with `'C'` or `'c'`.
   - `return_only`: Negative quantity lines without a `'C'` prefix.
   - `positive_sale`: Ordinary sales (`Quantity > 0`, non-cancelled, `UnitPrice > 0`).
2. **Primary Modeling Target**:
   - `demand` is constructed strictly from `positive_sale` transactions. Returns and cancellations are tracked in the data-quality audit ledger (`outputs/metrics/data_quality_summary.json`).
3. **Product-Lifespan Zero-Filling Policy**:
   - To avoid false zero-inflation prior to product introduction or after product discontinuation, zero-filling is applied strictly within each product's active lifespan ($[T_{\text{first\_sale}}, T_{\text{last\_sale}}]$).
