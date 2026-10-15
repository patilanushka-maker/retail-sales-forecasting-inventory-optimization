# DataCo Smart Supply Chain Dataset: Provenance & Attribution

> **Dataset Scope**: Secondary transaction and logistics dataset evaluating supply chain fulfillment, product-level demand forecasting, and operational simulation.

---

## 1. Authoritative Provenance Record

| Attribute | Specification |
| :--- | :--- |
| **Dataset Title** | DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS |
| **Original Authors** | Fabián Vinicio Constante Nicolalde, Fernando Silva, António Pereira |
| **Affiliations** | Universidad Central del Ecuador (Quito, Ecuador); Instituto Politécnico de Leiria, Centro de Investigação em Informática e Comunicações (CIIC, Portugal) |
| **Publication Date** | March 12, 2019 (Version 5) |
| **Authoritative Repository** | [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/5) |
| **Digital Object Identifier (DOI)** | `10.17632/8gx2fvg2k6.5` (Base DOI: `10.17632/8gx2fvg2k6`) |
| **Associated Research Paper** | Constante, F., Silva, F., & Pereira, A. (2019). *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS*. Data in Brief / Mendeley Data. |
| **Governing License** | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **Acquisition Access Date** | October 15, 2026 |
| **Local File Location** | `data/raw/DataCoSupplyChainDataset.csv` |
| **SHA-256 Checksum** | `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6` |
| **Byte Size** | 95,910,149 bytes (~91.47 MB uncompressed) |

---

## 2. Mandatory Attribution Notice

In accordance with Section 3 of the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, any reproduction, distribution, public display, or derived model must include the following formal citation:

```bibtex
@misc{dataco_supply_chain_2019,
  author       = {Constante, Fabi{\'{a}}n and Silva, Fernando and Pereira, Ant{\'{o}}nio},
  title        = {DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS},
  year         = {2019},
  publisher    = {Mendeley Data},
  version      = {5},
  doi          = {10.17632/8gx2fvg2k6.5},
  url          = {https://data.mendeley.com/datasets/8gx2fvg2k6/5}
}
```

**Human-Readable Citation**:  
> Constante, F., Silva, F., & Pereira, A. (2019). *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS* (Version 5) [Data set]. Mendeley Data. https://doi.org/10.17632/8gx2fvg2k6.5

---

## 3. Terms of Use & Redistribution Conditions

Under CC BY 4.0:
1. **You are free to**:
   - **Share**: Copy and redistribute the material in any medium or format.
   - **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.
2. **Under the following terms**:
   - **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
   - **No additional restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

---

## 4. Privacy, PII & Ethical Safeguards

Although published under an open scientific license, the raw dataset contains customer-related nominal fields:
- `Customer Email`: All 180,519 records contain masked placeholders (`XXXXXXXXX`).
- `Customer Password`: All 180,519 records contain masked placeholders (`XXXXXXXXX`).
- `Customer Fname` and `Customer Lname`: Contain synthetic/anonymized name tokens.
- `Customer Street`: Contains address strings.

### Engineering Privacy Rules for this Repository:
- **No Individual PII Logging**: Ingestion logs, validation scripts, test fixtures, and dashboard views must never display individual customer names, emails, or street addresses.
- **Aggregation Only**: Only aggregated regional indices (`Market`, `Order Region`, `Order Country`, `Order State`) may be surfaced in downstream reporting.
- **Scope Isolation**: Customer identifiers (`Customer Id`, `Order Customer Id`) are used solely for deduplication and cohort aggregation, not individual profiling.
