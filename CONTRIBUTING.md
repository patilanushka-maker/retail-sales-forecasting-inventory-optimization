# Contributing Guidelines

Thank you for contributing to the Retail Demand Forecasting & Inventory Optimization System.

---

## 🛠️ Development Setup

1. **Clone & Virtual Environment**:
   ```bash
   git clone <repo-url>
   cd Retail-Sales-Forecasting-Inventory-Optimization
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Editable Package Install**:
   ```bash
   pip install -e .
   ```

---

## 🧪 Testing & Code Quality

- **Run Test Suite**:
  ```bash
  pytest tests/
  ```
- **Code Standards**:
  - Follow PEP 8 style conventions.
  - Type annotations must be used on public interfaces.
  - Avoid lookahead data leakage in time-series transformations (shift rolling statistics before computation).

---

## 📦 Pull Request Process
1. Create a descriptive feature branch (`feature/feature-name` or `fix/issue-description`).
2. Add corresponding unit tests in `tests/`.
3. Verify all tests pass locally.
4. Submit PR against `main`.
