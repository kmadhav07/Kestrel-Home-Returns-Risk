# Kestrel Home — Returns Risk Submission Form

## 1. Submission Metadata
- **Project Name:** Kestrel Home — Returns Risk Intelligence Service
- **Submission Date:** October 1, 2026
- **Deliverables Included:**
  - `predictions.csv` (2,096 scored test orders matching `sample_submission.csv` format and IDs)
  - `model.py` (canonical deduplication, reference joins, shared features, preprocessing pipelines)
  - `train.py` (forward validation, holdout audit, refitting, and batch predictions)
  - `app.py` (HTTP service with `POST /predict`, `GET /health`, and UI serving)
  - `index.html` (single working HTML screen calling `POST /predict` via JavaScript `fetch()`)
  - `tests.py` (14 automated test suites covering data contracts, zero-leakage invariants, API consistency, and edge-case resiliency)
  - `model.joblib` (saved preprocessing pipeline, classifier, and explanation baseline)
  - `memo_to_ritu.md` & `memo_to_ritu.pdf` (Executive 1-page non-technical memo to Ritu Deshpande)
  - `evidence/` (validation metrics, calibration tables, segment audits, and test execution logs)

---

## 2. Model Performance Expectations on Unlabelled Test Data

### Expected Test Set Score & Metrics
- **Expected ROC-AUC:** `0.775 – 0.795` (April–June forward holdout audit achieved **0.7869**)
- **Expected Average Precision (PR-AUC):** `0.380 – 0.420` (April–June forward holdout audit achieved **0.4150**, 3.6x over random baseline `0.1142`)
- **Expected Brier Score (Calibration):** `0.082 – 0.086` (April–June audit achieved **0.0848**)
- **Expected Accuracy:** `~89.6%`
- **Expected Base Return Rate:** `~11.5%`

### Why We Expect This Score
1. **Strict Zero Data Leakage:** We verified that changing `pickup_scheduled_at`, `last_service_event_type`, or delivery notes does not alter model predictions. Only pre-dispatch fields are used, strictly matching the dispatch snapshot in `test_unlabelled.csv`.
2. **Canonical Deduplication:** We purged 651 duplicated partner-feed records in `train.csv` (1,302 duplicate rows) while verifying label agreement and prioritizing primary CRM records.
3. **Gateway Anomaly Correction:** We identified and resolved the 100x payment gateway value scaling bug in October 2025 orders via reconstructed expected basket values `list_price * qty * (1 - discount_pct/100)`.
4. **Empirical Bayes Smoothing:** Prior return rates are smoothed via `(prior_returns + 0.5) / (prior_orders + 5.0)` to avoid overreacting to 1-order customers.
5. **Time-Based Forward Validation:** The model was selected on January–March 2026 history and verified on April–June 2026 holdout history, avoiding random k-fold look-ahead bias.

---

## 3. Methodology & Technical Architecture

### 3.1 Data Cleaning & Feature Engineering
- **Deduplication:** `canonical()` verifies outcome consistency across duplicate partner feeds and retains exactly one record per `order_id` (10,504 clean unique orders).
- **Dispatch-Safe Feature Set:** 11 numerical and 6 categorical fields:
  - *Numerical:* `discount_pct`, `qty`, `promised_delivery_days`, `customer_prior_orders`, `customer_prior_returns`, `prior_return_rate_smooth`, `expected_value_inr`, `is_missing_address`, `is_gift_num`, `warranty_months`, `product_age_days`.
  - *Categorical:* `sales_channel`, `payment_mode`, `sku`, `family`, `shield_member`, `state`.
- **Preprocessing:** Numerical median imputation + StandardScaler; Categorical constant imputation + OneHotEncoder with `handle_unknown="ignore"`. Preprocessing is fit strictly within each training window.

### 3.2 Candidate Model Selection (January–March 2026 Validation Split)

| Candidate Model | Forward Validation PR-AUC | Forward Validation ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: |
| **Regularized Logistic Regression (C=0.20)** | **0.3883** (Selected) | **0.7708** | **0.0871** |
| **Histogram Gradient Boosting (L2=2.0)** | **0.3545** | **0.7630** | **0.0896** |

### 3.3 Holdout Audit (April–June 2026)
- **ROC-AUC:** `0.7869`
- **PR-AUC:** `0.4150`
- **Brier Score:** `0.0848`
- **Accuracy:** `89.6%`

---

## 4. Operational & Economic Decision Framework

### Break-Even Calling Rule

$$\text{Expected Net Call Benefit} = p \times 0.35 \times ₹1{,}150 - ₹45 \implies \mathbf{11.18\%\text{ Break-Even Threshold}}$$

### Annualized Policy Comparison (10,504 Orders Baseline)

| Metric | Option A: Status Quo | Option B: Naive Hold (>24h) | Option C: Smart Call & Dispatch (Ours) |
| :--- | :--- | :--- | :--- |
| **Annual Order Volume** | 8,403 | 8,403 | 8,403 |
| **Intervention Strategy** | Ship all blindly | Hold all score $\ge 0.20$ | Auto-dispatch low risk; ₹45 call for score $\ge 11.18\%$ |
| **Orders Contacted/Held** | 0 | 1,820 held (21.7%) | 2,605 calls placed (31.0%) |
| **Returns Prevented** | 0 | 420 | **225 returns avoided** |
| **Gross Return Savings** | ₹0 | ₹4,83,000 | **₹2,58,750** |
| **Intervention / Calling Cost** | ₹0 | ₹0 | **₹1,17,225** |
| **Lost Margin (12% Cancellations)** | ₹0 | **-₹5,24,160** | **₹0** (No cancellations) |
| **Net P&L Benefit vs Status Quo** | **₹0** | **-₹41,160 (Net Loss!)** | **+₹1,41,525 Net Profit (120.7% ROI)** |

### Shield Member Policy Recommendation
- Shield members account for 22.1% of orders and 36.4% of returns (18.7% return rate) because policy grants free 30-day returns.
- **Rule:** Never hold a Shield member's order. Route flagged Shield orders to Meenal's service desk for white-glove setup assistance to preserve high lifetime customer value.

---

## 5. Mathematical Explainability Engine (No LLM Required)

For each transformed feature $j$, the exact log-odds contribution is calculated as:
$$\text{contribution}_j = \beta_j \left(x_j - \overline{x}_j\right)$$
where $x_j$ is the transformed input and $\overline{x}_j$ is its training population mean. One-hot categories are grouped to parent fields and the top 4 absolute contributors are mapped to human-readable explanations in `POST /predict`.

---

## 6. AI Tool Usage Disclosure

### What We Used
- AI Coding Assistants used for rapid EDA, script scaffolding, mathematical validation checks, and unit test coverage.

### What It Cost
- Developer token expense: ~$0.15.
- Runtime third-party paid API cost: **$0.00**. The final service runs 100% locally on native Python and Scikit-Learn without external paid API keys.

### What We Discarded
1. **LightGBM / XGBoost C-Library Dependencies:** Discarded due to missing `libomp.dylib` runtime dependencies on clean machines.
2. **Deep Neural Networks (MLP):** Discarded because multi-layer perceptrons overfit sparse one-hot city features and offered inferior calibration compared to regularized GLM.
3. **Target Leaking Features (`pickup_scheduled_at`, `REVERSE_PICKUP`):** Discarded immediately after auditing data provenance because post-delivery events cannot exist at dispatch time.

---

## 7. How to Run the Working Service

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated test suite
python tests.py

# 3. Start the service
python app.py
```
- **Web Screen:** Open `http://127.0.0.1:8000`
- **Swagger API Docs:** Open `http://127.0.0.1:8000/docs`
- **Predict Endpoint:** `POST /predict` (accepts single JSON order record and returns calibrated score, action, mathematical reasons, warnings, and net benefit).
