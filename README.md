# 🦅 Kestrel Home Appliances — Returns Risk Service

A local Python return risk scoring service, mathematical explainability engine, and operational decision system for **Kestrel Home Appliances** (Pune, India).

Training and serving share the same feature code ([model.py](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/model.py)), guaranteeing identical feature definitions and numerical consistency between batch CSV scoring and online API inference.

---

## 🖥️ Interactive Web Dashboard Preview

The service includes a modern, high-performance operational dashboard running locally on `http://127.0.0.1:8000`:

### 1. Live Order Scorer & Mathematical XAI (Tab 1)
*Real-time return probability scoring, raw log-odds calculation, mathematical feature attribution, and counterfactual actionable levers.*
![Live Order Scorer](assets/dashboard_live_scorer.png)

<br>

### 2. Validation & Calibration Evidence + Live Profit Simulator (Tab 2)
*Historical holdout audit metrics, binned calibration table, and an interactive policy threshold & net annual profit simulator ($5\% - 30\%$).*
![Validation Evidence & Profit Simulator](assets/dashboard_validation_evidence.png)

<br>

### 3. Implementation Decisions & Business Economics Tuner (Tab 3)
*Architectural invariants (zero-leakage, gateway anomaly repair) and live business parameter tuner (Return Cost, Call Cost, Prevention Rate).*
![Implementation Decisions & Policy Tuner](assets/dashboard_policy_tuner.png)

---

## 📁 Repository Structure

| Location | Files | Purpose |
| :--- | :--- | :--- |
| **Root** | [`README.md`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/README.md), [`requirements.txt`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/requirements.txt), [`.env.example`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/.env.example), [`.gitignore`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/.gitignore) | Installation, environment configuration, and execution instructions |
| **Root** | [`model.py`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/model.py) | Data cleaning, reference joins, dispatch-safe features, and candidate model pipelines |
| **Root** | [`train.py`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/train.py) | Forward validation (Jan–Mar), holdout audit (Apr–Jun), refitting, and batch predictions |
| **Root** | [`app.py`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/app.py) | HTTP server, input validation, scoring, mathematical explanations, and business actions |
| **Root** | [`index.html`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/index.html) | Working single-page screen calling `POST /predict` via JavaScript `fetch()` |
| **Root** | [`tests.py`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/tests.py) | Automated test suite verifying data contracts, zero-leakage invariants, and API consistency |
| **Root** | [`model.joblib`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/model.joblib) | Saved preprocessing pipeline, classifier weights, and explanation baseline |
| **Root** | [`predictions.csv`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/predictions.csv) | Final test-order risk scores |
| **Root** | [`make_memo_pdf.py`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/make_memo_pdf.py), [`memo_to_ritu.pdf`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/memo_to_ritu.pdf) | Executive 1-page PDF memo generator and output |
| **Root** | [`memo_to_ritu.md`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/memo_to_ritu.md), [`submission-form.md`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/submission-form.md) | Business deliverables (Executive Memo & Completed Submission Form) |
| **`frontend/`** | `index.html`, `styles.css`, `app.js`, `demo-examples.json`, `example-order.json` | Modular dashboard UI assets, styles, and sample order presets |
| **`data/`** | Training, test, customers, products, sample submission, policy, and email thread | Input data pack |
| **`evidence/`** | Validation metrics, calibration tables, segment audits, and test execution logs | Model validation and failure analysis evidence |

---

## 🚀 Execution & Command Guide (What is for What?)

| Command | What It Does | When Do You Need to Run It? |
| :--- | :--- | :--- |
| `python app.py` | Starts the **Local REST API server** and serves the **Interactive Web Dashboard** on `http://127.0.0.1:8000`. | **Daily / Anytime you want to use the application.** This is the main command to launch the web dashboard and prediction service. |
| `python tests.py` | Executes all **14 automated unit, invariant, and edge-case test suites** in ~1.1 seconds. | **Whenever you want to verify code quality, data invariants, or test before submission.** |
| `python train.py` | Performs forward time-based validation (Jan–Mar selection, Apr–Jun holdout audit), refits on 100% history, saves `model.joblib`, and generates `predictions.csv`. | **Run ONLY ONCE (Already Done!)** or when new historical training datasets are added. |
| `python make_memo_pdf.py` | Compiles the executive 1-page PDF memorandum for Ritu Deshpande and Kestrel leadership (`memo_to_ritu.pdf`). | **Only when you need to re-generate the PDF business memo.** |

---

## 🛠️ Step-by-Step Instructions

### Step 1: Install Dependencies (One-Time Setup)
```bash
pip install -r requirements.txt
```

### Step 2: Start the Web Dashboard & API (Daily Use)
```bash
python app.py
```
* **Interactive Web Dashboard:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Direct Executive Memo Download (PDF):** Click the purple **"📄 Executive Memo (PDF)"** button in the dashboard header, or visit [http://127.0.0.1:8000/memo.pdf](http://127.0.0.1:8000/memo.pdf)
* **Interactive Swagger API Docs:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

*(Press `Ctrl + C` in the terminal when you are done).*

### Step 3: Run Automated Verification Tests
```bash
python tests.py
```
* **Status:** All **14 automated test suites** pass in ~1.1s.
* **Coverage:** Data leakage invariants, 100x October gateway anomaly fix, Empirical Bayes smoothing, economic break-even calling rules, Shield VIP protection, and corrupted input resilience.

### Step 4: Retrain Model & Regenerate CSV Predictions (Optional)
```bash
python train.py
```
*(Only needed if you modify feature engineering or add new historical orders).*

---

## 🔬 Training and Batch-Prediction Pipeline

```mermaid
flowchart TD
    A["Training orders (train.csv)"] --> B["Deduplicate by order_id (canonical)"]
    B --> C["Join customer & product references"]
    C --> D["Extract dispatch-safe features"]
    D --> E["Forward validation: Jan–Mar 2026"]
    E --> F["Select model by average precision (PR-AUC)"]
    F --> G["Audit on April–June 2026 holdout"]
    G --> H["Save evidence to evidence/"]
    F --> I["Refit on all 15 months of labeled history"]
    I --> J["Save model.joblib"]
    J --> K["Score July–September 2026 test orders"]
    K --> L["Validate IDs & write predictions.csv"]
```

### Feature Engineering & Preprocessing
The feature set contains **11 numerical and 6 categorical fields**:
- **Numerical:** `discount_pct`, `qty`, `promised_delivery_days`, `customer_prior_orders`, `customer_prior_returns`, `prior_return_rate_smooth`, `expected_value_inr`, `is_missing_address`, `is_gift_num`, `warranty_months`, `product_age_days`.
- **Categorical:** `sales_channel`, `payment_mode`, `sku`, `family`, `shield_member`, `state`.

#### Core Engineered Features:
1. **Empirical Bayes Smoothed Prior Return Rate:**
   $$\text{prior\_return\_rate\_smooth} = \frac{\text{prior\_returns} + 0.5}{\text{prior\_orders} + 5.0}$$
   *Avoids treating a customer with 1 order and 1 return as having an overconfident 100% return rate.*
2. **Reconstructed Basket Value:**
   $$\text{expected\_value\_inr} = \text{list\_price} \times \text{qty} \times \left(1 - \frac{\text{discount\_pct}}{100}\right)$$
   *Resolves the 100x payment gateway value anomaly from October 2025.*
3. **Product Age:**
   $$\text{product\_age\_days} = \max(0, \text{order\_placed\_at} - \text{launch\_date})$$

#### Model Selection:
In forward validation on January–March 2026, **Regularized Logistic Regression** achieved the highest Average Precision (PR-AUC):
* **Logistic Regression:** PR-AUC = `0.3883`, ROC-AUC = `0.7708`, Brier = `0.0871`
* **Histogram Gradient Boosting:** PR-AUC = `0.3545`, ROC-AUC = `0.7630`, Brier = `0.0896`

In the **April–June 2026 Holdout Audit**, the model achieved:
* **ROC-AUC:** `0.7869`
* **PR-AUC:** `0.4150` (3.6x over baseline `0.114`)
* **Brier Score:** `0.0848`
* **Accuracy:** `89.6%`

To retrain the model and regenerate all predictions and evidence:
```bash
python train.py
```

---

## ⚡ Online Scoring API & Operational Decision Rules

### Single Scoring Endpoint: `POST /predict`

#### Sample Request (`frontend/example-order.json`):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @frontend/example-order.json
```

#### Actual JSON Response:
```json
{
  "order_id": "KO2611078",
  "score": 0.350925,
  "return_probability_pct": 35.09,
  "action": "confirm_before_dispatch",
  "action_recommendation": "Recommend ₹45 pre-dispatch confirmation call. Projected net savings: ₹96.25.",
  "automatic_hold": false,
  "call_threshold": 0.111801,
  "estimated_call_net_inr": 96.25,
  "reasons": [
    "Payment Mode Cod increases return risk by 0.551 log-odds.",
    "Discount Pct increases return risk by 0.503 log-odds.",
    "Promised Delivery Days increases return risk by 0.215 log-odds.",
    "Sales Channel Marketplace increases return risk by 0.213 log-odds."
  ],
  "model_factors": [
    {
      "feature": "payment_mode_cod",
      "log_odds_contribution": 0.5513,
      "direction": "increases",
      "summary": "Payment Mode Cod increases return risk by 0.551 log-odds."
    },
    {
      "feature": "discount_pct",
      "log_odds_contribution": 0.5034,
      "direction": "increases",
      "summary": "Discount Pct increases return risk by 0.503 log-odds."
    },
    {
      "feature": "promised_delivery_days",
      "log_odds_contribution": 0.2155,
      "direction": "increases",
      "summary": "Promised Delivery Days increases return risk by 0.215 log-odds."
    },
    {
      "feature": "sales_channel_marketplace",
      "log_odds_contribution": 0.2129,
      "direction": "increases",
      "summary": "Sales Channel Marketplace increases return risk by 0.213 log-odds."
    }
  ],
  "warnings": [],
  "model_version": "1.0.0",
  "audit_metrics": {
    "audit_roc_auc": 0.7869,
    "audit_pr_auc": 0.415,
    "audit_brier_score": 0.0848,
    "audit_accuracy": 0.8956
  }
}
```

---

### Mathematical Feature Explanations (No External LLM Required)
For each transformed feature $j$, the exact log-odds contribution is calculated as:
$$\text{contribution}_j = \beta_j \left(x_j - \overline{x}_j\right)$$
where $x_j$ is the transformed input and $\overline{x}_j$ is its training population mean. One-hot categories are grouped to parent fields and the top absolute contributors are mapped to human-readable explanations.

---

### Economic Decision Framework

$$\text{Expected Net Call Benefit} = p \times 0.35 \times ₹1{,}150 - ₹45 \implies \mathbf{11.18\%\text{ Break-Even Threshold}}$$

| Condition | Action | Operational Logic |
| :--- | :--- | :--- |
| **Customer absent from reference database** | `verify_customer_details` | Verify account and address before dispatch |
| **Known customer, risk $\ge 11.18\%$** | `confirm_before_dispatch` | Trigger ₹45 confirmation call (avoids 35% of returns, positive ROI) |
| **Known customer, risk $< 11.18\%$** | `dispatch_normally` | Fast-track dispatch |
| **Any score** | `automatic_hold = false` | Never hold orders blindly (prevents 12% customer cancellation loss) |
| **Shield Member ($Y$)** | `shield_warning` | Active warning to preserve VIP lifetime customer value |

---

## 📄 Business Deliverables

### Executive 1-Page Memorandum to Ritu Deshpande
* **Canonical Briefing File:** [`memo_to_ritu.pdf`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/memo_to_ritu.pdf) (Markdown source: [`memo_to_ritu.md`](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/memo_to_ritu.md))
* **Dynamic Parameterized Generation:** The `/memo.pdf` endpoint supports live query parameters (e.g. `GET /memo.pdf?return_cost=1500&call_cost=60&prevention_rate=0.40`). When business cost parameters are modified in the web UI (Tab 3), clicking the **"📄 Executive Memo (PDF)"** button automatically generates and downloads a dynamically recalculated executive memo with updated thresholds, ROI figures, and P&L projections.

![Executive 1-Page Memo Preview](assets/executive_memo_preview.png)

- **Submission Form:** [submission-form.md](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/submission-form.md)
- **Predictions CSV:** [predictions.csv](file:///Users/madhavkumar/Desktop/Kestrel%20Home-Returns%20Risk/predictions.csv)
