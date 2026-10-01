"""
Kestrel Home Returns Risk - Model Definition & Feature Engineering Pipeline
Provides canonical deduplication, reference joins, dispatch-safe feature engineering,
and candidate scikit-learn preprocessing + classifier pipelines.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


# Approved, Dispatch-Safe Feature Sets (Strictly available prior to dispatch)
NUMERICAL_FEATURES: List[str] = [
    "discount_pct",
    "qty",
    "promised_delivery_days",
    "customer_prior_orders",
    "customer_prior_returns",
    "prior_return_rate_smooth",
    "expected_value_inr",
    "is_missing_address",
    "is_gift_num",
    "warranty_months",
    "product_age_days"
]

CATEGORICAL_FEATURES: List[str] = [
    "sales_channel",
    "payment_mode",
    "sku",
    "family",
    "shield_member",
    "state"
]


def canonical(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures exactly one row per unique order_id.
    When duplicate orders exist due to partner-feed re-imports,
    verifies outcome agreement and prioritizes the primary CRM record.
    """
    df = train_df.copy()
    if "order_id" not in df.columns:
        return df

    # Sort so that primary 'crm' source records come before 'partner_feed'
    if "source" in df.columns:
        df["_source_rank"] = df["source"].map(lambda s: 0 if str(s).strip().lower() == "crm" else 1)
        df = df.sort_values(by=["order_id", "_source_rank"]).drop(columns=["_source_rank"])

    # Verify duplicate labels agree
    if "returned" in df.columns:
        dup_conflicts = df.groupby("order_id")["returned"].nunique()
        if (dup_conflicts > 1).any():
            raise ValueError("Conflicting return outcomes detected in duplicate order records!")

    # Retain one row per order
    df_dedup = df.drop_duplicates(subset=["order_id"], keep="first").copy()
    return df_dedup


def features(
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins customer and product reference tables and extracts dispatch-safe features.
    Strictly excludes post-delivery events, pickup bookings, notes, and unverified fields.
    """
    df = orders_df.copy()
    
    # Ensure default fallback columns exist if called with a partial single-order dictionary
    default_columns = {
        "customer_id": "UNKNOWN_CUST",
        "sku": "UNKNOWN_SKU",
        "sales_channel": "web",
        "payment_mode": "prepaid_upi",
        "is_gift": "N",
        "qty": 1,
        "discount_pct": 0.0,
        "order_value_inr": 0.0,
        "promised_delivery_days": 4,
        "delivery_pincode": 0,
        "customer_prior_orders": 0,
        "customer_prior_returns": 0,
        "order_placed_at": "2026-07-01 12:00"
    }
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val

    # 1. Standardize reference data copies with robust fallbacks
    if customers_df is not None and not customers_df.empty:
        cust = customers_df.copy()
        cust["customer_id"] = cust["customer_id"].astype(str).str.strip()
        cust["state"] = cust["state"].fillna("Unknown").astype(str).str.strip() if "state" in cust.columns else "Unknown"
        cust["shield_member"] = cust["shield_member"].fillna("N").astype(str).str.strip().str.upper() if "shield_member" in cust.columns else "N"
    else:
        cust = pd.DataFrame(columns=["customer_id", "state", "shield_member"])

    if products_df is not None and not products_df.empty:
        prod = products_df.copy()
        prod["sku"] = prod["sku"].astype(str).str.strip()
        prod["family"] = prod["family"].fillna("Unknown").astype(str).str.strip() if "family" in prod.columns else "Unknown"
        prod["list_price_inr"] = pd.to_numeric(prod["list_price_inr"], errors="coerce").fillna(4999.0) if "list_price_inr" in prod.columns else 4999.0
        prod["warranty_months"] = pd.to_numeric(prod["warranty_months"], errors="coerce").fillna(12).astype(int) if "warranty_months" in prod.columns else 12
        prod["launch_date"] = pd.to_datetime(prod["launch_date"], errors="coerce") if "launch_date" in prod.columns else pd.to_datetime("2024-01-01")
    else:
        prod = pd.DataFrame(columns=["sku", "family", "list_price_inr", "warranty_months", "launch_date"])

    # Drop potential reference columns if they exist in orders_df as null to avoid _x/_y collision
    for col in ["state", "shield_member", "family", "list_price_inr", "warranty_months", "launch_date"]:
        if col in df.columns and df[col].isna().all():
            df = df.drop(columns=[col])

    # 2. Join reference tables safely
    if "shield_member" not in df.columns and not cust.empty:
        df = df.merge(cust[["customer_id", "state", "shield_member"]], on="customer_id", how="left")
    if "family" not in df.columns and not prod.empty:
        df = df.merge(prod[["sku", "family", "list_price_inr", "warranty_months", "launch_date"]], on="sku", how="left")

    # 3. Clean and impute reference defaults
    if "shield_member" not in df.columns:
        df["shield_member"] = "N"
    df["shield_member"] = df["shield_member"].fillna("N").astype(str)

    if "state" not in df.columns:
        df["state"] = "Unknown"
    df["state"] = df["state"].fillna("Unknown").astype(str)

    if "family" not in df.columns:
        df["family"] = "Unknown"
    df["family"] = df["family"].fillna("Unknown").astype(str)

    if "warranty_months" not in df.columns:
        df["warranty_months"] = 12.0
    df["warranty_months"] = pd.to_numeric(df["warranty_months"], errors="coerce").fillna(12.0).astype(float)

    df["sku"] = df["sku"].fillna("Unknown").astype(str)
    df["sales_channel"] = df["sales_channel"].fillna("web").astype(str).str.strip().str.lower()
    df["payment_mode"] = df["payment_mode"].fillna("prepaid_upi").astype(str).str.strip().str.lower()

    # 4. Dates & Product Age
    order_dt = pd.to_datetime(df["order_placed_at"], errors="coerce")
    launch_dt = pd.to_datetime(df["launch_date"], errors="coerce") if "launch_date" in df.columns else pd.to_datetime("2024-01-01")
    product_age = (order_dt - launch_dt).dt.days
    df["product_age_days"] = product_age.clip(lower=0).fillna(365.0)

    # 5. Missing address / Walk-in indicator
    pincode_numeric = pd.to_numeric(df["delivery_pincode"], errors="coerce").fillna(0).astype(int)
    df["is_missing_address"] = (pincode_numeric == 0).astype(int)

    # 6. Gift flag
    df["is_gift_num"] = df["is_gift"].astype(str).str.strip().str.upper().map(lambda x: 1 if x == "Y" else 0).fillna(0)

    # 7. Reconstructed Expected Basket Value (resolves October festive gateway bug)
    qty = pd.to_numeric(df["qty"], errors="coerce").fillna(1).clip(lower=1)
    discount = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0).clip(lower=0, upper=100)
    raw_val = pd.to_numeric(df["order_value_inr"], errors="coerce").fillna(0.0)
    if "list_price_inr" in df.columns:
        list_price = pd.to_numeric(df["list_price_inr"], errors="coerce")
    else:
        list_price = pd.Series(np.nan, index=df.index)
    list_price = list_price.fillna(raw_val / qty).fillna(4999.0)
    df["expected_value_inr"] = (list_price * qty * (1.0 - discount / 100.0)).clip(lower=100.0)

    # 8. Empirical Bayes Smoothed Prior Return Rate (with sanity check)
    prior_orders = pd.to_numeric(df["customer_prior_orders"], errors="coerce").fillna(0).clip(lower=0)
    prior_returns = pd.to_numeric(df["customer_prior_returns"], errors="coerce").fillna(0).clip(lower=0)
    # Ensure prior_returns cannot exceed prior_orders
    prior_returns = np.minimum(prior_returns, prior_orders)
    df["customer_prior_orders"] = prior_orders
    df["customer_prior_returns"] = prior_returns
    df["prior_return_rate_smooth"] = (prior_returns + 0.5) / (prior_orders + 5.0)

    # 9. Promised delivery SLA
    df["promised_delivery_days"] = pd.to_numeric(df["promised_delivery_days"], errors="coerce").fillna(4).clip(lower=1)
    df["discount_pct"] = discount
    df["qty"] = qty

    # Return only the approved dispatch-safe feature matrix
    selected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    return df[selected_cols].copy()


def build_preprocessor() -> ColumnTransformer:
    """Builds numerical scaling + categorical one-hot encoding pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES)
    ])


def candidate(model_type: str = "logistic_regression") -> Pipeline:
    """Constructs a complete scikit-learn preprocessing-and-classifier pipeline."""
    preprocessor = build_preprocessor()

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=0.20,
            max_iter=1000,
            random_state=42
        )
    elif model_type == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=300,
            max_depth=5,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
