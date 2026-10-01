import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


class FeatureEngineer:
    """
    Feature Engineering transformer for Kestrel Return Risk Prediction.
    Ensures exact consistency between offline training, batch scoring, and real-time inference.
    """

    CATEGORICAL_COLUMNS = [
        "sales_channel",
        "payment_mode",
        "is_gift",
        "shield_member",
        "family",
        "model_name",
        "city",
        "state",
        "pincode_prefix",
    ]

    NUMERICAL_COLUMNS = [
        "discount_pct",
        "qty",
        "order_value_cleaned",
        "promised_delivery_days",
        "customer_prior_orders",
        "customer_prior_returns",
        "prior_returns_ratio",
        "is_repeat_customer",
        "has_prior_returns",
        "multiple_prior_returns",
        "warranty_months",
        "list_price_inr",
        "is_zero_pincode",
        "is_long_delivery",
        "is_short_delivery",
        "hour",
        "dayofweek",
        "is_weekend",
        "month",
        "is_night_order",
        "has_note",
        "note_is_gate_code",
        "note_is_deliver_neighbour",
        "note_is_call_req",
        "note_is_no_call",
        "note_is_fragile",
        "note_is_security",
        "note_is_morning",
        "note_is_evening",
        "note_is_office",
        "note_is_landmark",
        "unit_price",
        "discount_amount",
        "discount_ratio",
        "is_high_value",
        "is_low_value",
        "cod_and_high_returns",
        "shield_and_high_value",
        "gift_and_long_delivery",
        "cod_and_marketplace",
        "shield_and_cod",
        "shield_and_prior_returns",
        "high_discount_and_cod",
        "is_install_booked",
    ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return cls.CATEGORICAL_COLUMNS + cls.NUMERICAL_COLUMNS

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts all engineered features from cleaned DataFrame.
        """
        df = df.copy()

        # Date features
        if "dt" not in df.columns:
            df["dt"] = pd.to_datetime(df["order_placed_at"], errors="coerce")

        df["hour"] = df["dt"].dt.hour.fillna(12).astype(int)
        df["dayofweek"] = df["dt"].dt.dayofweek.fillna(2).astype(int)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["month"] = df["dt"].dt.month.fillna(6).astype(int)
        df["is_night_order"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

        # Value & Pricing
        qty_safe = np.maximum(df["qty"].fillna(1), 1)
        df["unit_price"] = df["order_value_cleaned"] / qty_safe
        list_price_safe = df["list_price_inr"].fillna(df["unit_price"])
        df["discount_amount"] = (list_price_safe * qty_safe) - df["order_value_cleaned"]
        df["discount_ratio"] = df["discount_pct"].fillna(0) / 100.0
        df["is_high_value"] = (df["order_value_cleaned"] > 15000).astype(int)
        df["is_low_value"] = (df["order_value_cleaned"] < 3000).astype(int)

        # Logistics & Delivery
        days = df["promised_delivery_days"].fillna(4).astype(int)
        df["is_long_delivery"] = (days >= 7).astype(int)
        df["is_short_delivery"] = (days <= 3).astype(int)

        # Customer History
        prior_orders = df["customer_prior_orders"].fillna(0).astype(float)
        prior_returns = df["customer_prior_returns"].fillna(0).astype(float)
        df["prior_returns_ratio"] = prior_returns / (prior_orders + 1e-5)
        df["is_repeat_customer"] = (prior_orders > 0).astype(int)
        df["has_prior_returns"] = (prior_returns > 0).astype(int)
        df["multiple_prior_returns"] = (prior_returns >= 2).astype(int)

        # Delivery Note NLP rules
        notes = df["delivery_note"].fillna("").astype(str)
        df["has_note"] = (df["delivery_note"].notnull() & (notes.str.strip() != "")).astype(int)
        df["note_is_gate_code"] = notes.str.contains("Gate code", case=False).astype(int)
        df["note_is_deliver_neighbour"] = notes.str.contains("Deliver to neighbour", case=False).astype(int)
        df["note_is_call_req"] = notes.str.contains("Call before delivery", case=False).astype(int)
        df["note_is_no_call"] = notes.str.contains("Do not call|WhatsApp only", case=False).astype(int)
        df["note_is_fragile"] = notes.str.contains("Fragile", case=False).astype(int)
        df["note_is_security"] = notes.str.contains("security", case=False).astype(int)
        df["note_is_morning"] = notes.str.contains("morning", case=False).astype(int)
        df["note_is_evening"] = notes.str.contains("6 pm|evening", case=False).astype(int)
        df["note_is_office"] = notes.str.contains("Office address|weekdays", case=False).astype(int)
        df["note_is_landmark"] = notes.str.contains("Landmark", case=False).astype(int)

        # High-risk segment interactions
        is_cod = (df["payment_mode"] == "cod").astype(int)
        is_shield = (df["shield_member"] == "Y").astype(int)
        is_marketplace = (df["sales_channel"] == "marketplace").astype(int)
        is_gift_val = (df["is_gift"] == "Y").astype(int)

        df["cod_and_high_returns"] = (is_cod & df["has_prior_returns"]).astype(int)
        df["shield_and_high_value"] = (is_shield & df["is_high_value"]).astype(int)
        df["gift_and_long_delivery"] = (is_gift_val & df["is_long_delivery"]).astype(int)
        df["cod_and_marketplace"] = (is_cod & is_marketplace).astype(int)
        df["shield_and_cod"] = (is_shield & is_cod).astype(int)
        df["shield_and_prior_returns"] = (is_shield & df["has_prior_returns"]).astype(int)
        df["high_discount_and_cod"] = ((df["discount_pct"] >= 15) & is_cod).astype(int)

        # Service
        if "is_install_booked" not in df.columns:
            df["is_install_booked"] = (df.get("last_service_event_type", "NONE") == "INSTALL_BOOKED").astype(int)

        # Categoricals to string / category
        for c in cls.CATEGORICAL_COLUMNS:
            if c not in df.columns:
                df[c] = "Unknown"
            df[c] = df[c].fillna("Unknown").astype(str)

        for c in cls.NUMERICAL_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0).astype(float)

        all_cols = cls.get_feature_names()
        return df[all_cols]
