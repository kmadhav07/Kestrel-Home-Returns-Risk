import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    CUSTOMERS_DATA_PATH,
    PRODUCTS_DATA_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Robust data ingestion, cleaning, and preparation pipeline for Kestrel Returns Risk.
    Handles data anomalies, schema merging, and safeguards against data leakage.
    """

    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        customers_path: Optional[str] = None,
        products_path: Optional[str] = None
    ):
        self.train_path = train_path or TRAIN_DATA_PATH
        self.test_path = test_path or TEST_DATA_PATH
        self.customers_path = customers_path or CUSTOMERS_DATA_PATH
        self.products_path = products_path or PRODUCTS_DATA_PATH
        
        self.customers_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None

    def load_reference_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and pre-processes customer and product reference tables."""
        customers = pd.read_csv(self.customers_path)
        products = pd.read_csv(self.products_path)

        # Standardize types
        customers["customer_id"] = customers["customer_id"].astype(str).str.strip()
        customers["city"] = customers["city"].fillna("Unknown").astype(str).str.strip()
        customers["state"] = customers["state"].fillna("Unknown").astype(str).str.strip()
        customers["shield_member"] = customers["shield_member"].fillna("N").astype(str).str.strip().str.upper()
        customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")

        products["sku"] = products["sku"].astype(str).str.strip()
        products["family"] = products["family"].fillna("Unknown").astype(str).str.strip()
        products["model_name"] = products["model_name"].fillna("Unknown").astype(str).str.strip()
        products["list_price_inr"] = products["list_price_inr"].astype(float)
        products["warranty_months"] = products["warranty_months"].fillna(12).astype(int)
        products["launch_date"] = pd.to_datetime(products["launch_date"], errors="coerce")

        self.customers_df = customers
        self.products_df = products
        return customers, products

    def clean_and_repair_orders(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Cleans order records and resolves documented edge cases:
        1. Deduplicates multiple partner feed imports.
        2. Fixes October 2025 payment gateway anomaly (order values 100x higher due to paise error).
        3. Parses dates safely.
        4. Cleans pincode and note text.
        """
        df = df.copy()

        # 1. Deduplication on order_id for training data
        if is_train and "order_id" in df.columns:
            initial_len = len(df)
            df = df.drop_duplicates(subset=["order_id"], keep="first").copy()
            dedup_count = initial_len - len(df)
            if dedup_count > 0:
                logger.info(f"Deduplicated {dedup_count} partner-feed duplicate rows in train.")

        # 2. Date parsing
        df["dt"] = pd.to_datetime(df["order_placed_at"], errors="coerce")

        # 3. Merge reference tables if loaded
        if self.customers_df is None or self.products_df is None:
            self.load_reference_data()

        # Drop any null reference columns from df before merging to prevent _x/_y collision
        ref_cols_to_drop = [c for c in ["city", "state", "signup_date", "shield_member", "family", "model_name", "list_price_inr", "warranty_months", "launch_date"] if c in df.columns and df[c].isna().all()]
        if ref_cols_to_drop:
            df = df.drop(columns=ref_cols_to_drop)

        # Merge customers if missing customer details
        if "shield_member" not in df.columns or "city" not in df.columns:
            df = df.merge(self.customers_df, on="customer_id", how="left")

        # Merge products if missing product details
        if "family" not in df.columns or "list_price_inr" not in df.columns:
            df = df.merge(self.products_df, on="sku", how="left")

        # Fill missing reference info if any customer/sku was unmapped
        df["shield_member"] = df.get("shield_member", pd.Series(["N"] * len(df))).fillna("N")
        df["city"] = df.get("city", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
        df["state"] = df.get("state", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
        df["family"] = df.get("family", pd.Series(["Other"] * len(df))).fillna("Other")
        df["model_name"] = df.get("model_name", pd.Series(["Other"] * len(df))).fillna("Other")
        df["list_price_inr"] = df.get("list_price_inr", pd.Series([None] * len(df))).fillna(df["order_value_inr"] / np.maximum(df["qty"].fillna(1), 1))
        df["warranty_months"] = df.get("warranty_months", pd.Series([12] * len(df))).fillna(12)

        # 4. Repair October 2025 festive season payment gateway value corruption
        # Expected order value: list_price_inr * qty * (1 - discount_pct / 100)
        expected_val = df["list_price_inr"] * df["qty"] * (1.0 - df["discount_pct"] / 100.0)
        df["order_value_cleaned"] = df["order_value_inr"].copy()

        # Check for 100x gateway scaling anomaly in October 2025
        oct_mask = (df["dt"].dt.year == 2025) & (df["dt"].dt.month == 10)
        if oct_mask.any():
            ratio = df.loc[oct_mask, "order_value_inr"] / np.maximum(expected_val[oct_mask], 1.0)
            corrupted_mask = oct_mask & (ratio > 50)
            df.loc[corrupted_mask, "order_value_cleaned"] = df.loc[corrupted_mask, "order_value_inr"] / 100.0
            if is_train:
                logger.info(f"Repaired {corrupted_mask.sum()} October 2025 gateway-corrupted order value records.")

        # 5. Clean delivery pincode (000000 -> walk-in store orders without address)
        df["delivery_pincode_clean"] = df["delivery_pincode"].fillna(0).astype(int)
        df["is_zero_pincode"] = (df["delivery_pincode_clean"] == 0).astype(int)
        df["pincode_str"] = df["delivery_pincode_clean"].astype(str).str.zfill(6)
        df["pincode_prefix"] = df["pincode_str"].str[:2]
        df.loc[df["is_zero_pincode"] == 1, "pincode_prefix"] = "00"

        # 6. Normalize categorical columns
        df["sales_channel"] = df["sales_channel"].fillna("web").astype(str).str.strip().str.lower()
        df["payment_mode"] = df["payment_mode"].fillna("prepaid_upi").astype(str).str.strip().str.lower()
        df["is_gift"] = df["is_gift"].fillna("N").astype(str).str.strip().str.upper()

        # 7. Safe Service Event Encoding (strictly at dispatch snapshot)
        df["is_install_booked"] = (df.get("last_service_event_type", "NONE") == "INSTALL_BOOKED").astype(int)

        return df

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and cleans both train and test datasets."""
        self.load_reference_data()
        
        raw_train = pd.read_csv(self.train_path)
        raw_test = pd.read_csv(self.test_path)
        
        train_clean = self.clean_and_repair_orders(raw_train, is_train=True)
        test_clean = self.clean_and_repair_orders(raw_test, is_train=False)
        
        return train_clean, test_clean
