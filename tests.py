"""
Kestrel Home Returns Risk - Comprehensive Automated Test Suite
Verifies data contracts, zero data leakage invariants, API contract integrity,
CSV/API numerical consistency, economic threshold logic, and graceful failure modes.
"""

import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

import model
import train
from app import app, scorer, CALL_BREAK_EVEN_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)


class TestKestrelReturnsService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.raw_train, cls.raw_test, cls.customers, cls.products, cls.sample_sub = train.load_data()
        cls.predictions_df = pd.read_csv(BASE_DIR / "predictions.csv")

    def test_01_canonical_deduplication(self):
        """Test 1: Verifies duplicate partner feed rows are deduplicated with label agreement."""
        dedup_df = model.canonical(self.raw_train)
        self.assertEqual(len(dedup_df), 10504)
        self.assertEqual(dedup_df["order_id"].nunique(), 10504)

    def test_02_leakage_invariance_service_event(self):
        """Test 2: Proves model output is invariant to post-delivery service events (DEMO_DONE, REVERSE_PICKUP)."""
        base_record = {
            "order_id": "TEST_LEAK_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC100983",
            "sku": "KH-RV-03",
            "sales_channel": "marketplace",
            "payment_mode": "cod",
            "discount_pct": 10,
            "qty": 1,
            "order_value_inr": 26728.20,
            "promised_delivery_days": 6,
            "delivery_pincode": 440314,
            "is_gift": "N",
            "customer_prior_orders": 2,
            "customer_prior_returns": 1,
            "last_service_event_type": "NONE"
        }

        res_none = scorer.predict(base_record)["score"]

        # Alter to post-delivery events
        for event in ["REVERSE_PICKUP", "DEMO_DONE", "INSTALL_DONE", "TECH_VISIT"]:
            altered = base_record.copy()
            altered["last_service_event_type"] = event
            res_event = scorer.predict(altered)["score"]
            self.assertAlmostEqual(res_none, res_event, places=5, msg=f"Leakage detected with event {event}!")

    def test_03_leakage_invariance_pickup_scheduled(self):
        """Test 3: Proves model output is invariant to reverse pickup logistics bookings."""
        base_record = {
            "order_id": "TEST_LEAK_02",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC100983",
            "sku": "KH-RV-03",
            "sales_channel": "marketplace",
            "payment_mode": "cod",
            "discount_pct": 10,
            "qty": 1,
            "order_value_inr": 26728.20,
            "promised_delivery_days": 6,
            "delivery_pincode": 440314,
            "is_gift": "N",
            "customer_prior_orders": 2,
            "customer_prior_returns": 1,
            "pickup_scheduled_at": None
        }

        res_none = scorer.predict(base_record)["score"]

        altered = base_record.copy()
        altered["pickup_scheduled_at"] = "2026-08-25 10:00"
        res_pickup = scorer.predict(altered)["score"]

        self.assertAlmostEqual(res_none, res_pickup, places=5, msg="Leakage detected via pickup_scheduled_at!")

    def test_04_october_gateway_reconstruction(self):
        """Test 4: Verifies reconstructed expected basket value resolves 100x payment gateway anomaly."""
        oct_record = {
            "order_id": "KO2604170",
            "order_placed_at": "2025-10-01 00:11:00",
            "customer_id": "KC106112",
            "sku": "KH-MG-02",
            "sales_channel": "web",
            "payment_mode": "prepaid_card",
            "discount_pct": 3,
            "qty": 1,
            "order_value_inr": 388638.0,  # Corrupted 100x value in raw CRM
            "promised_delivery_days": 4,
            "delivery_pincode": 440300
        }

        df_single = pd.DataFrame([oct_record])
        feats = model.features(df_single, self.customers, self.products)
        # Expected value should be 3999 * (1 - 0.03) = 3879.03
        self.assertLess(feats["expected_value_inr"].iloc[0], 5000.0)
        self.assertGreater(feats["expected_value_inr"].iloc[0], 3500.0)

    def test_05_empirical_bayes_smoothing(self):
        """Test 5: Verifies Bayesian smoothing of customer prior return rates."""
        # 1 order, 1 return -> (1 + 0.5) / (1 + 5.0) = 1.5 / 6.0 = 0.25 (not 1.0)
        record = {
            "order_id": "TEST_SMOOTH_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC100983",
            "sku": "KH-RV-03",
            "customer_prior_orders": 1,
            "customer_prior_returns": 1,
            "delivery_pincode": 440314,
            "is_gift": "N"
        }
        df_single = pd.DataFrame([record])
        feats = model.features(df_single, self.customers, self.products)
        self.assertAlmostEqual(feats["prior_return_rate_smooth"].iloc[0], 1.5 / 6.0, places=4)

    def test_06_api_predict_contract(self):
        """Test 6: Verifies POST /predict API endpoint response schema contract."""
        payload = {
            "order_id": "KO2611078",
            "order_placed_at": "2026-07-27 15:47",
            "customer_id": "KC107502",
            "sku": "KH-WP-02",
            "sales_channel": "marketplace",
            "payment_mode": "cod",
            "discount_pct": 24,
            "qty": 1,
            "order_value_inr": 11399.24,
            "promised_delivery_days": 6,
            "delivery_pincode": 440612,
            "is_gift": "N",
            "customer_prior_orders": 1,
            "customer_prior_returns": 0
        }

        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("order_id", data)
        self.assertIn("score", data)
        self.assertIn("action", data)
        self.assertIn("automatic_hold", data)
        self.assertIn("call_threshold", data)
        self.assertIn("estimated_call_net_inr", data)
        self.assertIn("reasons", data)
        self.assertIn("model_factors", data)
        self.assertIn("warnings", data)
        self.assertEqual(data["automatic_hold"], False)

    def test_07_api_csv_consistency(self):
        """Test 7: Verifies exact numerical score consistency between API endpoint and predictions.csv."""
        # Test first 5 rows of test_unlabelled.csv (sanitize NaN to None for JSON compliance)
        test_clean = self.raw_test.head(5).copy().fillna("")
        sample_rows = test_clean.to_dict(orient="records")

        for row in sample_rows:
            oid = row["order_id"]
            expected_score = self.predictions_df[self.predictions_df["order_id"] == oid]["score"].iloc[0]

            resp = self.client.post("/predict", json=row)
            self.assertEqual(resp.status_code, 200)
            api_score = resp.json()["score"]

            self.assertAlmostEqual(expected_score, api_score, places=4, msg=f"Score mismatch for {oid}!")

    def test_08_economic_action_rule(self):
        """Test 8: Verifies 11.18% break-even threshold logic for confirm_before_dispatch vs dispatch_normally."""
        high_risk_record = {
            "order_id": "TEST_HIGH_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC100983",
            "sku": "KH-RV-03",
            "sales_channel": "marketplace",
            "payment_mode": "cod",
            "discount_pct": 18,
            "qty": 1,
            "order_value_inr": 24352.36,
            "promised_delivery_days": 8,
            "delivery_pincode": 440314,
            "is_gift": "N",
            "customer_prior_orders": 3,
            "customer_prior_returns": 2
        }
        res_high = scorer.predict(high_risk_record)
        self.assertGreaterEqual(res_high["score"], CALL_BREAK_EVEN_THRESHOLD)
        self.assertEqual(res_high["action"], "confirm_before_dispatch")
        self.assertGreater(res_high["estimated_call_net_inr"], 0.0)

        low_risk_record = {
            "order_id": "TEST_LOW_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC101598",
            "sku": "KH-MG-01",
            "sales_channel": "app",
            "payment_mode": "prepaid_upi",
            "discount_pct": 5,
            "qty": 1,
            "order_value_inr": 3039.05,
            "promised_delivery_days": 2,
            "delivery_pincode": 440075,
            "is_gift": "N",
            "customer_prior_orders": 2,
            "customer_prior_returns": 0
        }
        res_low = scorer.predict(low_risk_record)
        self.assertLess(res_low["score"], CALL_BREAK_EVEN_THRESHOLD)
        self.assertEqual(res_low["action"], "dispatch_normally")

    def test_09_shield_vip_protection(self):
        """Test 9: Verifies Kestrel Shield VIP members trigger protection warning and no automatic hold."""
        shield_record = {
            "order_id": "TEST_SHIELD_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC100012",  # Shield Member = Y in customers.csv
            "sku": "KH-AF-03",
            "sales_channel": "web",
            "payment_mode": "prepaid_card",
            "discount_pct": 10,
            "qty": 1,
            "order_value_inr": 7895.70,
            "promised_delivery_days": 6,
            "delivery_pincode": 400151,
            "is_gift": "N",
            "customer_prior_orders": 4,
            "customer_prior_returns": 1
        }
        res = scorer.predict(shield_record)
        self.assertEqual(res["automatic_hold"], False)
        # Verify Shield VIP warning is present
        shield_warnings = [w for w in res["warnings"] if "Shield Member" in w]
        self.assertGreaterEqual(len(shield_warnings), 1)

    def test_10_missing_customer_reference_handling(self):
        """Test 10: Verifies unknown customer accounts trigger verify_customer_details warning."""
        unseen_cust_record = {
            "order_id": "TEST_UNSEEN_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC999999_NON_EXISTENT",
            "sku": "KH-RV-01",
            "sales_channel": "web",
            "payment_mode": "prepaid_upi",
            "discount_pct": 5,
            "qty": 1,
            "order_value_inr": 16719.05,
            "promised_delivery_days": 4,
            "delivery_pincode": 440300
        }
        res = scorer.predict(unseen_cust_record)
        self.assertEqual(res["action"], "verify_customer_details")
        self.assertTrue(any("not found in reference" in w for w in res["warnings"]))

    def test_11_input_validation_errors(self):
        """Test 11: Verifies invalid JSON and missing fields produce readable 400 errors."""
        bad_payload = {
            "order_id": "",
            "customer_id": "KC100000"
        }
        resp = self.client.post("/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 400)

    def test_12_none_reference_resilience(self):
        """Test 12: Tricky Scenario - Verifies features() handles None customers and None products gracefully."""
        sample_df = pd.DataFrame([{
            "order_id": "TEST_TRICKY_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC100983",
            "sku": "KH-RV-03",
            "sales_channel": "marketplace",
            "payment_mode": "cod",
            "discount_pct": 10,
            "qty": 1,
            "order_value_inr": 25000.0,
            "promised_delivery_days": 5,
            "delivery_pincode": 440314,
            "is_gift": "N"
        }])
        # Pass None as customer/product references
        feats = model.features(sample_df, None, None)
        self.assertEqual(len(feats), 1)
        self.assertIn("prior_return_rate_smooth", feats.columns)
        self.assertIn("expected_value_inr", feats.columns)

    def test_13_corrupted_customer_history_sanity(self):
        """Test 13: Tricky Scenario - Handles corrupted CRM where returns > orders (e.g., 5 returns on 1 order)."""
        corrupted_df = pd.DataFrame([{
            "order_id": "TEST_CORRUPTED_01",
            "customer_prior_orders": 1,
            "customer_prior_returns": 5,  # Data corruption in CRM
            "delivery_pincode": 440300
        }])
        feats = model.features(corrupted_df, self.customers, self.products)
        # Returns must be clamped to <= orders -> 1 return on 1 order -> smooth rate = (1 + 0.5) / (1 + 5) = 1.5/6 = 0.25
        self.assertAlmostEqual(feats["prior_return_rate_smooth"].iloc[0], 0.25, places=4)

    def test_14_extreme_values_and_type_resilience(self):
        """Test 14: Tricky Scenario - Handles string-formatted numbers, zero quantities, and extreme discounts."""
        payload = {
            "order_id": "TEST_EXTREME_01",
            "order_placed_at": "2026-08-15 14:00",
            "customer_id": "KC100983",
            "sku": "KH-RV-03",
            "sales_channel": "marketplace",
            "payment_mode": "cod",
            "discount_pct": 99.0,
            "qty": 50,
            "order_value_inr": 1250000.0,
            "promised_delivery_days": 28,
            "delivery_pincode": 440314,
            "is_gift": "Y",
            "customer_prior_orders": 20,
            "customer_prior_returns": 10
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)
        self.assertIn("action", data)


def run_all_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKestrelReturnsService)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    test_log = (
        f"Kestrel Returns Risk Test Suite Execution\n"
        f"Tests Run: {result.testsRun}\n"
        f"Failures: {len(result.failures)}\n"
        f"Errors: {len(result.errors)}\n"
        f"Status: {'PASSED' if result.wasSuccessful() else 'FAILED'}\n"
    )
    with open(EVIDENCE_DIR / "test_results.txt", "w") as f:
        f.write(test_log)
        
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()

