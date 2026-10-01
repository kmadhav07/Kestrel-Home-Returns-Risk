import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Data Files
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test_unlabelled.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
CUSTOMERS_DATA_PATH = DATA_DIR / "customers.csv"
PRODUCTS_DATA_PATH = DATA_DIR / "products.csv"
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"
PREDICTIONS_OUTPUT_PATH = BASE_DIR / "predictions.csv"

# Economic & Cost Parameters (from ops-policy.pdf v4.1)
COST_RETURN_ALL_IN_INR = 1150.0       # Average all-in cost of a return (reverse pickup, inspection, repack, write-down)
COST_CONFIRMATION_CALL_INR = 45.0     # Cost per completed outsourced pre-dispatch confirmation call
CALL_RETURN_REDUCTION_RATE = 0.35     # Confirmation call prevents ~35% of returns that would otherwise happen
HOLD_CANCEL_RATE_24H = 0.12           # Orders held >24 hours cancelled by customer ~12% of the time
AVG_GROSS_MARGIN_PCT = 0.35           # Average gross margin on appliances

# Operational Decision Thresholds
# Call Threshold: P(return) >= 45 / (0.35 * 1150) = ~0.112 (positive expected value for call)
THRESHOLD_CALL = 0.115
# Audit / High Risk Threshold: Extreme return probability warranting mandatory priority verification
THRESHOLD_HIGH_RISK = 0.280
# Critical Risk Threshold: Very high risk of return (>50%)
THRESHOLD_CRITICAL = 0.500

# Random Seed for Reproducibility
RANDOM_SEED = 42
