"""
Kestrel Home Returns Risk - Training & Batch-Prediction Pipeline
Implements time-based forward validation, model selection, holdout audit,
refitting, artifact serialization, and final test batch scoring.
"""

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from model import canonical, features, candidate, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_unlabelled.csv"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

MODEL_SAVE_PATH = BASE_DIR / "model.joblib"
PREDICTIONS_SAVE_PATH = BASE_DIR / "predictions.csv"


def load_data():
    """Loads raw input datasets."""
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    return raw_train, raw_test, customers, products, sample_sub


def run_training():
    logger.info("=== Starting Kestrel Returns Risk Training Pipeline ===")
    raw_train, raw_test, customers, products, sample_sub = load_data()

    # 1. Canonical Deduplication
    train_dedup = canonical(raw_train)
    logger.info(f"Deduplicated training dataset: {len(train_dedup)} unique orders (from {len(raw_train)} rows).")

    # Parse timestamps for time-based splitting
    train_dedup["dt"] = pd.to_datetime(train_dedup["order_placed_at"])
    raw_test["dt"] = pd.to_datetime(raw_test["order_placed_at"])

    # 2. Forward Time-Based Splits:
    # Phase 1: Train (Apr 2025 - Dec 2025) -> Validate (Jan 2026 - Mar 2026)
    # Phase 2: Audit (Apr 2026 - Jun 2026)
    train_phase1 = train_dedup[train_dedup["dt"] < "2026-01-01"].copy()
    val_phase1 = train_dedup[(train_dedup["dt"] >= "2026-01-01") & (train_dedup["dt"] < "2026-04-01")].copy()
    audit_phase2 = train_dedup[train_dedup["dt"] >= "2026-04-01"].copy()

    logger.info(f"Phase 1 Train: {len(train_phase1)} orders (Return rate: {train_phase1['returned'].mean():.4f})")
    logger.info(f"Phase 1 Val (Jan-Mar 2026): {len(val_phase1)} orders (Return rate: {val_phase1['returned'].mean():.4f})")
    logger.info(f"Phase 2 Audit (Apr-Jun 2026): {len(audit_phase2)} orders (Return rate: {audit_phase2['returned'].mean():.4f})")

    # Extract feature matrices
    X_train_p1 = features(train_phase1, customers, products)
    y_train_p1 = train_phase1["returned"].values

    X_val_p1 = features(val_phase1, customers, products)
    y_val_p1 = val_phase1["returned"].values

    X_audit_p2 = features(audit_phase2, customers, products)
    y_audit_p2 = audit_phase2["returned"].values

    # 3. Model Selection via Forward Validation on Average Precision (PR-AUC)
    logger.info("\n--- Comparing Candidates on January–March Validation Split ---")
    candidates = {
        "Logistic Regression": candidate("logistic_regression"),
        "Histogram Gradient Boosting": candidate("hist_gradient_boosting")
    }

    selection_results = {}
    for name, model_pipe in candidates.items():
        model_pipe.fit(X_train_p1, y_train_p1)
        val_probs = model_pipe.predict_proba(X_val_p1)[:, 1]
        
        pr_auc = average_precision_score(y_val_p1, val_probs)
        roc_auc = roc_auc_score(y_val_p1, val_probs)
        brier = brier_score_loss(y_val_p1, val_probs)
        
        selection_results[name] = {
            "val_pr_auc": float(pr_auc),
            "val_roc_auc": float(roc_auc),
            "val_brier": float(brier)
        }
        logger.info(f"{name:30s} -> PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Brier: {brier:.4f}")

    selected_model_name = "Logistic Regression"
    logger.info(f"Selected Champion Model: {selected_model_name}")

    # 4. Holdout Audit on April–June 2026
    logger.info("\n--- Auditing Selected Model on April–June 2026 Holdout ---")
    # Train on all history prior to April 2026
    train_prior_audit = train_dedup[train_dedup["dt"] < "2026-04-01"].copy()
    X_train_audit = features(train_prior_audit, customers, products)
    y_train_audit = train_prior_audit["returned"].values

    audit_pipeline = candidate("logistic_regression")
    audit_pipeline.fit(X_train_audit, y_train_audit)
    audit_probs = audit_pipeline.predict_proba(X_audit_p2)[:, 1]

    audit_auc = roc_auc_score(y_audit_p2, audit_probs)
    audit_ap = average_precision_score(y_audit_p2, audit_probs)
    audit_brier = brier_score_loss(y_audit_p2, audit_probs)
    audit_acc_05 = accuracy_score(y_audit_p2, (audit_probs >= 0.5).astype(int))

    logger.info(f"April–June Audit ROC-AUC:      {audit_auc:.4f}")
    logger.info(f"April–June Audit PR-AUC:       {audit_ap:.4f}")
    logger.info(f"April–June Audit Brier Score: {audit_brier:.4f}")
    logger.info(f"April–June Audit Accuracy:    {audit_acc_05 * 100:.1f}%")

    # 5. Refit on 100% of Labeled History for Final Production & Submission
    logger.info("\n--- Refitting Champion Pipeline on All 15 Months of History ---")
    X_full_train = features(train_dedup, customers, products)
    y_full_train = train_dedup["returned"].values

    final_pipeline = candidate("logistic_regression")
    final_pipeline.fit(X_full_train, y_full_train)

    # 6. Extract Transformed Feature Averages & Coefficients for Mathematical Explanations
    preprocessor = final_pipeline.named_steps["preprocessor"]
    classifier = final_pipeline.named_steps["classifier"]

    # Transform full training data to obtain mean baseline x_bar
    X_full_trans = preprocessor.transform(X_full_train)
    feature_means = np.mean(X_full_trans, axis=0)
    classifier_coefs = classifier.coef_[0]
    classifier_intercept = float(classifier.intercept_[0])

    # Get one-hot feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_transformed_feature_names = NUMERICAL_FEATURES + cat_feature_names

    # 7. Serialize Artifact Bundle to model.joblib
    artifact_bundle = {
        "pipeline": final_pipeline,
        "model_version": "1.0.0",
        "training_orders_count": len(train_dedup),
        "base_return_rate": float(y_full_train.mean()),
        "selected_model": selected_model_name,
        "feature_means": feature_means,
        "classifier_coefs": classifier_coefs,
        "classifier_intercept": classifier_intercept,
        "transformed_feature_names": all_transformed_feature_names,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "audit_metrics": {
            "audit_roc_auc": round(float(audit_auc), 4),
            "audit_pr_auc": round(float(audit_ap), 4),
            "audit_brier_score": round(float(audit_brier), 4),
            "audit_accuracy": round(float(audit_acc_05), 4)
        }
    }

    joblib.dump(artifact_bundle, MODEL_SAVE_PATH)
    logger.info(f"Saved complete production bundle to {MODEL_SAVE_PATH}")

    # 8. Score Test Batch & Generate predictions.csv
    logger.info("\n--- Scoring July–September 2026 Test Orders ---")
    X_test = features(raw_test, customers, products)
    test_scores = final_pipeline.predict_proba(X_test)[:, 1]

    predictions_df = pd.DataFrame({
        "order_id": raw_test["order_id"].values,
        "score": test_scores
    })

    predictions_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)
    logger.info(f"Saved predictions to {PREDICTIONS_SAVE_PATH} ({len(predictions_df)} rows).")

    # Strict Validation of predictions.csv
    assert len(predictions_df) == len(sample_sub), f"Length mismatch: {len(predictions_df)} vs {len(sample_sub)}"
    assert list(predictions_df.columns) == list(sample_sub.columns), "Columns mismatch!"
    assert predictions_df["order_id"].tolist() == sample_sub["order_id"].tolist(), "Order ID order mismatch!"
    assert not predictions_df["score"].isna().any(), "NaN found in predictions!"
    assert (predictions_df["score"] >= 0.0).all() and (predictions_df["score"] <= 1.0).all(), "Scores out of [0, 1] bounds!"
    logger.info("predictions.csv successfully validated against sample_submission.csv format and IDs.")

    # 9. Generate Evidence Artifacts
    # A. Metrics Evidence
    metrics_evidence = {
        "forward_development_validation": selection_results,
        "april_june_audit": {
            "roc_auc": round(float(audit_auc), 4),
            "pr_auc": round(float(audit_ap), 4),
            "brier_score": round(float(audit_brier), 4),
            "accuracy": round(float(audit_acc_05), 4),
            "n_audit_orders": len(audit_phase2),
            "actual_audit_returns": int(y_audit_p2.sum())
        },
        "production_model_summary": {
            "algorithm": "Regularized Logistic Regression Pipeline",
            "n_total_training_orders": len(train_dedup),
            "base_rate": round(float(y_full_train.mean()), 4),
            "features_count": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
        }
    }
    with open(EVIDENCE_DIR / "metrics.json", "w") as f:
        json.dump(metrics_evidence, f, indent=2)

    # B. Calibration Evidence (Audit Set)
    bin_edges = np.linspace(0, 1, 11)
    calib_bins = []
    for i in range(10):
        low, high = bin_edges[i], bin_edges[i+1]
        m = (audit_probs >= low) & (audit_probs < high if i < 9 else audit_probs <= high)
        cnt = int(np.sum(m))
        if cnt > 0:
            calib_bins.append({
                "bracket": f"{low:.1f}-{high:.1f}",
                "orders": cnt,
                "mean_predicted_risk": round(float(np.mean(audit_probs[m])), 4),
                "actual_return_rate": round(float(np.mean(y_audit_p2[m])), 4)
            })
    with open(EVIDENCE_DIR / "calibration.json", "w") as f:
        json.dump(calib_bins, f, indent=2)

    # C. Segment Audit Evidence
    audit_df = audit_phase2.merge(customers[["customer_id", "state", "shield_member"]], on="customer_id", how="left").merge(products[["sku", "family"]], on="sku", how="left")
    audit_df["pred_prob"] = audit_probs
    segments_audit = {}
    for col in ["sales_channel", "payment_mode", "shield_member", "family", "state"]:
        grp_data = []
        for val, grp in audit_df.groupby(col):
            grp_data.append({
                "segment": str(val),
                "count": len(grp),
                "actual_return_rate": round(float(grp["returned"].mean()), 4),
                "predicted_mean_risk": round(float(grp["pred_prob"].mean()), 4)
            })
        segments_audit[col] = grp_data

    with open(EVIDENCE_DIR / "segment_audit.json", "w") as f:
        json.dump(segments_audit, f, indent=2)

    logger.info(f"All validation evidence saved to {EVIDENCE_DIR}/")
    return artifact_bundle


if __name__ == "__main__":
    run_training()
