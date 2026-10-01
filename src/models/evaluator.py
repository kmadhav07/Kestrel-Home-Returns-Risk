import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from src.config import METRICS_REPORT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation suite for Kestrel Return Risk Prediction.
    Provides multi-dimensional slice analysis, calibration assessment, and operational metric reports.
    """

    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray, df: pd.DataFrame):
        self.y_true = np.array(y_true, dtype=int)
        self.y_prob = np.array(y_prob, dtype=float)
        self.df = df.copy()
        self.df["y_true"] = self.y_true
        self.df["y_prob"] = self.y_prob

    def evaluate_global_metrics(self) -> Dict[str, Any]:
        """Calculates global ranking, discrimination, and calibration metrics."""
        auc = roc_auc_score(self.y_true, self.y_prob)
        pr_auc = average_precision_score(self.y_true, self.y_prob)
        brier = brier_score_loss(self.y_true, self.y_prob)
        ll = log_loss(self.y_true, self.y_prob)
        base_rate = float(np.mean(self.y_true))

        # Precision-Recall curve points
        precisions, recalls, thresholds = precision_recall_curve(self.y_true, self.y_prob)
        
        # Best F1 threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_f1_thresh = float(thresholds[best_f1_idx]) if best_f1_idx < len(thresholds) else 0.5
        best_f1 = float(f1_scores[best_f1_idx])

        return {
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 4),
            "log_loss": round(float(ll), 4),
            "base_return_rate": round(base_rate, 4),
            "best_f1_threshold": round(best_f1_thresh, 4),
            "best_f1_score": round(best_f1, 4),
            "total_orders_evaluated": len(self.y_true),
            "total_returns_evaluated": int(np.sum(self.y_true))
        }

    def evaluate_threshold(self, threshold: float) -> Dict[str, Any]:
        """Evaluates binary classification metrics and confusion matrix at a given threshold."""
        y_pred = (self.y_prob >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(self.y_true, y_pred, zero_division=0)
        recall = recall_score(self.y_true, y_pred, zero_division=0)
        f1 = f1_score(self.y_true, y_pred, zero_division=0)
        acc = accuracy_score(self.y_true, y_pred)
        flagged_count = int(np.sum(y_pred))
        flagged_pct = float(flagged_count / len(self.y_true))

        return {
            "threshold": round(threshold, 4),
            "flagged_orders": flagged_count,
            "flagged_pct": round(flagged_pct * 100, 2),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "accuracy": round(float(acc), 4)
        }

    def evaluate_calibration(self, n_bins: int = 10) -> List[Dict[str, Any]]:
        """Binned calibration analysis (observed vs predicted probability)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_data = []

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i+1]
            mask = (self.y_prob >= low) & (self.y_prob < high if i < n_bins - 1 else self.y_prob <= high)
            count = int(np.sum(mask))
            if count > 0:
                mean_pred = float(np.mean(self.y_prob[mask]))
                mean_true = float(np.mean(self.y_true[mask]))
                actual_returns = int(np.sum(self.y_true[mask]))
            else:
                mean_pred = float((low + high) / 2)
                mean_true = 0.0
                actual_returns = 0

            calibration_data.append({
                "bin_range": f"{low:.1f}-{high:.1f}",
                "order_count": count,
                "mean_predicted_risk": round(mean_pred, 4),
                "actual_return_rate": round(mean_true, 4),
                "actual_returns": actual_returns
            })

        return calibration_data

    def evaluate_slices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Calculates granular slice performance across key business dimensions."""
        slice_dimensions = [
            ("sales_channel", "Sales Channel"),
            ("payment_mode", "Payment Mode"),
            ("shield_member", "Shield Membership"),
            ("family", "Product Family"),
            ("state", "Customer State")
        ]

        slices_report = {}

        for col, display_name in slice_dimensions:
            if col not in self.df.columns:
                continue

            slice_items = []
            for val, grp in self.df.groupby(col):
                if len(grp) < 10:
                    continue
                
                y_t = grp["y_true"].values
                y_p = grp["y_prob"].values
                
                try:
                    auc = roc_auc_score(y_t, y_p) if len(np.unique(y_t)) > 1 else 0.5
                except Exception:
                    auc = 0.5

                ap = average_precision_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.mean(y_t)
                brier = brier_score_loss(y_t, y_p)

                slice_items.append({
                    "segment": str(val),
                    "order_count": len(grp),
                    "actual_return_rate": round(float(y_t.mean()), 4),
                    "mean_predicted_risk": round(float(y_p.mean()), 4),
                    "roc_auc": round(float(auc), 4),
                    "pr_auc": round(float(ap), 4),
                    "brier_score": round(float(brier), 4)
                })

            slice_items.sort(key=lambda x: x["order_count"], reverse=True)
            slices_report[display_name] = slice_items

        return slices_report

    def generate_full_report(self) -> Dict[str, Any]:
        """Generates full evaluation report dictionary and writes to JSON."""
        global_metrics = self.evaluate_global_metrics()
        
        # Test key thresholds
        thresholds_to_test = [0.10, 0.115, 0.15, 0.20, 0.28, 0.35, 0.50]
        threshold_evals = [self.evaluate_threshold(t) for t in thresholds_to_test]
        
        calibration = self.evaluate_calibration(n_bins=10)
        slices = self.evaluate_slices()

        report = {
            "summary": global_metrics,
            "threshold_performance": threshold_evals,
            "calibration_curve": calibration,
            "slice_analysis": slices
        }

        with open(METRICS_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Generated comprehensive evaluation report at {METRICS_REPORT_PATH}")
        return report
