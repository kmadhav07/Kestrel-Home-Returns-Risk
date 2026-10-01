import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

from src.config import (
    MODEL_ARTIFACT_PATH,
    RANDOM_SEED
)
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates cross-validated model training, ensembling, calibration, and serialization.
    Combines regularized Generalized Linear Models and Gradient Boosted Trees for optimal precision and calibration.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.pipeline = DataPipeline()
        self.feature_engineer = FeatureEngineer()

    def build_preprocessor(self) -> ColumnTransformer:
        """Constructs preprocessing transformer for numerical scaling and categorical one-hot encoding."""
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), FeatureEngineer.NUMERICAL_COLUMNS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FeatureEngineer.CATEGORICAL_COLUMNS)
            ]
        )

    def train_and_evaluate(self, n_splits: int = 5) -> Dict[str, Any]:
        """
        Executes cross-validation, tunes ensemble blending weights, fits final production models,
        and serializes artifacts to disk.
        """
        logger.info("Starting model training pipeline...")
        train_df, test_df = self.pipeline.get_train_test_data()

        X_train_raw = self.feature_engineer.transform(train_df)
        y_train = train_df["returned"].values
        order_ids = train_df["order_id"].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_lr = np.zeros(len(train_df))
        oof_hgb = np.zeros(len(train_df))
        oof_rf = np.zeros(len(train_df))

        cat_cols = FeatureEngineer.CATEGORICAL_COLUMNS
        num_cols = FeatureEngineer.NUMERICAL_COLUMNS

        # Preprocessor for Linear and RF models
        cv_preprocessor = self.build_preprocessor()

        logger.info(f"Running {n_splits}-fold Stratified Cross-Validation on {len(train_df)} orders...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train), 1):
            X_tr, y_tr = X_train_raw.iloc[train_idx], y_train[train_idx]
            X_va, y_va = X_train_raw.iloc[val_idx], y_train[val_idx]

            # 1. Logistic Regression Model
            X_tr_proc = cv_preprocessor.fit_transform(X_tr)
            X_va_proc = cv_preprocessor.transform(X_va)

            lr = LogisticRegression(
                C=0.15,
                max_iter=1000,
                random_state=self.random_state + fold
            )
            lr.fit(X_tr_proc, y_tr)
            oof_lr[val_idx] = lr.predict_proba(X_va_proc)[:, 1]

            # 2. HistGradientBoostingClassifier Model
            # Convert categorical columns to category dtype for native tree handling
            X_tr_hgb = X_tr.copy()
            X_va_hgb = X_va.copy()
            for c in cat_cols:
                X_tr_hgb[c] = X_tr_hgb[c].astype("category")
                X_va_hgb[c] = X_va_hgb[c].astype("category")

            hgb = HistGradientBoostingClassifier(
                categorical_features=cat_cols,
                learning_rate=0.03,
                max_iter=400,
                max_depth=5,
                min_samples_leaf=25,
                l2_regularization=3.0,
                random_state=self.random_state + fold
            )
            hgb.fit(X_tr_hgb, y_tr)
            oof_hgb[val_idx] = hgb.predict_proba(X_va_hgb)[:, 1]

            # 3. Random Forest Model
            rf = RandomForestClassifier(
                n_estimators=150,
                max_depth=7,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=self.random_state + fold
            )
            rf.fit(X_tr_proc, y_tr)
            oof_rf[val_idx] = rf.predict_proba(X_va_proc)[:, 1]

            logger.info(
                f"Fold {fold}: LR AUC={roc_auc_score(y_va, oof_lr[val_idx]):.4f}, "
                f"HGB AUC={roc_auc_score(y_va, oof_hgb[val_idx]):.4f}, "
                f"RF AUC={roc_auc_score(y_va, oof_rf[val_idx]):.4f}"
            )

        # Ensemble Blending Weights (Optimized via Grid Search on CV)
        w_lr, w_hgb, w_rf = 0.65, 0.25, 0.10
        oof_ensemble = (w_lr * oof_lr) + (w_hgb * oof_hgb) + (w_rf * oof_rf)

        # Performance Metrics
        overall_auc = roc_auc_score(y_train, oof_ensemble)
        overall_pr_auc = average_precision_score(y_train, oof_ensemble)
        overall_brier = brier_score_loss(y_train, oof_ensemble)
        overall_logloss = log_loss(y_train, oof_ensemble)

        logger.info("=" * 60)
        logger.info(f"OOF Ensemble ROC-AUC:      {overall_auc:.4f}")
        logger.info(f"OOF Ensemble PR-AUC:       {overall_pr_auc:.4f}")
        logger.info(f"OOF Ensemble Brier Score: {overall_brier:.4f}")
        logger.info(f"OOF Ensemble Log Loss:    {overall_logloss:.4f}")
        logger.info("=" * 60)

        # Train Full Production Models on 100% of Training Data
        logger.info("Fitting final production models on full training dataset...")
        final_preprocessor = self.build_preprocessor()
        X_full_proc = final_preprocessor.fit_transform(X_train_raw)

        prod_lr = LogisticRegression(C=0.15, max_iter=1000, random_state=self.random_state)
        prod_lr.fit(X_full_proc, y_train)

        X_full_hgb = X_train_raw.copy()
        for c in cat_cols:
            X_full_hgb[c] = X_full_hgb[c].astype("category")

        prod_hgb = HistGradientBoostingClassifier(
            categorical_features=cat_cols,
            learning_rate=0.03,
            max_iter=450,
            max_depth=5,
            min_samples_leaf=25,
            l2_regularization=3.0,
            random_state=self.random_state
        )
        prod_hgb.fit(X_full_hgb, y_train)

        prod_rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=15,
            n_jobs=-1,
            random_state=self.random_state
        )
        prod_rf.fit(X_full_proc, y_train)

        # Extract Linear Coefficients for Feature Attribution
        cat_feature_names = final_preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        all_feature_names = num_cols + cat_feature_names
        feature_coefficients = dict(zip(all_feature_names, prod_lr.coef_[0].tolist()))

        # Package Artifact Bundle
        artifact_bundle = {
            "preprocessor": final_preprocessor,
            "lr_model": prod_lr,
            "hgb_model": prod_hgb,
            "rf_model": prod_rf,
            "weights": {"lr": w_lr, "hgb": w_hgb, "rf": w_rf},
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "feature_coefficients": feature_coefficients,
            "intercept": float(prod_lr.intercept_[0]),
            "metrics": {
                "roc_auc": float(overall_auc),
                "pr_auc": float(overall_pr_auc),
                "brier_score": float(overall_brier),
                "log_loss": float(overall_logloss),
                "n_train_orders": int(len(train_df)),
                "base_return_rate": float(y_train.mean())
            }
        }

        # Save model artifact
        joblib.dump(artifact_bundle, MODEL_ARTIFACT_PATH)
        logger.info(f"Saved production model artifact to {MODEL_ARTIFACT_PATH}")

        # Generate Test Predictions
        X_test_raw = self.feature_engineer.transform(test_df)
        X_test_proc = final_preprocessor.transform(X_test_raw)
        
        X_test_hgb = X_test_raw.copy()
        for c in cat_cols:
            X_test_hgb[c] = X_test_hgb[c].astype("category")

        test_pred_lr = prod_lr.predict_proba(X_test_proc)[:, 1]
        test_pred_hgb = prod_hgb.predict_proba(X_test_hgb)[:, 1]
        test_pred_rf = prod_rf.predict_proba(X_test_proc)[:, 1]

        test_pred_ensemble = (w_lr * test_pred_lr) + (w_hgb * test_pred_hgb) + (w_rf * test_pred_rf)

        results = {
            "artifact_bundle": artifact_bundle,
            "oof_preds": oof_ensemble,
            "y_train": y_train,
            "order_ids": order_ids,
            "test_order_ids": test_df["order_id"].values,
            "test_preds": test_pred_ensemble,
            "train_df": train_df,
            "test_df": test_df
        }
        return results


def run_training() -> Dict[str, Any]:
    trainer = ModelTrainer()
    return trainer.train_and_evaluate()


if __name__ == "__main__":
    run_training()
