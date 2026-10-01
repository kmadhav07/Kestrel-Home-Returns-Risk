import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.config import (
    MODEL_ARTIFACT_PATH,
    METRICS_REPORT_PATH,
    FRONTEND_DIR,
    COST_RETURN_ALL_IN_INR,
    COST_CONFIRMATION_CALL_INR,
    CALL_RETURN_REDUCTION_RATE,
    THRESHOLD_CALL,
    THRESHOLD_HIGH_RISK
)
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.models.explainer import ModelExplainer
from src.models.trainer import ModelTrainer
from src.economics.policy_optimizer import PolicyOptimizer
from src.api.schemas import (
    OrderInput,
    PredictionResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PolicySimulationRequest
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global state
model_bundle: Dict[str, Any] = {}
explainer: ModelExplainer = None
data_pipeline: DataPipeline = None


def get_or_load_model():
    """Loads the model artifact into memory or runs training if not present."""
    global model_bundle, explainer, data_pipeline
    
    if data_pipeline is None:
        data_pipeline = DataPipeline()
        data_pipeline.load_reference_data()

    if not model_bundle:
        if not MODEL_ARTIFACT_PATH.exists():
            logger.info("Model artifact not found. Initializing training...")
            trainer = ModelTrainer()
            trainer.train_and_evaluate()

        logger.info(f"Loading model bundle from {MODEL_ARTIFACT_PATH}...")
        model_bundle = joblib.load(MODEL_ARTIFACT_PATH)
        explainer = ModelExplainer(model_bundle)
        logger.info("Model bundle loaded successfully.")

    return model_bundle, explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        get_or_load_model()
    except Exception as e:
        logger.error(f"Error initializing model on startup: {e}")
    yield
    # Shutdown


app = FastAPI(
    title="Kestrel Home Returns Risk Prediction Service",
    description="Pre-dispatch risk scoring, explainability, and policy optimization engine for Kestrel Home Appliances.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local web interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint confirming service status and model readiness."""
    has_model = MODEL_ARTIFACT_PATH.exists()
    return {
        "status": "healthy",
        "service": "Kestrel Returns Risk Engine",
        "model_loaded": bool(model_bundle),
        "model_artifact_exists": has_model,
        "version": "1.0.0"
    }


def score_single_order_record(order_dict: Dict[str, Any]) -> PredictionResponse:
    """Internal helper to score and explain a single order record."""
    bundle, exp = get_or_load_model()
    
    # Clean and merge reference fields if missing
    df_raw = pd.DataFrame([order_dict])
    cleaned_df = data_pipeline.clean_and_repair_orders(df_raw, is_train=False)
    
    # Feature extraction
    feat_df = FeatureEngineer.transform(cleaned_df)
    
    # Model inference
    preproc = bundle["preprocessor"]
    X_proc = preproc.transform(feat_df)
    
    cat_cols = bundle["cat_cols"]
    X_hgb = feat_df.copy()
    for c in cat_cols:
        X_hgb[c] = X_hgb[c].astype("category")
        
    p_lr = bundle["lr_model"].predict_proba(X_proc)[0, 1]
    p_hgb = bundle["hgb_model"].predict_proba(X_hgb)[0, 1]
    p_rf = bundle["rf_model"].predict_proba(X_proc)[0, 1]
    
    w = bundle["weights"]
    ensemble_score = float((w["lr"] * p_lr) + (w["hgb"] * p_hgb) + (w["rf"] * p_rf))
    
    # Generate human-readable operational explanation
    explanation = exp.explain_order(cleaned_df.iloc[0].to_dict(), ensemble_score)
    
    return PredictionResponse(
        order_id=str(order_dict.get("order_id", "UNKNOWN")),
        return_probability=explanation["score"],
        return_probability_pct=explanation["return_probability_pct"],
        risk_tier=explanation["risk_tier"],
        risk_badge=explanation["risk_badge"],
        recommended_action=explanation["recommended_action"],
        action_description=explanation["action_description"],
        human_readable_reasons=explanation["human_readable_reasons"],
        risk_drivers=explanation["risk_drivers"],
        mitigating_factors=explanation["mitigating_factors"],
        economics=explanation["economics"]
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict_single_order(order: OrderInput):
    """
    Main API endpoint: Takes a single order record as JSON and returns return risk score,
    risk tier, human-readable operational reasons, and recommended dispatch actions.
    """
    try:
        return score_single_order_record(order.model_dump() if hasattr(order, 'model_dump') else order.dict())
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/batch_predict", response_model=BatchPredictResponse)
def predict_batch_orders(request: BatchPredictRequest):
    """Batch prediction endpoint for lists of order records."""
    try:
        orders_list = request.orders
        predictions = [score_single_order_record(order.model_dump() if hasattr(order, 'model_dump') else order.dict()) for order in orders_list]
        
        scores = [p.return_probability for p in predictions]
        tiers = [p.risk_tier for p in predictions]
        actions = [p.recommended_action for p in predictions]
        
        summary = {
            "total_orders": len(predictions),
            "mean_risk_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "high_risk_count": tiers.count("HIGH_RISK") + tiers.count("CRITICAL_RISK"),
            "call_recommended_count": sum(1 for a in actions if "CALL" in a),
            "auto_dispatch_count": actions.count("AUTO_DISPATCH"),
            "total_expected_return_cost_inr": round(sum(p.economics.expected_return_cost_inr for p in predictions), 2),
            "total_net_savings_if_called_inr": round(sum(p.economics.net_savings_if_called_inr for p in predictions), 2)
        }
        
        return BatchPredictResponse(
            total_orders=len(predictions),
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.get("/api/model_info")
def get_model_info():
    """Returns model architecture details, cross-validation metrics, and feature weights."""
    bundle, _ = get_or_load_model()
    
    report_data = {}
    if METRICS_REPORT_PATH.exists():
        with open(METRICS_REPORT_PATH, "r") as f:
            report_data = json.load(f)
            
    # Sort top feature weights
    coefs = bundle.get("feature_coefficients", {})
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
    
    return {
        "model_type": "Tri-Model Calibrated Ensemble (Regularized GLM + HistGradientBoosting + Random Forest)",
        "weights": bundle["weights"],
        "metrics": bundle.get("metrics", {}),
        "evaluation_report": report_data,
        "top_feature_coefficients": [{"feature": k, "coefficient": round(v, 4)} for k, v in sorted_coefs],
        "operational_costs": {
            "return_cost_inr": COST_RETURN_ALL_IN_INR,
            "confirmation_call_cost_inr": COST_CONFIRMATION_CALL_INR,
            "call_reduction_rate": CALL_RETURN_REDUCTION_RATE
        }
    }


@app.post("/api/simulate_policy")
def simulate_policy(req: PolicySimulationRequest):
    """Simulates economic return and cost outcomes across customizable thresholds."""
    try:
        # Load train data for simulation
        train_clean, _ = data_pipeline.get_train_test_data()
        y_train = train_clean["returned"].values
        order_values = train_clean["order_value_cleaned"].values
        shield_flags = train_clean["shield_member"].values

        bundle, _ = get_or_load_model()
        X_train_raw = FeatureEngineer.transform(train_clean)
        X_proc = bundle["preprocessor"].transform(X_train_raw)
        
        cat_cols = bundle["cat_cols"]
        X_hgb = X_train_raw.copy()
        for c in cat_cols:
            X_hgb[c] = X_hgb[c].astype("category")

        p_lr = bundle["lr_model"].predict_proba(X_proc)[:, 1]
        p_hgb = bundle["hgb_model"].predict_proba(X_hgb)[:, 1]
        p_rf = bundle["rf_model"].predict_proba(X_proc)[:, 1]
        w = bundle["weights"]
        y_prob = (w["lr"] * p_lr) + (w["hgb"] * p_hgb) + (w["rf"] * p_rf)

        optimizer = PolicyOptimizer(
            return_cost=req.custom_return_cost or COST_RETURN_ALL_IN_INR,
            call_cost=req.custom_call_cost or COST_CONFIRMATION_CALL_INR
        )

        simulations = optimizer.run_all_simulations(y_train, y_prob, order_values, shield_flags)
        return simulations
    except Exception as e:
        logger.error(f"Policy simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample_orders")
def get_sample_orders():
    """Provides representative real-world test cases for instant interactive UI evaluation."""
    return [
        {
            "label": "High-Risk: Cash on Delivery with Prior Returns (Robot Vacuum)",
            "data": {
                "order_id": "TEST_HIGH_RISK_01",
                "order_placed_at": "2026-10-01 21:15",
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
                "customer_prior_returns": 2,
                "delivery_note": "Deliver to neighbour flat 7A if not home",
                "last_service_event_type": "NONE",
                "source": "crm"
            }
        },
        {
            "label": "Moderate-Risk: Kestrel Shield VIP Member (Air Fryer Max)",
            "data": {
                "order_id": "TEST_SHIELD_VIP_02",
                "order_placed_at": "2026-10-01 11:20",
                "customer_id": "KC100012",
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
                "customer_prior_returns": 1,
                "delivery_note": "Leave with security",
                "last_service_event_type": "INSTALL_BOOKED",
                "source": "crm"
            }
        },
        {
            "label": "Low-Risk: Express Prepaid UPI (Mixer Grinder Pro)",
            "data": {
                "order_id": "TEST_LOW_RISK_03",
                "order_placed_at": "2026-10-01 10:14",
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
                "customer_prior_returns": 0,
                "delivery_note": "Ring twice",
                "last_service_event_type": "NONE",
                "source": "crm"
            }
        },
        {
            "label": "Walk-in Store Purchase: Partner Outlet (Ceiling Fan)",
            "data": {
                "order_id": "TEST_WALKIN_04",
                "order_placed_at": "2026-10-01 16:40",
                "customer_id": "KC103801",
                "sku": "KH-CF-02",
                "sales_channel": "partner_outlet",
                "payment_mode": "prepaid_upi",
                "discount_pct": 10,
                "qty": 1,
                "order_value_inr": 3149.10,
                "promised_delivery_days": 3,
                "delivery_pincode": 0,
                "is_gift": "N",
                "customer_prior_orders": 1,
                "customer_prior_returns": 0,
                "delivery_note": "Fragile - handle with care",
                "last_service_event_type": "NONE",
                "source": "crm"
            }
        }
    ]


# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_dashboard():
    """Serves the interactive web UI dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Kestrel Return Risk API is running. Access API docs at /docs"}
