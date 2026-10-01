"""
Kestrel Home Returns Risk - Production Online Scoring Service & HTTP API
Implements input validation, shared feature extraction, mathematical log-odds
explanations, economic decision rules, Shield VIP preservation, and graceful failure handling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from model import features, canonical, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.joblib"
INDEX_HTML_PATH = FRONTEND_DIR / "index.html" if (FRONTEND_DIR / "index.html").exists() else BASE_DIR / "index.html"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"

# Economic Constants (from Operations Policy v4.1)
RETURN_COST_INR = 1150.0
CONFIRMATION_CALL_COST_INR = 45.0
CALL_PREVENTION_RATE = 0.35
CALL_BREAK_EVEN_THRESHOLD = CONFIRMATION_CALL_COST_INR / (CALL_PREVENTION_RATE * RETURN_COST_INR)  # ~0.111801 (11.18%)


class OrderPayload(BaseModel):
    order_id: str = Field(..., description="Unique order identifier")
    order_placed_at: str = Field(..., description="Order creation timestamp in IST")
    customer_id: str = Field(..., description="Key into customers.csv")
    sku: str = Field(..., description="Key into products.csv")
    sales_channel: str = Field(default="web", description="app | web | marketplace | partner_outlet")
    payment_mode: str = Field(default="prepaid_upi", description="prepaid_upi | prepaid_card | cod | emi")
    discount_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    qty: int = Field(default=1, ge=1)
    order_value_inr: float = Field(default=0.0, ge=0.0)
    promised_delivery_days: int = Field(default=4, ge=1, le=30)
    delivery_pincode: int = Field(default=0, ge=0)
    is_gift: str = Field(default="N", pattern="^(Y|N|y|n)$")
    customer_prior_orders: int = Field(default=0, ge=0)
    customer_prior_returns: int = Field(default=0, ge=0)
    delivery_note: Optional[str] = Field(default=None)
    last_service_event_type: Optional[str] = Field(default="NONE")
    source: Optional[str] = Field(default="crm")


class Scorer:
    """
    Central scoring and operational reasoning engine.
    Loads reference data and serialized model pipeline once on startup.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model_bundle: Optional[Dict[str, Any]] = None
        self.customers_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.is_ready = False
        self.load_resources()

    def load_resources(self):
        """Loads reference datasets and fitted model pipeline."""
        try:
            if CUSTOMERS_PATH.exists():
                self.customers_df = pd.read_csv(CUSTOMERS_PATH)
            if PRODUCTS_PATH.exists():
                self.products_df = pd.read_csv(PRODUCTS_PATH)

            if self.model_path.exists():
                self.model_bundle = joblib.load(self.model_path)
                self.is_ready = True
                logger.info(f"Loaded production model artifact from {self.model_path}")
            else:
                logger.warning(f"Model artifact not found at {self.model_path}. Service in fallback/training mode.")
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")
            self.is_ready = False

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates payload fields, ranges, and existence."""
        required = ["order_id", "order_placed_at", "customer_id", "sku"]
        for r in required:
            if r not in record or str(record[r]).strip() == "":
                return False, f"Missing required field: {r}"

        # Range checks
        try:
            qty = int(record.get("qty", 1))
            if qty < 1:
                return False, "Quantity must be at least 1."
            disc = float(record.get("discount_pct", 0))
            if disc < 0 or disc > 100:
                return False, "Discount percentage must be between 0 and 100."
        except (ValueError, TypeError):
            return False, "Numeric fields (qty, discount_pct) have invalid formats."

        return True, None

    def explain_factors(self, X_transformed: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computes exact mathematical feature contributions:
        contribution_j = beta_j * (x_j - x_bar_j) in log-odds.
        Groups one-hot encodings to their parent categorical features.
        """
        if not self.model_bundle:
            return []

        coefs = self.model_bundle["classifier_coefs"]
        means = self.model_bundle["feature_means"]
        names = self.model_bundle["transformed_feature_names"]

        # Raw log-odds delta per transformed column
        deltas = coefs * (X_transformed[0] - means)

        # Group one-hot features back to parent fields
        field_contributions: Dict[str, float] = {}

        for name, delta in zip(names, deltas):
            # Check if one-hot categorical (e.g. 'cat__payment_mode_cod')
            if name.startswith("cat__"):
                clean_name = name.replace("cat__", "")
                parent_field = clean_name.split("_")[0]
                field_contributions[parent_field] = field_contributions.get(parent_field, 0.0) + delta
            elif name.startswith("num__"):
                clean_name = name.replace("num__", "")
                field_contributions[clean_name] = field_contributions.get(clean_name, 0.0) + delta
            else:
                field_contributions[name] = field_contributions.get(name, 0.0) + delta

        # Sort by absolute impact
        sorted_factors = sorted(field_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factors = sorted_factors[:4]

        # Readable mappings
        descriptions = {
            "payment_mode": "Payment method ({val}) alters baseline commitment risk.",
            "customer_prior_returns": "Customer return history ({val} prior returns) is a primary risk driver.",
            "prior_return_rate_smooth": "Historical customer return rate indicates repeat return velocity.",
            "promised_delivery_days": "Promised delivery window ({val} days SLA) impacts buyer remorse.",
            "shield_member": "Kestrel Shield membership grants 30-day free returns policy.",
            "family": "Product category ({val}) complexity and installation requirements.",
            "sku": "Specific appliance model line.",
            "discount_pct": "Checkout discount level ({val}% off).",
            "expected_value_inr": "Order basket value (₹{val:,.2f}).",
            "is_gift_num": "Gift designation indicator.",
            "is_missing_address": "Walk-in store order without standard delivery address."
        }

        explanations = []
        for feat, contrib in top_factors:
            direction = "increases" if contrib > 0 else "reduces"
            explanations.append({
                "feature": feat,
                "log_odds_contribution": round(float(contrib), 4),
                "direction": direction,
                "summary": f"{feat.replace('_', ' ').title()} {direction} return risk by {abs(contrib):.3f} log-odds."
            })

        return explanations

    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes prediction, explanations, warnings, and business action selection.
        """
        if not self.is_ready or not self.model_bundle:
            raise HTTPException(
                status_code=503,
                detail="Model service is currently unavailable. Ensure model.joblib is trained and present."
            )

        valid, err = self.validate_record(record)
        if not valid:
            raise HTTPException(status_code=400, detail=err)

        order_id = str(record["order_id"])
        customer_id = str(record["customer_id"])
        sku = str(record["sku"])

        # Warnings container
        warnings: List[str] = []

        # Check references
        known_customer = True
        if self.customers_df is not None:
            if customer_id not in self.customers_df["customer_id"].values:
                warnings.append(f"Customer '{customer_id}' not found in reference database.")
                known_customer = False

        known_sku = True
        if self.products_df is not None:
            if sku not in self.products_df["sku"].values:
                warnings.append(f"SKU '{sku}' not recognized in product catalog.")
                known_sku = False

        # Build feature DataFrame
        single_df = pd.DataFrame([record])
        X_df = features(single_df, self.customers_df, self.products_df)

        pipeline = self.model_bundle["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        X_trans = preprocessor.transform(X_df)
        score = float(classifier.predict_proba(X_trans)[0, 1])

        # Mathematical explanations
        explanation_items = self.explain_factors(X_trans)

        # Check Shield status
        is_shield = False
        if self.customers_df is not None and known_customer:
            cust_row = self.customers_df[self.customers_df["customer_id"] == customer_id]
            if not cust_row.empty and str(cust_row.iloc[0].get("shield_member", "")).upper() == "Y":
                is_shield = True
                warnings.append("Customer is a Kestrel Shield Member (Top 20% Lifetime Value). Preserve relationship: do not hold dispatch without contact.")

        # Business Action Rule
        # Break-even threshold: Score >= 0.1118 (11.18%)
        # Expected Net Benefit = P(return) * 0.35 * 1150 - 45
        estimated_net_call_benefit_inr = round((score * CALL_PREVENTION_RATE * RETURN_COST_INR) - CONFIRMATION_CALL_COST_INR, 2)

        if not known_customer:
            action = "verify_customer_details"
            action_text = "Verify customer account details and address prior to fulfillment."
        elif score >= CALL_BREAK_EVEN_THRESHOLD:
            action = "confirm_before_dispatch"
            action_text = f"Recommend ₹45 pre-dispatch confirmation call. Projected net savings: ₹{max(0.0, estimated_net_call_benefit_inr):.2f}."
        else:
            action = "dispatch_normally"
            action_text = "Return risk is below the 11.18% calling break-even threshold. Dispatch normally."

        # Human-readable reason strings
        reasons = [item["summary"] for item in explanation_items]

        # Advanced Model-Driven Counterfactual Actionable Recourse (Actionable XAI)
        # Evaluates what operational modifications would bring a high-risk order below threshold
        counterfactuals = []
        if score >= CALL_BREAK_EVEN_THRESHOLD:
            # Counterfactual 1: Payment Mode Conversion to Prepaid UPI
            if str(record.get("payment_mode", "")).lower() != "prepaid_upi":
                cf_rec = record.copy()
                cf_rec["payment_mode"] = "prepaid_upi"
                cf_X = features(pd.DataFrame([cf_rec]), self.customers_df, self.products_df)
                cf_score = float(classifier.predict_proba(preprocessor.transform(cf_X))[0, 1])
                counterfactuals.append({
                    "lever": "Convert Payment Mode to Prepaid UPI",
                    "original_risk_pct": round(score * 100, 2),
                    "counterfactual_risk_pct": round(cf_score * 100, 2),
                    "risk_reduction_pct": round((score - cf_score) * 100, 2),
                    "actionable_intervention": "Offer ₹75 instant checkout cashback or WhatsApp payment link to switch COD to UPI."
                })

            # Counterfactual 2: Expedited 2-Day Air SLA
            cur_days = int(record.get("promised_delivery_days", 4))
            if cur_days > 2:
                cf_rec = record.copy()
                cf_rec["promised_delivery_days"] = 2
                cf_X = features(pd.DataFrame([cf_rec]), self.customers_df, self.products_df)
                cf_score = float(classifier.predict_proba(preprocessor.transform(cf_X))[0, 1])
                counterfactuals.append({
                    "lever": "Expedite Delivery SLA to 2 Days",
                    "original_risk_pct": round(score * 100, 2),
                    "counterfactual_risk_pct": round(cf_score * 100, 2),
                    "risk_reduction_pct": round((score - cf_score) * 100, 2),
                    "actionable_intervention": "Route via Priority Air Express courier to eliminate buyer remorse window."
                })

        return {
            "order_id": order_id,
            "score": round(score, 6),
            "return_probability_pct": round(score * 100, 2),
            "action": action,
            "action_recommendation": action_text,
            "automatic_hold": False,
            "call_threshold": round(CALL_BREAK_EVEN_THRESHOLD, 6),
            "estimated_call_net_inr": estimated_net_call_benefit_inr,
            "reasons": reasons,
            "model_factors": explanation_items,
            "counterfactual_recourse": counterfactuals,
            "warnings": warnings,
            "model_version": self.model_bundle.get("model_version", "1.0.0"),
            "audit_metrics": self.model_bundle.get("audit_metrics", {})
        }


# Global Scorer Instance
scorer = Scorer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reload scorer on startup
    scorer.load_resources()
    yield


app = FastAPI(
    title="Kestrel Home Returns Risk Prediction API",
    description="Operational dispatch scoring service, explainability, and policy optimization engine.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check verifying model readiness."""
    return {
        "status": "healthy" if scorer.is_ready else "degraded",
        "model_loaded": scorer.is_ready,
        "model_path": str(MODEL_PATH),
        "version": "1.0.0"
    }


@app.post("/predict")
def predict_endpoint(payload: Dict[str, Any]):
    """
    Single scoring endpoint: Takes an order record as JSON and returns risk score,
    reasons, warnings, and business action.
    """
    return scorer.predict(payload)


@app.get("/model_info")
def model_info():
    """Returns model metadata, audit metrics, and calibration data."""
    if not scorer.is_ready or not scorer.model_bundle:
        raise HTTPException(status_code=503, detail="Model artifact not available.")

    evidence_metrics = {}
    metrics_path = BASE_DIR / "evidence" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            evidence_metrics = json.load(f)

    return {
        "model_version": scorer.model_bundle.get("model_version", "1.0.0"),
        "algorithm": "Regularized Logistic Regression Pipeline",
        "training_orders": scorer.model_bundle.get("training_orders_count", 10504),
        "audit_metrics": scorer.model_bundle.get("audit_metrics", {}),
        "call_break_even_threshold": round(CALL_BREAK_EVEN_THRESHOLD, 6),
        "evidence": evidence_metrics
    }


@app.get("/demo-examples.json")
def get_demo_examples():
    """Returns pre-configured demonstration order records."""
    for p in [BASE_DIR / "frontend" / "demo-examples.json", BASE_DIR / "demo-examples.json"]:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return []


@app.get("/memo.pdf")
def get_memo_pdf(
    return_cost: float = 1150.0,
    call_cost: float = 45.0,
    prevention_rate: float = 0.35
):
    """
    Serves the executive 1-page PDF memorandum.
    If custom economic parameters are provided, generates a dynamically tuned memo.
    """
    import make_memo_pdf
    if return_cost == 1150.0 and call_cost == 45.0 and prevention_rate == 0.35:
        memo_path = BASE_DIR / "memo_to_ritu.pdf"
        if not memo_path.exists():
            make_memo_pdf.build_pdf(memo_path, return_cost, call_cost, prevention_rate)
        return FileResponse(
            memo_path,
            media_type="application/pdf",
            filename="Kestrel_Executive_Memo_to_Ritu.pdf"
        )
    else:
        custom_pdf_path = BASE_DIR / "scratch" / f"memo_custom_{int(return_cost)}_{int(call_cost)}.pdf"
        custom_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        make_memo_pdf.build_pdf(custom_pdf_path, return_cost, call_cost, prevention_rate)
        return FileResponse(
            custom_pdf_path,
            media_type="application/pdf",
            filename="Kestrel_Executive_Memo_Custom.pdf"
        )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silences browser favicon requests cleanly."""
    return Response(status_code=204)


@app.get("/")
def serve_screen():
    """Serves the working HTML screen."""
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH)
    return {"message": "Kestrel Returns Service is running. Open /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn
    print("=========================================================")
    print("🦅 Kestrel Home Appliances - Returns Risk Service")
    print("=========================================================")
    print("Server running on http://127.0.0.1:8000")
    print("Swagger API Docs: http://127.0.0.1:8000/docs")
    print("=========================================================")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
