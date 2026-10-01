from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class OrderInput(BaseModel):
    order_id: Optional[str] = Field(default="KO_SAMPLE_001", description="Order ID")
    order_placed_at: Optional[str] = Field(default="2026-10-01 14:30", description="Order placed datetime (IST)")
    customer_id: Optional[str] = Field(default="KC105000", description="Customer ID")
    sku: Optional[str] = Field(default="KH-RV-03", description="Product SKU")
    sales_channel: Optional[str] = Field(default="marketplace", description="app | web | marketplace | partner_outlet")
    payment_mode: Optional[str] = Field(default="cod", description="prepaid_upi | prepaid_card | cod | emi")
    discount_pct: Optional[float] = Field(default=15.0, description="Discount percentage at checkout")
    qty: Optional[int] = Field(default=1, description="Quantity")
    order_value_inr: Optional[float] = Field(default=25243.30, description="Order value stored in INR")
    promised_delivery_days: Optional[int] = Field(default=8, description="Promised delivery days at checkout")
    delivery_pincode: Optional[int] = Field(default=440500, description="Delivery Pincode (0 or 000000 for walk-in)")
    is_gift: Optional[str] = Field(default="N", description="Y / N")
    customer_prior_orders: Optional[int] = Field(default=3, description="Customer prior orders count")
    customer_prior_returns: Optional[int] = Field(default=2, description="Customer prior returns count")
    delivery_note: Optional[str] = Field(default="Call before delivery", description="Delivery note text")
    last_service_event_type: Optional[str] = Field(default="NONE", description="INSTALL_BOOKED | NONE")
    source: Optional[str] = Field(default="crm", description="crm | partner_feed")

    # Optional explicit customer/product fields if not in database
    city: Optional[str] = Field(default=None, description="Customer city")
    state: Optional[str] = Field(default=None, description="Customer state")
    shield_member: Optional[str] = Field(default=None, description="Y / N")
    family: Optional[str] = Field(default=None, description="Product family")
    model_name: Optional[str] = Field(default=None, description="Product model name")
    list_price_inr: Optional[float] = Field(default=None, description="Product list price in INR")
    warranty_months: Optional[int] = Field(default=None, description="Warranty period in months")


class RiskFactor(BaseModel):
    factor: str
    impact: str
    detail: str


class EconomicsDetail(BaseModel):
    expected_return_cost_inr: float
    call_cost_inr: float
    net_savings_if_called_inr: float
    roi_on_confirmation_call: float


class PredictionResponse(BaseModel):
    order_id: str
    return_probability: float
    return_probability_pct: float
    risk_tier: str
    risk_badge: str
    recommended_action: str
    action_description: str
    human_readable_reasons: List[str]
    risk_drivers: List[RiskFactor]
    mitigating_factors: List[RiskFactor]
    economics: EconomicsDetail


class BatchPredictRequest(BaseModel):
    orders: List[OrderInput]


class BatchPredictResponse(BaseModel):
    total_orders: int
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class PolicySimulationRequest(BaseModel):
    call_threshold: float = Field(default=0.115, ge=0.01, le=0.99)
    high_risk_threshold: float = Field(default=0.280, ge=0.01, le=0.99)
    custom_return_cost: Optional[float] = Field(default=1150.0)
    custom_call_cost: Optional[float] = Field(default=45.0)
