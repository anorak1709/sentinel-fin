"""Request and response models for the fraud scoring API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """
    Full feature payload expected by the model.
    Mirrors the columns in features_ready.parquet (excluding TransactionID, isFraud, TransactionDT).

    In production this would be assembled by an upstream feature store
    that joins the raw transaction event with cached user/card aggregates.
    For the capstone we accept the full payload and trust upstream callers.
    """
    # Allow arbitrary additional fields — the model's feature set has 440+ columns
    # and we'd rather be permissive than declare every one.
    model_config = {"extra": "allow"}

    TransactionAmt: float = Field(..., gt=0, description="Transaction amount")
    ProductCD: str | None = None
    card1: int | None = None
    card4: str | None = None
    card6: str | None = None


class ScoreRequest(BaseModel):
    """Top-level request payload."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    features: dict[str, Any] = Field(..., description="Feature payload (see TransactionFeatures)")


class ShapContribution(BaseModel):
    """A single feature's contribution to the prediction."""
    feature: str
    value: Any
    contribution: float


class ScoreResponse(BaseModel):
    """Response payload from /score."""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    top_shap_features: list[ShapContribution]
    model_version: str
    scoring_latency_ms: float
    scored_at: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    primary_model: str
    trained_on_rows: int
    best_iteration: int
    metrics: dict[str, Any]
    risk_tiers: dict[str, float]

# Add at the bottom of the existing file

class AlertListItem(BaseModel):
    """Single alert as returned by /alerts (queue view — abbreviated)."""
    transaction_id: str
    user_id: str
    timestamp: str
    final_risk_tier: str
    combined_risk_score: float
    ml_fraud_probability: float
    correlation_boost: float
    mitre_techniques: list[str]
    matched_rule_names: list[str]
    transaction_amount_inr: float | None = None


class AlertListResponse(BaseModel):
    alerts: list[AlertListItem]
    total: int
    limit: int
    offset: int


class AlertDetailResponse(BaseModel):
    """Full alert payload (detail view) — same shape as correlation_alerts.jsonl rows."""
    transaction_id: str
    user_id: str
    timestamp: str
    ml_fraud_probability: float
    ml_risk_tier: str
    correlation_boost: float
    combined_risk_score: float
    final_risk_tier: str
    matched_rules: list[dict[str, Any]]
    mitre_techniques: list[str]
    transaction: dict[str, Any]
    suggested_action: str


class StatsResponse(BaseModel):
    total: int
    by_tier: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    by_mitre: dict[str, int] = {}
    rules_per_alert: dict[str, int] = {}
    score_distribution: dict[str, int] = {}
    by_hour: dict[str, int] = {}