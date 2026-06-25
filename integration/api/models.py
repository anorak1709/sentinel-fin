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