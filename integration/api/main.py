"""Sentinel-Fin fraud scoring API."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ScoreRequest,
    ScoreResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .scorer import FraudScorer, initialize_scorer, get_scorer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sentinel-fin")


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "ml" / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at app startup; release at shutdown."""
    log.info(f"Loading artifacts from {ARTIFACTS_DIR}")
    scorer = initialize_scorer(ARTIFACTS_DIR)
    log.info(
        f"Loaded model v{scorer.metadata.get('model_version')} "
        f"(trained on {scorer.metadata.get('trained_on_rows'):,} rows, "
        f"PR-AUC {scorer.metadata['metrics']['xgboost']['pr_auc']:.4f})"
    )
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Sentinel-Fin Fraud Scoring API",
    description=(
        "ML-driven fraud scoring with SHAP explanations. "
        "Part of the Sentinel-Fin capstone — see github.com/<your-handle>/sentinel-fin."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS so the local Streamlit dashboard can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health(scorer: FraudScorer = Depends(get_scorer)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=scorer.model is not None,
        model_version=scorer.metadata.get("model_version"),
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Meta"])
def model_info(scorer: FraudScorer = Depends(get_scorer)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=scorer.metadata.get("model_version", "unknown"),
        primary_model=scorer.metadata.get("primary_model", "xgboost"),
        trained_on_rows=scorer.metadata.get("trained_on_rows", 0),
        best_iteration=scorer.metadata.get("best_iteration", 0),
        metrics=scorer.metadata.get("metrics", {}),
        risk_tiers=scorer.risk_tiers,
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score(
    request: ScoreRequest,
    scorer: FraudScorer = Depends(get_scorer),
) -> ScoreResponse:
    try:
        result = scorer.score(request.features)
    except Exception as exc:
        log.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}")

    return ScoreResponse(
        transaction_id=request.transaction_id,
        fraud_probability=result["fraud_probability"],
        risk_tier=result["risk_tier"],
        top_shap_features=result["top_shap_features"],
        model_version=result["model_version"],
        scoring_latency_ms=result["scoring_latency_ms"],
        scored_at=datetime.now(timezone.utc),
    )