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
    AlertListResponse,
    AlertListItem,
    AlertDetailResponse,
    StatsResponse,
)
from .scorer import FraudScorer,initialize_scorer, get_scorer
from .alert_store import AlertStore, initialize_alert_store, get_alert_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sentinel-fin")


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "ml" / "models"


ALERTS_PATH = Path(__file__).resolve().parents[2] / "data" / "logs" / "correlation_alerts.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Loading artifacts from {ARTIFACTS_DIR}")
    scorer = initialize_scorer(ARTIFACTS_DIR)
    log.info(
        f"Loaded model v{scorer.metadata.get('model_version')} "
        f"(trained on {scorer.metadata.get('trained_on_rows'):,} rows, "
        f"PR-AUC {scorer.metadata['metrics']['xgboost']['pr_auc']:.4f})"
    )
    store = initialize_alert_store(ALERTS_PATH)
    log.info(f"Loaded {len(store._alerts):,} alerts from {ALERTS_PATH.name}")
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

@app.get("/alerts", response_model=AlertListResponse, tags=["Alerts"])
def list_alerts(
    tier: str | None = None,
    rule: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    store: AlertStore = Depends(get_alert_store),
) -> AlertListResponse:
    """Paginated alert queue. Filter by tier, rule, or user_id."""
    page, total = store.list(tier=tier, rule=rule, user_id=user_id, limit=limit, offset=offset)
    return AlertListResponse(
        alerts=[
            AlertListItem(
                transaction_id=a["transaction_id"],
                user_id=a["user_id"],
                timestamp=a["timestamp"],
                final_risk_tier=a["final_risk_tier"],
                combined_risk_score=a["combined_risk_score"],
                ml_fraud_probability=a["ml_fraud_probability"],
                correlation_boost=a["correlation_boost"],
                mitre_techniques=a["mitre_techniques"],
                matched_rule_names=[m["rule_name"] for m in a["matched_rules"]],
                transaction_amount_inr=a.get("transaction", {}).get("amount_inr"),
            )
            for a in page
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/alerts/stats", response_model=StatsResponse, tags=["Alerts"])
def alert_stats(store: AlertStore = Depends(get_alert_store)) -> StatsResponse:
    """Aggregate stats across all alerts (for the dashboard's model-performance page)."""
    return StatsResponse(**store.stats())


@app.get("/alerts/{transaction_id}", response_model=AlertDetailResponse, tags=["Alerts"])
def get_alert(
    transaction_id: str,
    store: AlertStore = Depends(get_alert_store),
) -> AlertDetailResponse:
    """Full alert detail by transaction ID."""
    alert = store.get(transaction_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {transaction_id} not found")
    return AlertDetailResponse(**alert)