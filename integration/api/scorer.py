"""Fraud scoring core — loads model + explainer, scores single transactions."""
from __future__ import annotations

import json
from pyexpat import features
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


class FraudScorer:
    """
    Encapsulates the loaded model, SHAP explainer, and category maps.
    Instantiated once at app startup and held in memory.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.model: xgb.XGBClassifier | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.category_maps: dict[str, dict[str, int]] = {}
        self.metadata: dict[str, Any] = {}
        self.feature_columns: list[str] = []
        self.risk_tiers: dict[str, float] = {}

    def load(self) -> None:
        """Load all artifacts from disk. Called once at app startup."""
        self.model = joblib.load(self.artifacts_dir / "xgboost_model.joblib")
        self.explainer = joblib.load(self.artifacts_dir / "shap_explainer.joblib")
        self.category_maps = joblib.load(self.artifacts_dir / "category_maps.joblib")
        with open(self.artifacts_dir / "model_metadata.json") as f:
            self.metadata = json.load(f)
        self.feature_columns = self.metadata["feature_columns"]
        self.risk_tiers = self.metadata["risk_tiers"]

    def _encode_features(self, features: dict[str, Any]) -> pd.DataFrame:
        """
        Convert a single-transaction feature dict into a model-ready DataFrame.

        Builds output column-by-column with explicit dtypes to avoid pandas
        inferring PyArrow string dtype from incoming JSON values, which would
        then reject integer assignments during categorical encoding.
        """
        import math

        categorical_cols = set(self.category_maps.keys())
        encoded_values: dict[str, Any] = {}

        for col in self.feature_columns:
            val = features.get(col, None)

            # Detect missing uniformly
            is_missing = (
                val is None
                or (isinstance(val, float) and math.isnan(val))
                or (isinstance(val, str) and val.lower() in ("nan", "none", ""))
            )

            if col in categorical_cols:
                mapping = self.category_maps[col]
                if is_missing:
                    encoded = mapping.get("__missing__", len(mapping))
                else:
                    encoded = mapping.get(val) # type: ignore
                    if encoded is None:
                        encoded = mapping.get(str(val))
                    if encoded is None:
                        try:
                            encoded = mapping.get(int(float(val)))  #type: ignore
                        except (ValueError, TypeError):
                            encoded = None
                    if encoded is None:
                        encoded = len(mapping)
                encoded_values[col] = np.int32(encoded)
            else:
                if is_missing:
                    encoded_values[col] = np.float64(np.nan)
                else:
                    try:
                        encoded_values[col] = np.float64(val)
                    except (ValueError, TypeError):
                        encoded_values[col] = np.float64(np.nan)

        # Build DataFrame in one shot with explicit dtypes
        df = pd.DataFrame(
            {col: [encoded_values[col]] for col in self.feature_columns}
        )

        # Ensure dtypes are clean (int32 for categoricals, float64 for everything else)
        for col in self.feature_columns:
            if col in categorical_cols:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float64)

        return df   

    def _assign_tier(self, probability: float) -> str:
        """Map fraud_probability to LOW/MEDIUM/HIGH/CRITICAL using saved thresholds."""
        # Tiers in the metadata file are: LOW=0, MEDIUM=x, HIGH=y, CRITICAL=z
        if probability >= self.risk_tiers["CRITICAL"]:
            return "CRITICAL"
        if probability >= self.risk_tiers["HIGH"]:
            return "HIGH"
        if probability >= self.risk_tiers["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Score a single transaction. Returns the full response payload
        (minus transaction_id and timestamps, which the API layer adds).
        """
        if self.model is None or self.explainer is None:
            raise RuntimeError("Scorer not loaded — call .load() first.")

        start = time.perf_counter()
        X = self._encode_features(features)

        # Predict
        proba = float(self.model.predict_proba(X)[0, 1])
        tier = self._assign_tier(proba)

        # SHAP — get the top 5 contributing features by |contribution|
        shap_values = self.explainer.shap_values(X)[0]  # 1D array, one value per feature
        contributions = sorted(
            zip(self.feature_columns, X.iloc[0].tolist(), shap_values.tolist()),
            key=lambda t: abs(t[2]),
            reverse=True,
        )[:5]
        top_features = [
            {"feature": f, "value": v, "contribution": float(c)}
            for f, v, c in contributions
        ]

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "fraud_probability": proba,
            "risk_tier": tier,
            "top_shap_features": top_features,
            "model_version": self.metadata.get("model_version", "unknown"),
            "scoring_latency_ms": round(latency_ms, 2),
        }


# Global instance — populated at app startup by main.py's lifespan
_scorer: FraudScorer | None = None


def get_scorer() -> FraudScorer:
    """Dependency injection for FastAPI endpoints."""
    if _scorer is None:
        raise RuntimeError("Scorer not initialized.")
    return _scorer


def initialize_scorer(artifacts_dir: Path) -> FraudScorer:
    global _scorer
    _scorer = FraudScorer(artifacts_dir)
    _scorer.load()
    return _scorer