"""
Sentinel-Fin Correlation Engine.

Consumes auth events and transaction events in time-ordered streams.
For each transaction, queries the scoring API for ML fraud probability
and runs the rules in correlation_rules.py against the user's recent
auth history. When rules fire, emits a correlation_alert object that
combines ML risk + security context + MITRE mapping.

Stateful: maintains per-user history (recent auth events, known devices,
historical transaction amounts) in memory. Old data is pruned to keep
memory bounded.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import httpx

from cybersec.correlation_rules import (
    ALL_RULES, RuleMatch, UserContext,
)

log = logging.getLogger("correlation-engine")

# ---------- Config ----------

API_URL = "http://localhost:8000"
AUTH_HISTORY_MINUTES = 30          # how long to keep auth events per user
TRANSACTION_HISTORY = 100          # cap on stored amounts per user (rolling)
SCORE_TIMEOUT_SECONDS = 5.0


# ---------- Output schema ----------

@dataclass
class CorrelationAlert:
    """The final output of the engine for one flagged transaction."""
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


# ---------- The engine ----------

class CorrelationEngine:
    """
    Streaming correlation engine. Process auth and transaction events in
    chronological order; emit alerts when rules fire.
    """

    def __init__(self, api_url: str = API_URL) -> None:
        self.api_url = api_url
        # Per-user state — bounded by AUTH_HISTORY_MINUTES and TRANSACTION_HISTORY
        self._auth_history: dict[str, deque] = defaultdict(deque)
        # device fingerprint → first time we saw it (used to gate "known device" status)
        self._device_first_seen: dict[str, dict[str, datetime]] = defaultdict(dict)
        self._amount_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=TRANSACTION_HISTORY)
        )
        self._http = httpx.Client(timeout=SCORE_TIMEOUT_SECONDS)

    # ----- Ingest auth events (build context only, never emit alerts) -----

    def ingest_auth(self, event: dict[str, Any]) -> None:
        user_id = event["user_id"]
        self._auth_history[user_id].append(event)
        # Track devices observed on successful logins, with their first-seen timestamp
        if event["event_type"] == "login_success":
            dev = event["device_fingerprint"]
            if dev not in self._device_first_seen[user_id]:
                self._device_first_seen[user_id][dev] = datetime.fromisoformat(event["timestamp"])
        self._prune_old_auth(user_id, datetime.fromisoformat(event["timestamp"]))

    def _prune_old_auth(self, user_id: str, now: datetime) -> None:
        """Drop auth events older than AUTH_HISTORY_MINUTES."""
        cutoff = now - timedelta(minutes=AUTH_HISTORY_MINUTES)
        q = self._auth_history[user_id]
        while q and datetime.fromisoformat(q[0]["timestamp"]) < cutoff:
            q.popleft()

    # ----- Process a transaction: score + correlate -----

    def process_transaction(self, txn: dict[str, Any]) -> CorrelationAlert | None:
        user_id = txn["user_id"]
        txn_time = datetime.fromisoformat(txn["timestamp"])

        # Prune stale auth before evaluating
        self._prune_old_auth(user_id, txn_time)

        # 1. Get ML score from the API
        ml_score, ml_tier = self._score_via_api(txn)

        DEVICE_TENURE_MIN = 60  # a device counts as "known" only after 60 min of history

        known_devices = {
            dev for dev, first_seen in self._device_first_seen[user_id].items()
            if (txn_time - first_seen).total_seconds() / 60.0 >= DEVICE_TENURE_MIN
        }
        # 2. Build UserContext and run all rules
        context = UserContext(
            user_id=user_id,
            recent_auth_events=list(self._auth_history[user_id]),
            known_device_fingerprints=known_devices,
            historical_amounts=list(self._amount_history[user_id]),
        )
        matches: list[RuleMatch] = []
        for rule_fn in ALL_RULES:
            try:
                match = rule_fn(txn, context, txn_time)
                if match is not None:
                    matches.append(match)
            except Exception as exc:
                log.warning(f"Rule {rule_fn.__name__} errored on {txn['event_id']}: {exc}")

        # 3. Update transaction history AFTER scoring (so the new txn doesn't
        #    pollute its own baseline)
        self._amount_history[user_id].append(float(txn.get("amount_inr", 0)))

        # 4. Only emit an alert if EITHER the ML score is non-trivial OR any rule fired
        ml_meaningful = ml_score >= 0.05    # filter out clearly benign txns
        if not ml_meaningful and not matches:
            return None

        # 5. Combine ML score and rule boosts into a final risk score
        total_boost = sum(m.boost for m in matches)
        combined = min(1.0, ml_score + total_boost)
        final_tier = self._assign_combined_tier(combined, has_rules_fired=bool(matches))

        # 6. Build the alert
        mitre_set: set[str] = set()
        for m in matches:
            mitre_set.update(m.mitre_techniques)

        return CorrelationAlert(
            transaction_id=txn["event_id"],
            user_id=user_id,
            timestamp=txn["timestamp"],
            ml_fraud_probability=ml_score,
            ml_risk_tier=ml_tier,
            correlation_boost=total_boost,
            combined_risk_score=combined,
            final_risk_tier=final_tier,
            matched_rules=[asdict(m) for m in matches],
            mitre_techniques=sorted(mitre_set),
            transaction=txn,
            suggested_action=self._suggested_action(final_tier, matches),
        )

    # ----- Helpers -----

    def _score_via_api(self, txn: dict[str, Any]) -> tuple[float, str]:
        """Call the FastAPI /score endpoint. Returns (probability, tier)."""
        # The synthetic txn schema doesn't match the IEEE-CIS feature schema.
        # For the capstone demo, we map what we have onto a stub feature payload.
        # In production, an upstream feature store would do this.
        features = {
            "TransactionAmt": float(txn.get("amount_inr", 0)),
            "ProductCD": "C",  # default for the synthetic stream
            # Other features default to NaN/None and the model handles them
        }
        payload = {"transaction_id": txn["event_id"], "features": features}
        try:
            r = self._http.post(f"{self.api_url}/score", json=payload)
            r.raise_for_status()
            body = r.json()
            return float(body["fraud_probability"]), str(body["risk_tier"])
        except Exception as exc:
            log.warning(f"Scoring failed for {txn['event_id']}: {exc}")
            return 0.0, "LOW"

    @staticmethod
    def _assign_combined_tier(combined: float, has_rules_fired: bool) -> str:
        """
        Map combined score to a tier. Rules firing escalates the floor:
        if any rule fires, we never report LOW — at minimum the analyst sees MEDIUM.
        """
        if combined >= 0.55:
            return "CRITICAL"
        if combined >= 0.30:
            return "HIGH"
        if combined >= 0.10 or has_rules_fired:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _suggested_action(tier: str, matches: list[RuleMatch]) -> str:
        """Human-readable recommended action for the SOC analyst."""
        if tier == "CRITICAL":
            return "Immediate review. Escalate to Principal Officer for STR filing with FIU-IND (PMLA 7-day window). Do not freeze account unilaterally."
        if tier == "HIGH":
            if any("brute_force" in m.rule_name for m in matches):
                return "Tier-2 SOC review. Suspected account takeover via brute force. Initiate user contact for re-verification."
            return "Tier-2 SOC review. Flag for compliance hand-off if pattern persists."
        if tier == "MEDIUM":
            return "Tier-1 monitoring. Continue passive observation; no immediate user-facing action."
        return "No action required."


# ---------- Stream processing entry point ----------

def run_stream(
    auth_events_path: Path,
    txn_events_path: Path,
    alerts_out_path: Path | None = None,
    api_url: str = API_URL,
    max_events: int | None = None,
) -> list[CorrelationAlert]:
    """
    Process auth + transaction event streams chronologically.

    For the capstone, both streams are JSONL files. The engine merges them
    by timestamp so events are processed in the order they would have arrived
    in production.

    Returns the list of all emitted alerts. If alerts_out_path is given,
    also writes them to disk as JSONL.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load both streams and merge
    auth_events = [json.loads(line) for line in auth_events_path.read_text().splitlines() if line.strip()]
    txn_events = [json.loads(line) for line in txn_events_path.read_text().splitlines() if line.strip()]

    # Annotate kind and merge
    stream: list[tuple[str, dict[str, Any]]] = (
        [("auth", e) for e in auth_events] + [("txn", e) for e in txn_events]
    )
    stream.sort(key=lambda x: x[1]["timestamp"])
    if max_events:
        stream = stream[:max_events]

    log.info(f"Processing {len(stream):,} events ({len(auth_events):,} auth + {len(txn_events):,} txn)")

    engine = CorrelationEngine(api_url=api_url)
    alerts: list[CorrelationAlert] = []

    for i, (kind, event) in enumerate(stream):
        if kind == "auth":
            engine.ingest_auth(event)
        else:
            alert = engine.process_transaction(event)
            if alert is not None:
                alerts.append(alert)

        if (i + 1) % 1000 == 0:
            log.info(f"Processed {i+1:,} / {len(stream):,} events, {len(alerts):,} alerts so far")

    log.info(f"Done. Generated {len(alerts):,} alerts from {len(stream):,} events.")

    if alerts_out_path:
        alerts_out_path.parent.mkdir(parents=True, exist_ok=True)
        with alerts_out_path.open("w") as f:
            for a in alerts:
                f.write(json.dumps(asdict(a)) + "\n")
        log.info(f"Wrote alerts to {alerts_out_path}")

    return alerts


if __name__ == "__main__":
    DATA = Path("data/logs")
    OUT = Path("data/logs/correlation_alerts.jsonl")
    alerts = run_stream(
        auth_events_path=DATA / "auth_events.jsonl",
        txn_events_path=DATA / "transaction_events.jsonl",
        alerts_out_path=OUT,
    )

    # Quick summary
    from collections import Counter
    tier_counts = Counter(a.final_risk_tier for a in alerts)
    print("\n=== Alert summary ===")
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        print(f"  {tier:10s}: {tier_counts.get(tier, 0):,}")

    rules_fired = Counter(
        m["rule_name"]
        for a in alerts
        for m in a.matched_rules
    )
    print("\n=== Rules fired ===")
    for rule, count in rules_fired.most_common():
        print(f"  {rule:30s}: {count:,}")