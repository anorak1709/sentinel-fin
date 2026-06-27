"""Alert storage and retrieval — file-backed for capstone, swappable for SQLite later."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


class AlertStore:
    """
    Reads correlation_alerts.jsonl into memory at startup; serves filtered queries.
    File is reloaded on each /alerts call via reload() if the file's modification
    time changed — keeps the demo fresh when the correlation engine re-runs.
    """

    def __init__(self, alerts_path: Path) -> None:
        self.alerts_path = alerts_path
        self._alerts: list[dict[str, Any]] = []
        self._loaded_mtime: float = 0.0
        self.reload()

    def reload(self) -> None:
        if not self.alerts_path.exists():
            self._alerts = []
            self._loaded_mtime = 0.0
            return
        mtime = self.alerts_path.stat().st_mtime
        if mtime == self._loaded_mtime and self._alerts:
            return
        with self.alerts_path.open() as f:
            self._alerts = [json.loads(line) for line in f if line.strip()]
        # newest first
        self._alerts.sort(key=lambda a: a["timestamp"], reverse=True)
        self._loaded_mtime = mtime

    def list(
        self,
        tier: str | None = None,
        rule: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return filtered alerts. Returns (page, total_count)."""
        self.reload()
        out = self._alerts
        if tier:
            out = [a for a in out if a["final_risk_tier"] == tier]
        if rule:
            out = [a for a in out if any(m["rule_name"] == rule for m in a["matched_rules"])]
        if user_id:
            out = [a for a in out if a["user_id"] == user_id]
        total = len(out)
        return out[offset : offset + limit], total

    def get(self, transaction_id: str) -> dict[str, Any] | None:
        self.reload()
        for a in self._alerts:
            if a["transaction_id"] == transaction_id:
                return a
        return None

    def stats(self) -> dict[str, Any]:
        self.reload()
        if not self._alerts:
            return {"total": 0}

        tier_counts = Counter(a["final_risk_tier"] for a in self._alerts)
        rule_counts = Counter(
            m["rule_name"]
            for a in self._alerts
            for m in a["matched_rules"]
        )
        mitre_counts = Counter(
            t
            for a in self._alerts
            for t in a["mitre_techniques"]
        )
        rules_per_alert = Counter(len(a["matched_rules"]) for a in self._alerts)

        # Combined score distribution
        scores = [a["combined_risk_score"] for a in self._alerts]
        score_buckets = Counter()
        for s in scores:
            bucket = int(s * 10) / 10  # 0.0, 0.1, 0.2, ..., 1.0
            score_buckets[bucket] += 1

        # Alerts by hour
        hour_counts = Counter()
        for a in self._alerts:
            dt = datetime.fromisoformat(a["timestamp"])
            hour_counts[dt.hour] += 1

        return {
            "total": len(self._alerts),
            "by_tier": dict(tier_counts),
            "by_rule": dict(rule_counts),
            "by_mitre": dict(mitre_counts),
            "rules_per_alert": {str(k): v for k, v in sorted(rules_per_alert.items())},
            "score_distribution": {f"{k:.1f}": v for k, v in sorted(score_buckets.items())},
            "by_hour": {str(h): hour_counts.get(h, 0) for h in range(24)},
        }


_store: AlertStore | None = None


def get_alert_store() -> AlertStore:
    if _store is None:
        raise RuntimeError("AlertStore not initialized.")
    return _store


def initialize_alert_store(alerts_path: Path) -> AlertStore:
    global _store
    _store = AlertStore(alerts_path)
    return _store