"""
Detection rules for the Sentinel-Fin correlation engine.

Each rule encodes one fraud-indicative pattern that combines security telemetry
(auth events) with transaction context. Rules return a RuleMatch if they fire,
None otherwise. Each match carries:
  - Matched rule name
  - MITRE ATT&CK technique(s) the pattern maps to
  - Evidence (the specific auth events that triggered)
  - Confidence (0-1): how strong the pattern signal is
  - Boost (0-1): how much to add to the ML fraud probability

Rules are deliberately conservative — false positives waste analyst time;
false negatives miss real fraud. The thresholds here (5 failed logins,
500km, 30 minutes) come from real ATO attack analysis and would be tuned
in production against ground-truth fraud labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


# ---------- Data classes ----------

@dataclass
class UserContext:
    """
    Snapshot of what the engine knows about one user at the moment of scoring
    a transaction. Assembled by the engine; rules query it but don't mutate.
    """
    user_id: str
    recent_auth_events: list[dict[str, Any]] = field(default_factory=list)
    known_device_fingerprints: set[str] = field(default_factory=set)
    historical_amounts: list[float] = field(default_factory=list)

    def auth_events_within(self, ref_time: datetime, minutes: int) -> list[dict[str, Any]]:
        """Return auth events within `minutes` before ref_time."""
        cutoff = ref_time - timedelta(minutes=minutes)
        return [
            e for e in self.recent_auth_events
            if datetime.fromisoformat(e["timestamp"]) >= cutoff
        ]


@dataclass
class RuleMatch:
    """Output of a fired rule. Multiple matches can occur per transaction."""
    rule_name: str
    mitre_techniques: list[str]
    evidence: list[dict[str, Any]]
    confidence: float
    boost: float
    description: str


# ---------- Helpers ----------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two lat/lon points."""
    R = 6371.0  # Earth's mean radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------- Rules ----------

def rule_impossible_travel(
    transaction: dict[str, Any],
    user: UserContext,
    now: datetime,
) -> RuleMatch | None:
    """
    Rule 1: Two successful logins from points >500km apart within 10 minutes.

    Real users can't be in Bengaluru at 13:00 and Lagos at 13:05.
    This is a strong signal that one of the two logins used stolen credentials.

    Maps to MITRE T1078 (Valid Accounts) — the attacker has working credentials
    and is logging in legitimately, but from an impossible location.
    """
    recent = user.auth_events_within(now, minutes=10)
    successes = [e for e in recent if e["event_type"] == "login_success"]

    if len(successes) < 2:
        return None

    # Check each pair of successful logins for impossible travel
    successes.sort(key=lambda e: e["timestamp"])
    for i in range(len(successes) - 1):
        a, b = successes[i], successes[i + 1]
        distance = haversine_km(
            a["latitude"], a["longitude"],
            b["latitude"], b["longitude"],
        )
        if distance < 500:
            continue

        t_a = datetime.fromisoformat(a["timestamp"])
        t_b = datetime.fromisoformat(b["timestamp"])
        minutes = (t_b - t_a).total_seconds() / 60.0
        # Even a commercial flight averages ~800 km/h.
        # 500km in <10 min implies ~3000 km/h — physically impossible for a human.

        return RuleMatch(
            rule_name="impossible_travel",
            mitre_techniques=["T1078"],
            evidence=[a, b],
            confidence=min(1.0, distance / 5000),  # higher distance = higher confidence
            boost=0.35,
            description=(
                f"Successful logins from {a['city']} ({a['country']}) and "
                f"{b['city']} ({b['country']}) — {distance:.0f}km apart, "
                f"{minutes:.1f} minutes between events."
            ),
        )
    return None


def rule_brute_force_precursor(
    transaction: dict[str, Any],
    user: UserContext,
    now: datetime,
) -> RuleMatch | None:
    """
    Rule 2: ≥5 failed logins followed by a successful login within 30 minutes.

    This is the credential-stuffing / brute-force signature: the attacker tries
    multiple passwords, eventually gets one right, then transacts.

    Maps to MITRE T1110 (Brute Force) as the precursor technique, plus
    T1078 (Valid Accounts) for the resulting authenticated session.
    """
    recent = user.auth_events_within(now, minutes=30)
    if not recent:
        return None

    # Find the most recent success
    successes = [e for e in recent if e["event_type"] == "login_success"]
    if not successes:
        return None

    successes.sort(key=lambda e: e["timestamp"])
    latest_success = successes[-1]
    t_success = datetime.fromisoformat(latest_success["timestamp"])

    # Count failures in the 30 min PRECEDING the success
    failures_before = [
        e for e in recent
        if e["event_type"] == "login_failure"
        and datetime.fromisoformat(e["timestamp"]) < t_success
        and datetime.fromisoformat(e["timestamp"]) >= t_success - timedelta(minutes=30)
    ]

    if len(failures_before) < 5:
        return None

    return RuleMatch(
        rule_name="brute_force_precursor",
        mitre_techniques=["T1110", "T1078"],
        evidence=failures_before + [latest_success],
        confidence=min(1.0, len(failures_before) / 10),
        boost=0.40,
        description=(
            f"{len(failures_before)} failed login attempts preceded a successful "
            f"login at {latest_success['timestamp']} (within 30 min window). "
            f"Failures originated from IP {failures_before[0]['ip_address']}."
        ),
    )


def rule_new_device_high_value(
    transaction: dict[str, Any],
    user: UserContext,
    now: datetime,
) -> RuleMatch | None:
    """
    Rule 3: Login from never-seen device fingerprint, followed within 5 minutes
    by a transaction above the user's 95th percentile historical amount.

    The combination is what matters. A new device alone is normal (people get
    new phones). A large transaction alone is normal (people make big purchases).
    A new device followed immediately by a large transfer is the cash-out signature
    of account takeover.

    Maps to MITRE T1078 (Valid Accounts) — same technique as rule 1, but a
    different observable pattern.
    """
    # Need historical baseline; cold-start users get a pass
    if len(user.historical_amounts) < 5:
        return None

    p95 = sorted(user.historical_amounts)[int(0.95 * len(user.historical_amounts))]
    txn_amount = float(transaction.get("amount_inr", 0))
    if txn_amount <= p95:
        return None

    # Look for a login from an unknown device in the last 5 minutes
    recent = user.auth_events_within(now, minutes=5)
    successes = [e for e in recent if e["event_type"] == "login_success"]
    new_device_logins = [
        e for e in successes
        if e["device_fingerprint"] not in user.known_device_fingerprints
    ]

    if not new_device_logins:
        return None

    latest = new_device_logins[-1]
    return RuleMatch(
        rule_name="new_device_high_value",
        mitre_techniques=["T1078"],
        evidence=[latest, transaction],
        confidence=min(1.0, txn_amount / (p95 * 3)),  # higher amount = higher confidence
        boost=0.30,
        description=(
            f"Transaction of {txn_amount:,.2f} INR (>{p95:,.2f} = user's p95) "
            f"within 5 min of login from new device "
            f"{latest['device_fingerprint']} in {latest['city']}."
        ),
    )


# ---------- Rule registry ----------

ALL_RULES = [
    rule_impossible_travel,
    rule_brute_force_precursor,
    rule_new_device_high_value,
]