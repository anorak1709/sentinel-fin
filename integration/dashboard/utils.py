"""Shared helpers for the Sentinel-Fin dashboard."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
import streamlit as st


API_URL = "http://localhost:8000"
TIMEOUT = 10.0


TIER_COLORS = {
    "CRITICAL": "#dc2626",   # red-600
    "HIGH":     "#ea580c",   # orange-600
    "MEDIUM":   "#ca8a04",   # yellow-600
    "LOW":      "#16a34a",   # green-600
}

TIER_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

MITRE_DESCRIPTIONS = {
    "T1078":   "Valid Accounts — adversary uses legitimate credentials",
    "T1110":   "Brute Force — adversary attempts to access accounts via password guessing",
    "T1566":   "Phishing — adversary sends deceptive messages to steal credentials",
    "T1539":   "Steal Web Session Cookie",
    "T1056":   "Input Capture — keyloggers stealing banking credentials",
}


@st.cache_data(ttl=10)
def fetch_alerts(
    tier: str | None = None,
    rule: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Fetch a paginated alert list from the API."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if tier:
        params["tier"] = tier
    if rule:
        params["rule"] = rule
    if user_id:
        params["user_id"] = user_id
    r = httpx.get(f"{API_URL}/alerts", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def fetch_alert_detail(transaction_id: str) -> dict[str, Any]:
    r = httpx.get(f"{API_URL}/alerts/{transaction_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30)
def fetch_stats() -> dict[str, Any]:
    r = httpx.get(f"{API_URL}/alerts/stats", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def fetch_model_info() -> dict[str, Any]:
    r = httpx.get(f"{API_URL}/model_info", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def format_timestamp(ts: str) -> str:
    """Human-readable timestamp."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts


def tier_badge(tier: str) -> str:
    """Markdown-rendered colored tier badge."""
    color = TIER_COLORS.get(tier, "#6b7280")
    return f"<span style='background-color: {color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px;'>{tier}</span>"


def check_api_health() -> bool:
    """Returns True if the API is reachable and the model is loaded."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False