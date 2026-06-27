"""
Sentinel-Fin — Analyst Dashboard
Page 1: Alert Queue

A SOC analyst's working view. Lists alerts in newest-first order with filters
for tier, rule, and user. Clicking a row navigates to the detail page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import (
    fetch_alerts, fetch_stats, format_timestamp,
    tier_badge, TIER_EMOJI, check_api_health,
)


# ----- Page config -----
st.set_page_config(
    page_title="Sentinel-Fin — Alert Queue",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<h1 style='margin-bottom:0;'>🛡️ Sentinel-Fin</h1>"
    "<p style='color:#6b7280; margin-top:0;'>"
    "ML-driven fraud detection with security telemetry correlation"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()


# ----- Health check banner -----
if not check_api_health():
    st.error(
        "⚠️ Scoring API not reachable at http://localhost:8000. "
        "Start it with: `uvicorn integration.api.main:app --port 8000`"
    )
    st.stop()


# ----- Top stats strip -----
stats = fetch_stats()
total = stats.get("total", 0)
by_tier = stats.get("by_tier", {})

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total alerts", f"{total:,}")
col2.metric("🔴 Critical", by_tier.get("CRITICAL", 0))
col3.metric("🟠 High", by_tier.get("HIGH", 0))
col4.metric("🟡 Medium", by_tier.get("MEDIUM", 0))
col5.metric("🟢 Low", by_tier.get("LOW", 0))

st.divider()


# ----- Filters -----
st.subheader("Alert queue")

fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 2, 1])

with fcol1:
    tier_filter = st.selectbox(
        "Risk tier",
        options=["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        index=0,
    )

with fcol2:
    rule_filter = st.selectbox(
        "Rule fired",
        options=["All", "brute_force_precursor", "impossible_travel", "new_device_high_value"],
        index=0,
    )

with fcol3:
    user_filter = st.text_input("User ID (e.g., U_0170)", value="")

with fcol4:
    limit = st.number_input("Rows", min_value=10, max_value=200, value=50, step=10)


# ----- Fetch alerts based on filters -----
data = fetch_alerts(
    tier=tier_filter if tier_filter != "All" else None,
    rule=rule_filter if rule_filter != "All" else None,
    user_id=user_filter or None,
    limit=int(limit),
    offset=0,
)

alerts = data["alerts"]
total_filtered = data["total"]

st.caption(f"Showing {len(alerts)} of {total_filtered} matching alerts.")

if not alerts:
    st.info("No alerts match the current filters.")
    st.stop()


# ----- Render alert table -----
rows = []
for a in alerts:
    rows.append({
        "Tier": f"{TIER_EMOJI.get(a['final_risk_tier'], '')} {a['final_risk_tier']}",
        "Time": format_timestamp(a["timestamp"]),
        "User": a["user_id"],
        "Amount (₹)": f"{(a.get('transaction_amount_inr') or 0):,.2f}",
        "Risk": f"{a['combined_risk_score']:.2f}",
        "ML": f"{a['ml_fraud_probability']:.3f}",
        "Boost": f"+{a['correlation_boost']:.2f}",
        "Rules": ", ".join(a["matched_rule_names"]),
        "MITRE": ", ".join(a["mitre_techniques"]),
        "txn_id": a["transaction_id"],  # hidden for selection
    })

df = pd.DataFrame(rows)

# Render with selection enabled
event = st.dataframe(
    df.drop(columns=["txn_id"]),
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# When a row is selected, store the transaction_id and offer a button to view detail
if event.selection.rows: # type: ignore
    selected_idx = event.selection.rows[0] # type: ignore
    selected_txn_id = df.iloc[selected_idx]["txn_id"]
    st.session_state["selected_txn_id"] = selected_txn_id

    col_a, col_b = st.columns([3, 1])
    col_a.success(f"Selected alert: `{selected_txn_id}`")
    with col_b:
        if st.button("Open detail view →", type="primary", use_container_width=True):
            st.switch_page("pages/2_Alert_Detail.py")