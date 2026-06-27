"""
Sentinel-Fin Dashboard — Page 2: Alert Detail

Full evidence view for one alert. Designed for SOC analyst escalation:
shows the triggering transaction, fired rules with evidence, MITRE techniques,
auth event timeline, and a one-click evidence pack export.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (
    fetch_alert_detail, format_timestamp,
    tier_badge, TIER_EMOJI, MITRE_DESCRIPTIONS, check_api_health,
)


st.set_page_config(
    page_title="Sentinel-Fin — Alert Detail",
    page_icon="🛡️",
    layout="wide",
)


if not check_api_health():
    st.error("⚠️ Scoring API not reachable. Start it with uvicorn first.")
    st.stop()


# Get the selected transaction_id from session state
txn_id = st.session_state.get("selected_txn_id")
if not txn_id:
    st.warning("No alert selected. Return to the Alert Queue and pick one.")
    if st.button("← Back to queue"):
        st.switch_page("app.py")
    st.stop()


# ----- Fetch detail -----
try:
    alert = fetch_alert_detail(txn_id)
except Exception as e:
    st.error(f"Failed to load alert: {e}")
    st.stop()


# ----- Header -----
tier = alert["final_risk_tier"]
st.markdown(
    f"### {TIER_EMOJI.get(tier, '')} Alert Detail "
    f"<span style='font-size:14px; color:#6b7280;'>· `{txn_id}`</span>",
    unsafe_allow_html=True,
)
st.markdown(tier_badge(tier), unsafe_allow_html=True)

# ----- Top-level metrics -----
m1, m2, m3, m4 = st.columns(4)
m1.metric("Combined risk", f"{alert['combined_risk_score']:.3f}")
m2.metric("ML probability", f"{alert['ml_fraud_probability']:.4f}",
          help="Score from the XGBoost model (full feature set)")
m3.metric("Correlation boost", f"+{alert['correlation_boost']:.2f}",
          help="Added to ML score from fired rules")
m4.metric("Rules fired", len(alert["matched_rules"]))

st.divider()


# ----- Suggested action -----
st.markdown("##### Suggested action")
st.warning(alert["suggested_action"])


# ----- Two-column layout: transaction (L) + rules+MITRE (R) -----
col_l, col_r = st.columns([1, 1])

with col_l:
    st.markdown("##### 💳 Transaction")
    txn = alert["transaction"]
    tdf = pd.DataFrame([
        {"Field": "User", "Value": txn.get("user_id", "-")},
        {"Field": "Time", "Value": format_timestamp(txn.get("timestamp", ""))},
        {"Field": "Amount (₹)", "Value": f"{txn.get('amount_inr', 0):,.2f}"},
        {"Field": "Channel", "Value": txn.get("channel", "-").upper()},
        {"Field": "Category", "Value": txn.get("merchant_category", "-")},
        {"Field": "Payee", "Value": txn.get("payee_id", "-")},
        {"Field": "Payee is new?", "Value": "✓ YES" if txn.get("payee_is_new") else "no"},
        {"Field": "Device fingerprint", "Value": txn.get("device_fingerprint", "-")[:20]},
    ])
    st.dataframe(tdf, hide_index=True, use_container_width=True)

with col_r:
    st.markdown("##### 🎯 MITRE ATT&CK techniques")
    for tech in alert["mitre_techniques"]:
        desc = MITRE_DESCRIPTIONS.get(tech, "(no local description)")
        st.markdown(
            f"**[`{tech}`](https://attack.mitre.org/techniques/{tech}/)** — {desc}"
        )

    st.markdown("##### ⚙️ Rules fired")
    for rule in alert["matched_rules"]:
        with st.expander(
            f"**{rule['rule_name']}**  "
            f"(confidence={rule['confidence']:.2f}, boost=+{rule['boost']:.2f})"
        ):
            st.write(rule["description"])
            st.caption(f"MITRE: {', '.join(rule['mitre_techniques'])}")
            st.caption(f"Evidence events: {len(rule['evidence'])}")


st.divider()


# ----- Auth event timeline -----
st.markdown("##### 📅 Auth event timeline (preceding the transaction)")

# Aggregate all evidence events from all rules
all_events = []
for rule in alert["matched_rules"]:
    for ev in rule["evidence"]:
        if "event_type" in ev:  # an auth event
            all_events.append(ev)
        elif "amount_inr" in ev:  # the transaction itself — skip
            continue

# Deduplicate by event_id, sort by timestamp
seen = set()
deduped = []
for ev in all_events:
    if ev.get("event_id") in seen:
        continue
    seen.add(ev.get("event_id"))
    deduped.append(ev)
deduped.sort(key=lambda e: e.get("timestamp", ""))

if deduped:
    timeline_rows = []
    for ev in deduped:
        ev_type = ev.get("event_type", "")
        icon = "✅" if ev_type == "login_success" else "❌"
        timeline_rows.append({
            "": icon,
            "Time": format_timestamp(ev.get("timestamp", "")),
            "Event": ev_type,
            "IP": ev.get("ip_address", "-"),
            "City": f"{ev.get('city', '-')} ({ev.get('country', '-')})",
            "Device": ev.get("device_fingerprint", "")[:18],
        })
    st.dataframe(
        pd.DataFrame(timeline_rows),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.caption("No auth events in evidence.")


st.divider()


# ----- Evidence pack export -----
st.markdown("##### 📦 Evidence pack")
st.caption(
    "Download a JSON file containing this complete alert (transaction, "
    "fired rules with evidence, MITRE mapping, suggested action) for "
    "compliance hand-off to the Principal Officer."
)
evidence_pack = json.dumps(alert, indent=2, default=str)
st.download_button(
    label="⬇ Download evidence_pack.json",
    data=evidence_pack,
    file_name=f"evidence_{txn_id[:8]}.json",
    mime="application/json",
    type="primary",
)


# ----- Back button -----
st.divider()
if st.button("← Back to Alert Queue"):
    st.switch_page("app.py")