"""
Sentinel-Fin Dashboard — Page 3: Model Performance & Risk

Shows aggregate alert stats, model metrics, and the model-risk disclaimer
required under RBI model risk management guidance.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import fetch_stats, fetch_model_info, check_api_health


st.set_page_config(
    page_title="Sentinel-Fin — Model Performance",
    page_icon="📊",
    layout="wide",
)


if not check_api_health():
    st.error("⚠️ Scoring API not reachable.")
    st.stop()


st.title("📊 Model Performance & Risk")
st.caption("For compliance and risk review. Refreshed every 30 seconds.")
st.divider()


# ----- Fetch -----
stats = fetch_stats()
model = fetch_model_info()


# ----- Top metrics -----
m1, m2, m3, m4 = st.columns(4)
metrics = model.get("metrics", {})
xgb_pr = metrics.get("xgboost", {}).get("pr_auc", 0)
xgb_roc = metrics.get("xgboost", {}).get("roc_auc", 0)

m1.metric("PR-AUC (holdout)", f"{xgb_pr:.4f}", help="Precision-Recall AUC on temporal holdout")
m2.metric("ROC-AUC (holdout)", f"{xgb_roc:.4f}")
m3.metric("Trained on rows", f"{model.get('trained_on_rows', 0):,}")
m4.metric("Best iteration", f"{model.get('best_iteration', 0):,}")

st.divider()


# ----- Tier and rule breakdowns -----
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Alerts by tier")
    tier_df = pd.DataFrame(
        [{"Tier": k, "Count": v} for k, v in stats["by_tier"].items()]
    )
    tier_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    tier_df["sort"] = tier_df["Tier"].map(tier_order).fillna(99)
    tier_df = tier_df.sort_values("sort").drop(columns="sort")
    st.bar_chart(tier_df.set_index("Tier"))

with col_r:
    st.subheader("Rules fired (counts)")
    rule_df = pd.DataFrame(
        [{"Rule": k, "Count": v} for k, v in stats["by_rule"].items()]
    ).sort_values("Count", ascending=False)
    st.bar_chart(rule_df.set_index("Rule"))


col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("MITRE techniques")
    mitre_df = pd.DataFrame(
        [{"Technique": k, "Count": v} for k, v in stats["by_mitre"].items()]
    ).sort_values("Count", ascending=False)
    st.dataframe(mitre_df, hide_index=True, use_container_width=True)

with col_r2:
    st.subheader("Rules per alert")
    rpa = stats.get("rules_per_alert", {})
    rpa_df = pd.DataFrame(
        [{"# Rules fired": k, "# Alerts": v} for k, v in rpa.items()]
    )
    st.dataframe(rpa_df, hide_index=True, use_container_width=True)


st.divider()


# ----- Alerts by hour -----
st.subheader("Alert volume by hour of day")
hour_data = stats.get("by_hour", {})
hour_df = pd.DataFrame([
    {"Hour": int(h), "Alerts": v}
    for h, v in hour_data.items()
]).sort_values("Hour")
st.bar_chart(hour_df.set_index("Hour"))


st.divider()


# ----- Model risk disclaimer (the regulatory artifact) -----
st.subheader("⚠️ Model Risk Disclaimer")
st.markdown(
    """
**Per RBI Master Direction on Information Technology Governance (2023) and broader model
risk management principles, the following limitations of the Sentinel-Fin scoring model
must be considered before any automated decisioning:**

- **Synthetic training data.** The model was trained on the publicly available
  IEEE-CIS Fraud Detection dataset (e-commerce chargebacks, ~3.5% fraud rate).
  Production deployment would require retraining on the institution's own transaction
  data with appropriate class balance and feature engineering.

- **Achieved PR-AUC: 0.56.** The architecture-doc target was 0.65; the realized metric
  is below target. Operationally usable but below state-of-the-art on this dataset.
  Higher PR-AUC achievable with: full target encoding feature work (1-2 weeks),
  hyperparameter sweeps, and ensemble methods beyond XGBoost.

- **No drift monitoring.** This capstone does not include a model drift monitor.
  Production deployment must include automated drift detection (e.g., Evidently AI,
  custom KS-tests on score distributions) with retraining triggers.

- **Limited adversarial testing.** The model has not been subjected to adversarial
  red-teaming. Production deployment should include adversarial robustness evaluation.

- **Explainability.** SHAP TreeExplainer is integrated and provides per-prediction
  feature attribution. Compliance teams can review alert-level explanations via the
  evidence pack export. SHAP values are computed at scoring time, not cached.

- **Suggested actions are advisory, not mandatory.** The dashboard recommends
  escalation to the Principal Officer for review. **No transaction is auto-blocked**
  and **no account is auto-frozen** — those actions require human review and, for
  freezing, regulatory/judicial authorization per PMLA.
    """
)


# ----- Negative finding: Isolation Forest rejection -----
st.subheader("Negative findings preserved")
iso = metrics.get("isolation_forest_rejected", {})
if iso:
    st.info(
        f"**Isolation Forest tested and rejected.** Standalone PR-AUC: {iso.get('pr_auc', 0):.4f}, "
        f"ROC-AUC: {iso.get('roc_auc', 0):.4f}. "
        f"{iso.get('note', '')} "
        f"Documented in `ml/notebooks/03_modeling.ipynb`."
    )