# Sentinel-Fin

**TL;DR:** End-to-end fraud detection system combining ML scoring (XGBoost, PR-AUC 0.56) with security event correlation (100% recall on injected ATO attack chains, 89% precision), MITRE ATT&CK-tagged alerts, and SHAP explainability — surfaced through a SOC analyst dashboard.

| Metric                                | Value                          |
| ------------------------------------- | ------------------------------ |
| Fraud model PR-AUC (temporal holdout) | **0.56**                       |
| Fraud model ROC-AUC                   | **0.89**                       |
| Logistic regression baseline PR-AUC   | 0.20 (2.8× lift)               |
| Correlation engine recall             | **100%** (25 / 25 attacks)     |
| Correlation engine precision          | **89%** (3 FPs / 28 alerts)    |
| Multi-rule alerts (2+ rules fired)    | **26 / 28** (93%)              |
| Scoring API p95 latency               | **< 50 ms**                    |
| SHAP attribution latency (p95)        | **< 16 ms**                    |

**Built by:** [Kartik Khera](https://linkedin.com/in/kartikkhera) ([@anorak1709](https://github.com/anorak1709))
**Stack:** Python · XGBoost · SHAP · FastAPI · Streamlit · pandas

---

## What this is

Sentinel-Fin is a fraud detection system that combines two signals most bank security teams treat separately:

1. **A transaction's ML-predicted fraud probability** (XGBoost trained on IEEE-CIS Fraud Detection)
2. **Security telemetry around that transaction** (authentication events, device fingerprints, geo patterns)

When both signals correlate — say, a high-risk transaction occurring minutes after an authentication chain that looks like account takeover — the system emits an alert tagged with the relevant MITRE ATT&CK technique(s), with SHAP-based feature explanations and a recommended action calibrated to PMLA reporting requirements.

The architecture mirrors how a Tier-1 SOC analyst at a mid-size Indian bank or NBFC would triage fraud alerts before escalating to the Principal Officer for STR filing with FIU-IND.

---

## Why this matters

Most student fraud projects stop at the model. Most production fraud systems don't — they sit downstream of a SOC, ingest auth/device telemetry, correlate across signals, and produce evidence packages that can be defended under regulatory review.

Sentinel-Fin demonstrates the full pipeline:

- **Feature engineering with leakage discipline.** Bayesian-smoothed target encoding, train-only aggregations, temporal holdout splits.
- **An ML model with honest metrics.** PR-AUC 0.56 (below the architecture-doc target of 0.65) is reported as-is, with the gap explained. The Isolation Forest was tested and rejected — also reported as-is.
- **A correlation engine that detects attack *chains*, not isolated events.** Three rules covering ATO patterns (impossible travel, brute-force precursor, new-device + high-value transaction) with MITRE T1078/T1110 mapping.
- **An analyst dashboard with regulatory framing.** Alerts surface evidence, MITRE links, suggested actions — and never recommend unilateral account freezing, which under PMLA requires a regulatory or judicial order.

---

## Architecture

```
┌─────────────────────┐       ┌──────────────────────────┐
│  Stream simulator   │       │  FastAPI scoring service │
│  (replays synthetic │──txn──>  /score                   │
│   auth + txn logs   │       │  /alerts                  │
│   chronologically)  │       │  /alerts/{id}             │
└─────────┬───────────┘       │  /alerts/stats            │
          │                   │  /model_info  /health     │
          ├──auth─────────────>                          │
          │                   │  ┌────────────────────┐  │
          │                   │  │ XGBoost  +  SHAP   │  │
          │                   │  │  TreeExplainer     │  │
          │                   │  └────────────────────┘  │
          │                   └───────────┬──────────────┘
          │                               │
          ▼                               ▼
┌──────────────────────────────────────────────────────┐
│         Correlation Engine (Python)                  │
│   • per-user auth history (30 min rolling)           │
│   • device fingerprint tenure tracking               │
│   • 3 rules → MITRE T1078, T1110                     │
│   • combined risk score = ML score + rule boosts     │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  Streamlit Dashboard     │
            │  • Alert queue           │
            │  • Per-alert evidence    │
            │  • Model performance     │
            └──────────────────────────┘
```

Full architecture decisions, scope-outs, and rejected alternatives are documented in [`docs/architecture.md`](docs/architecture.md).

---

## Repository layout

```
sentinel-fin/
├── cybersec/
│   ├── correlation_engine.py       # streaming engine
│   ├── correlation_rules.py        # rule definitions (3 rules, MITRE-mapped)
│   ├── mitre_mapping.md
│   └── log_generator/
│       └── auth_log_synth.py       # synthetic auth + txn streams w/ injected ATO chains
├── ml/
│   ├── notebooks/
│   │   ├── 01_fraud_eda.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_modeling.ipynb
│   │   └── 04_shap_interpretability.ipynb
│   └── models/                     # xgboost, shap explainer, category maps, metadata
├── integration/
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   ├── models.py               # Pydantic request/response models
│   │   ├── scorer.py               # XGBoost + SHAP scoring core
│   │   └── alert_store.py          # alert storage and retrieval
│   └── dashboard/
│       ├── app.py                  # alert queue (page 1)
│       ├── pages/
│       │   ├── 2_Alert_Detail.py
│       │   └── 3_Model_Performance.py
│       └── utils.py
├── data/                           # gitignored — see "Setup" below
├── docs/
│   ├── architecture.md             # system design + scope decisions
│   ├── eda_findings.md             # data understanding (documented EDA)
│   └── domain_notes.md             # fraud + AML + MITRE synthesis
└── README.md
```

---

## Notable engineering decisions

### Caught and fixed: target-encoding data leak

The first iteration target-encoded `card1_fraud_rate` (the per-card historical fraud rate) without smoothing. The SHAP analysis flagged this immediately:

> `card1_fraud_rate` dominated importance at mean |SHAP| = 2.54 — **~7× the next feature**.

Rare cards in training had `card1_fraud_rate = 1.0` (every observation happened to be fraud), making the feature an oblique label-leak. Bayesian smoothing with a prior of 100 observations at the global rate (3.51%) fixed this: cards with sparse history get anchored to the global rate while well-observed cards retain meaningful signal. Post-fix, PR-AUC moved from 0.552 (leaky) to 0.560 (clean), ROC-AUC improved 0.874 → 0.895, and SHAP importance redistributed across V-features and counter columns.

The pre-fix notebook is preserved in git history. The post-fix smoothing parameter (`SMOOTHING_PRIOR = 100`) is documented in `ml/notebooks/02_feature_engineering.ipynb`.

### Tested and rejected: Isolation Forest ensemble

The architecture doc originally specified an XGBoost + Isolation Forest ensemble. Empirical testing:

| Model                 | PR-AUC     | ROC-AUC    |
| --------------------- | ---------- | ---------- |
| Logistic Regression   | 0.1952     | 0.8494     |
| XGBoost               | **0.5600** | 0.8949     |
| Isolation Forest      | 0.0953     | 0.7519     |
| Ensemble (0.75/0.25)  | 0.5517     | **0.8987** |

Score correlation between XGBoost and IsoForest was only **0.15** — the models capture orthogonal signal. But the orthogonal information did not translate to higher precision at operational thresholds. The ensemble reduced PR-AUC below XGBoost alone, so the IsoForest was removed from the production scoring path. The training code remains in `ml/notebooks/03_modeling.ipynb` for reproducibility and the model metadata file preserves the negative finding.

### Designed for production swap-in

- **Scoring service** is a thin FastAPI wrapper around a single `FraudScorer` class. The Pydantic models in `integration/api/models.py` define the production-facing contract. The Streamlit dashboard talks to it via HTTP — a React frontend would plug in with no API changes.
- **Correlation engine** is stateful but its rule definitions are pure functions in `correlation_rules.py`. Adding a fourth rule is one function and one entry in `ALL_RULES`. No engine modification needed.
- **Streaming layer is deliberately simulated, not Kafka.** A `stream_simulator.py` replays JSONL files at a configurable rate. Production deployment would swap this for a Kafka consumer with no business-logic changes. Documented as Week 9 extension.

---

## Setup

### Prerequisites

- Python 3.12
- ~3 GB free RAM (the merged IEEE-CIS frame is ~2 GB)
- macOS / Linux (tested on Apple Silicon M5)

### Install

```bash
git clone https://github.com/anorak1709/sentinel-fin.git
cd sentinel-fin

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install (uv recommended; pip works fine)
uv pip install -r requirements.txt
```

### Get the IEEE-CIS dataset

Download the four files from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) — requires accepting the competition rules:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

Place them in `data/raw/`.

### Reproduce the model

Run the notebooks in order from `ml/notebooks/`:

1. `01_fraud_eda.ipynb` — produces `data/processed/merged_clean.parquet`
2. `02_feature_engineering.ipynb` — produces `data/processed/features_ready.parquet`
3. `03_modeling.ipynb` — trains XGBoost, saves to `ml/models/`
4. `04_shap_interpretability.ipynb` — fits explainer, saves to `ml/models/`

Total wall time on M5: ~25 minutes (notebook 03 dominates).

### Run the system

You'll need three terminals.

**Terminal 1 — generate synthetic auth + transaction streams:**

```bash
python cybersec/log_generator/auth_log_synth.py
```

Creates `data/logs/auth_events.jsonl`, `data/logs/transaction_events.jsonl`, and `data/logs/attack_labels.json` (the ground truth used to compute precision/recall).

**Terminal 2 — start the scoring API:**

```bash
uvicorn integration.api.main:app --port 8000
```

OpenAPI docs at `http://localhost:8000/docs`.

**Terminal 3 — run the correlation engine:**

```bash
python -m cybersec.correlation_engine
```

Processes ~26K events; takes ~2–5 minutes. Writes `data/logs/correlation_alerts.jsonl`.

**Then launch the dashboard:**

```bash
streamlit run integration/dashboard/app.py
```

Opens at `http://localhost:8501`.

---

## Roadmap

Items deferred from the capstone scope, in priority order:

- **React frontend** — replace Streamlit with a TypeScript/React dashboard using the same `/alerts` endpoints.
- **Kafka streaming** — replace `stream_simulator.py` with a Kafka consumer; same producer/consumer interfaces.
- **Model drift monitoring** — Evidently AI or custom KS-tests on score distributions, with retraining triggers.
- **Adversarial robustness evaluation** — red-team the model against synthetic adversarial inputs.
- **SIEM integration** — feed Wazuh or Splunk alerts upstream of the correlation engine.
- **Production feature store** — replace the per-transaction feature payload with a service-side feature assembly layer.

---

## Author

**Kartik Khera** — B.Tech (CyberSecurity), VIT Vellore '28.
PMO Intern at KPMG. Interested in fraud ML, financial systems engineering, and applied security.

- LinkedIn: [linkedin.com/in/kartikkhera](https://linkedin.com/in/kartikkhera)
- GitHub: [@anorak1709](https://github.com/anorak1709)

If you're hiring for a fraud-ML, risk-engineering, or fintech-security internship — I'd love to talk.

---

## License

MIT. See `LICENSE`.

