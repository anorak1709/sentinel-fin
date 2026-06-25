"""End-to-end smoke test for the scoring API.

Pulls one real transaction from the holdout parquet, sends it through /score,
and verifies the response is sensible.
"""
import json
from pathlib import Path

import httpx
import pandas as pd

API_URL = "http://localhost:8000"
DATA = Path("data/processed/features_ready.parquet")

# Load a few real transactions from the test split
df = pd.read_parquet(DATA)
cutoff_dt = df["TransactionDT"].quantile(0.80)
test_df = df[df["TransactionDT"] > cutoff_dt]

# Pick one known fraud and one known non-fraud
fraud_row = test_df[test_df["isFraud"] == 1].iloc[0]
clean_row = test_df[test_df["isFraud"] == 0].iloc[0]


def score_row(row: pd.Series, label: str) -> None:
    """Send one row through /score and print the response."""
    # Drop columns the model shouldn't see
    features = row.drop(["TransactionID", "isFraud", "TransactionDT"]).to_dict()

    # Replace NaN with None for JSON serialization
    features = {k: (None if pd.isna(v) else v) for k, v in features.items()}

    payload = {
        "transaction_id": f"smoke_{label}_{int(row['TransactionID'])}",
        "features": features,
    }

    r = httpx.post(f"{API_URL}/score", json=payload, timeout=30.0)
    print(f"\n=== {label.upper()} (actual: {'fraud' if row['isFraud']==1 else 'clean'}) ===")
    print(f"  HTTP {r.status_code}")
    if r.status_code == 200:
        resp = r.json()
        print(f"  fraud_probability: {resp['fraud_probability']:.4f}")
        print(f"  risk_tier:         {resp['risk_tier']}")
        print(f"  latency_ms:        {resp['scoring_latency_ms']}")
        print(f"  top 3 SHAP features:")
        for f in resp["top_shap_features"][:3]:
            print(f"    {f['feature']:25s} value={f['value']}  contrib={f['contribution']:+.3f}")
    else:
        print(f"  ERROR: {r.text}")


if __name__ == "__main__":
    print("Testing 5 fraud rows and 5 clean rows...\n")
    fraud_samples = test_df[test_df["isFraud"] == 1].sample(5, random_state=42)
    clean_samples = test_df[test_df["isFraud"] == 0].sample(5, random_state=42)

    for i, (_, row) in enumerate(fraud_samples.iterrows()):
        score_row(row, f"fraud_{i}")
    for i, (_, row) in enumerate(clean_samples.iterrows()):
        score_row(row, f"clean_{i}")
