"""Diagnostic: compare API output vs direct model inference on the same row."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path("ml/models")
DATA = Path("data/processed/features_ready.parquet")

# Load artifacts directly (mirrors notebook 03/04)
xgb_clf = joblib.load(ARTIFACTS / "xgboost_model.joblib")
category_maps = joblib.load(ARTIFACTS / "category_maps.joblib")
with open(ARTIFACTS / "model_metadata.json") as f:
    metadata = json.load(f)

# Load test row exactly as notebook does
df = pd.read_parquet(DATA)
cutoff_dt = df["TransactionDT"].quantile(0.80)
test_df = df[df["TransactionDT"] > cutoff_dt]
fraud_row = test_df[test_df["isFraud"] == 1].iloc[0]

# Direct path: replicate notebook 03's encoding
all_features = metadata["feature_columns"]
X = df.loc[[fraud_row.name], all_features].copy()
for col, mapping in category_maps.items():
    if col in X.columns:
        X[col] = (
            X[col].fillna("__missing__")
            .map(mapping)
            .fillna(len(mapping))
            .astype(np.int32)
        )

proba_direct = xgb_clf.predict_proba(X)[0, 1]
print(f"Direct path (notebook-style):  {proba_direct:.4f}")
print(f"X column dtypes sample: {dict(X.dtypes.head(10))}")
print(f"\nFirst 5 categorical values:")
for col in list(category_maps.keys())[:5]:
    if col in X.columns:
        print(f"  {col}: {X[col].iloc[0]} (dtype: {X[col].dtype})")

# Compare against what the API would receive
features_dict = fraud_row.drop(["TransactionID", "isFraud", "TransactionDT"]).to_dict()
features_dict = {k: (None if pd.isna(v) else v) for k, v in features_dict.items()}
print(f"\nFirst 5 values as API receives them:")
for col in list(category_maps.keys())[:5]:
    if col in features_dict:
        print(f"  {col}: {features_dict[col]} (Python type: {type(features_dict[col]).__name__})")