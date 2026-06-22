# EDA Findings — IEEE-CIS Fraud Detection

## Dataset shape
- 590,540 transactions × 394 columns (transaction table)
- 144,233 identities × 41 columns (identity table)
- Merged via left join → 590,540 × 434
- Time span: 182 days (~6 months)

## Class balance
- Fraud rate: 3.50%
- Fraud count: 20,663 ; Non-fraud: 569,877
- ~28:1 imbalance — handled via `scale_pos_weight` in XGBoost, no SMOTE needed

## Missingness
- 12 cols >90% missing → drop
- 214 cols >50% missing → keep (XGBoost handles NaN)
- 20 cols never missing (the foundation: TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD, card1, C1–C10)

## Identity coverage (key finding)
- Non-fraud: 23.3% have identity data
- Fraud: 54.8% have identity data
- Ratio: 2.35× → identity-data-present is itself a fraud signal
- Decision: left join, not inner join, to preserve population distribution

## Transaction amount (counterintuitive finding)
- Median fraud: ₹75 ; median non-fraud: ₹68.50 (only 9% higher)
- Max fraud: ₹5,191 ; max non-fraud: ₹31,937
- → Amount alone is a WEAK feature. The largest transactions are mostly legitimate.
- → Fraud detection here must rely on contextual features, not amount thresholds.

## Categorical signals
- **ProductCD: strong signal** — C: 11.7%, S: 5.9%, H: 4.8%, R: 3.8%, W: 2.0% (5.7× spread)
- **card4: weak** — Visa/MC/AmEx all near global rate; only Discover stands out (7.7%) on small sample (6.7K)

## Temporal structure
- Clear hour-of-day spike in late night / early morning (cardholder-absent fraud pattern)
- Day-level burst(s) visible across the 6 months (likely coordinated campaign)
- → Random train/test split would leak; use temporal split
- → Build hour-cyclical and day-of-dataset features

## Modeling implications
1. Temporal train/test split (first 5 months train, last month holdout)
2. Drop the 12 >90%-missing columns
3. Keep all sparse identity columns
4. Add a `has_identity` binary feature
5. Engineer time features: hour (cyclical sin/cos), day-of-dataset
6. Engineer amount features: log(amount), amount-vs-card1-median
7. Frequency-encode high-cardinality categoricals (card1, addr1)
8. XGBoost handles native NaN, no imputation needed
9. Class imbalance: scale_pos_weight ≈ 28 (=non-fraud/fraud ratio)
10. Primary metric: PR-AUC. Secondary: recall at 30% precision