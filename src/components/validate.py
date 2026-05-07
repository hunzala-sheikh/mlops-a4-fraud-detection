"""Component 2: Data Validation.

Schema check, missing-value thresholds, fraud-rate sanity check.
Raises if data quality fails — supports retry via component-level retries.
"""
from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(base_image="mlops-a4-base:v1")
def validate_data(
    raw_data: Input[Dataset],
    validated: Output[Dataset],
    max_missing_pct: float = 50.0,
    min_fraud_rate: float = 0.005,
    max_fraud_rate: float = 0.20,
) -> str:
    import json

    import pandas as pd

    df = pd.read_csv(raw_data.path)

    # Schema: required columns
    required = {"TransactionID", "TransactionAmt", "isFraud"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"[VALIDATE] schema mismatch: missing {missing_cols}")

    # Per-column missing %
    miss_pct = (df.isna().mean() * 100).round(2).to_dict()
    bad = {k: v for k, v in miss_pct.items() if v > max_missing_pct}
    if bad:
        raise RuntimeError(f"[VALIDATE] columns over {max_missing_pct}% missing: {bad}")

    # Fraud rate sanity
    fr = df["isFraud"].mean()
    if not (min_fraud_rate <= fr <= max_fraud_rate):
        raise RuntimeError(f"[VALIDATE] fraud_rate {fr:.4f} outside [{min_fraud_rate}, {max_fraud_rate}]")

    df.to_csv(validated.path, index=False)
    report = {"n_rows": len(df), "fraud_rate": float(fr), "max_missing_col_pct": float(max(miss_pct.values()))}
    print(f"[VALIDATE] OK: {report}")
    return json.dumps(report)
