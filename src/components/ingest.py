"""Component 1: Data Ingestion.

Reads the IEEE-CIS-style fraud-detection CSV (50k rows, 26 cols) into a KFP
Dataset artifact. The CSV is expected to be available inside the container
at /data/fraud.csv (mounted via PVC) or generated on-the-fly from the seed.
"""
from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(base_image="mlops-a4-base:v1")
def ingest_data(raw_data: Output[Dataset]) -> str:
    """Load fraud CSV. If not present, regenerate from seed."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    candidates = ["/data/fraud.csv", "/mnt/data/fraud.csv", "data/fraud.csv"]
    src = next((p for p in candidates if Path(p).exists()), None)

    if src is None:
        # Inline generator — synthetic dataset that mimics IEEE CIS structure
        rng = np.random.default_rng(42)
        n_rows, fraud_rate = 50_000, 0.035
        n_fraud = int(n_rows * fraud_rate)
        n_legit = n_rows - n_fraud

        def _block(n, base, shift_v):
            d = pd.DataFrame({
                "TransactionID": np.arange(n) + base,
                "TransactionDT": rng.integers(86_400, 86_400 * 180, size=n),
                "TransactionAmt": np.exp(rng.normal(3.5 + shift_v * 0.7, 1.0, n)).round(2),
                "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n),
                "card1": rng.integers(1, 18000, n),
                "card4": rng.choice(["visa", "mastercard", "discover", "amex", np.nan], n),
                "card6": rng.choice(["debit", "credit", np.nan], n),
                "P_emaildomain": rng.choice(
                    ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", np.nan], n),
                "DeviceType": rng.choice(["mobile", "desktop", np.nan], n),
            })
            for i in range(1, 15):
                col = rng.normal(shift_v * (i % 3 - 1) * 0.6, 1.0 + shift_v * 0.1, n)
                col[rng.random(n) < 0.05 + shift_v * 0.03] = np.nan
                d[f"V{i}"] = col
            d["isFraud"] = int(shift_v > 0)
            return d

        df = pd.concat([_block(n_legit, 1_000_000, 0.0),
                        _block(n_fraud, 2_000_000, 1.0)], ignore_index=True)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        df = pd.read_csv(src)

    df.to_csv(raw_data.path, index=False)
    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "fraud_rate": float(df["isFraud"].mean()),
        "missing_total": int(df.isna().sum().sum()),
    }
    print(f"[INGEST] {summary}")
    return json.dumps(summary)
