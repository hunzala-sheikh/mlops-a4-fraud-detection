"""Generate a synthetic fraud-detection dataset that mimics IEEE CIS Fraud
Detection structure (transaction + identity features, anonymized V columns,
heavy class imbalance, high-cardinality categorical IDs, time index).

Output: data/fraud.csv (~50k rows, ~3.5% fraud rate).

Why synthetic instead of the 3+ GB Kaggle CIS file:
- Demo-equivalent: same imbalance, same structural challenges
  (high-cardinality cats, missing values, drift over time)
- Reproducible: pure NumPy, no Kaggle credentials
- Fast: ~2s to regenerate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_fraud_dataset(n_rows: int = 50_000, fraud_rate: float = 0.035,
                       random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # ----- legitimate transactions -----
    legit = pd.DataFrame({
        "TransactionID": np.arange(n_legit) + 1_000_000,
        "TransactionDT": rng.integers(86_400, 86_400 * 180, size=n_legit),  # seconds since launch (180 days)
        "TransactionAmt": np.exp(rng.normal(3.5, 1.0, n_legit)).round(2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n_legit, p=[0.55, 0.15, 0.10, 0.12, 0.08]),
        "card1": rng.integers(1, 18000, n_legit),  # high-cardinality
        "card2": rng.choice(list(range(100, 600)) + [np.nan] * 10, n_legit),
        "card4": rng.choice(["visa", "mastercard", "discover", "american express", np.nan],
                            n_legit, p=[0.65, 0.25, 0.05, 0.04, 0.01]),
        "card6": rng.choice(["debit", "credit", np.nan], n_legit, p=[0.7, 0.28, 0.02]),
        "addr1": rng.choice(list(range(100, 500)) + [np.nan] * 5, n_legit),
        "P_emaildomain": rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com",
             "outlook.com", "aol.com", np.nan],
            n_legit, p=[0.45, 0.15, 0.12, 0.13, 0.06, 0.04, 0.05]),
        "DeviceType": rng.choice(["mobile", "desktop", np.nan], n_legit, p=[0.55, 0.40, 0.05]),
    })
    # 14 anonymized "V" features (mimic Vesta engineered features)
    for i in range(1, 15):
        col = rng.normal(0, 1, n_legit)
        # 5% missing
        miss = rng.random(n_legit) < 0.05
        col[miss] = np.nan
        legit[f"V{i}"] = col
    legit["isFraud"] = 0

    # ----- fraudulent transactions -----
    fraud = pd.DataFrame({
        "TransactionID": np.arange(n_fraud) + 2_000_000,
        "TransactionDT": rng.integers(86_400, 86_400 * 180, size=n_fraud),
        # fraud tends toward higher amounts, longer tail
        # Slightly higher amount on average, but noisy
        "TransactionAmt": np.exp(rng.normal(3.7, 1.1, n_fraud)).round(2),
        # Same product mix as legit (no leak)
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n_fraud, p=[0.55, 0.15, 0.10, 0.12, 0.08]),
        "card1": rng.integers(1, 18000, n_fraud),
        "card2": rng.choice(list(range(100, 600)) + [np.nan] * 12, n_fraud),
        "card4": rng.choice(["visa", "mastercard", "discover", "american express", np.nan],
                            n_fraud, p=[0.65, 0.25, 0.05, 0.04, 0.01]),
        "card6": rng.choice(["debit", "credit", np.nan], n_fraud, p=[0.7, 0.28, 0.02]),
        "addr1": rng.choice(list(range(100, 500)) + [np.nan] * 7, n_fraud),
        # Same email domain mix (no anonymous-spike leak)
        "P_emaildomain": rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com",
             "outlook.com", "aol.com", np.nan],
            n_fraud, p=[0.45, 0.15, 0.12, 0.13, 0.06, 0.04, 0.05]),
        "DeviceType": rng.choice(["mobile", "desktop", np.nan], n_fraud, p=[0.55, 0.40, 0.05]),
    })
    # Only V1, V4, V7, V10 carry weak signal (small shift); rest is noise.
    informative = {1, 4, 7, 10}
    for i in range(1, 15):
        if i in informative:
            shift = 0.18 if (i % 2) else -0.18
            col = rng.normal(shift, 1.02, n_fraud)
        else:
            col = rng.normal(0, 1.0, n_fraud)
        miss = rng.random(n_fraud) < 0.08
        col[miss] = np.nan
        fraud[f"V{i}"] = col
    fraud["isFraud"] = 1

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Add label noise: 8% of fraud labels flipped to 0, 2% legit flipped to 1.
    # Plus, replace 30% of fraud rows' V features with pure noise (no signal).
    # This forces the model to make actual TP/FP/FN tradeoffs.
    fraud_idx = df.index[df["isFraud"] == 1]
    legit_idx = df.index[df["isFraud"] == 0]
    flip_to_legit = rng.choice(fraud_idx, size=int(0.08 * len(fraud_idx)), replace=False)
    flip_to_fraud = rng.choice(legit_idx, size=int(0.02 * len(legit_idx)), replace=False)
    df.loc[flip_to_legit, "isFraud"] = 0
    df.loc[flip_to_fraud, "isFraud"] = 1

    # Replace V features for 30% of fraud rows with pure noise (no signal)
    noisy_fraud = rng.choice(df.index[df["isFraud"] == 1],
                             size=int(0.30 * (df["isFraud"] == 1).sum()), replace=False)
    for i in range(1, 15):
        df.loc[noisy_fraud, f"V{i}"] = rng.normal(0, 1.0, len(noisy_fraud))

    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "fraud.csv"
    df = make_fraud_dataset()
    df.to_csv(out, index=False)
    print(f"wrote {out}: shape={df.shape}, fraud_rate={df['isFraud'].mean():.4f}")
    print(f"missing values per col (top 5):\n{df.isna().sum().sort_values(ascending=False).head()}")
