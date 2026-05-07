"""Validate the expected schema of data/fraud.csv."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fraud.csv"


def main() -> int:
    if not DATA.exists():
        print(f"SKIP: {DATA} not present")
        return 0
    df = pd.read_csv(DATA)

    required = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"missing required columns: {missing}"

    fr = df["isFraud"].mean()
    assert 0.005 <= fr <= 0.30, f"fraud_rate {fr} outside reasonable range"

    miss_pct = (df.isna().mean() * 100)
    bad = miss_pct[miss_pct > 30]
    assert bad.empty, f"columns over 30% missing: {bad.to_dict()}"

    print(f"OK: shape={df.shape}, fraud_rate={fr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
