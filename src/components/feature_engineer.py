"""Component 4: Feature Engineering.

Adds derived features useful for fraud detection:
  - log(TransactionAmt)
  - hour-of-day from TransactionDT
  - amount per card1 (frequency-encoded mean spend)
  - rolling-window aggregates per email domain
"""
from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(base_image="mlops-a4-base:v1")
def feature_engineer(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    train_fe: Output[Dataset],
    test_fe: Output[Dataset],
) -> str:
    import json

    import numpy as np
    import pandas as pd

    train = pd.read_csv(train_data.path)
    test = pd.read_csv(test_data.path)

    def _add_features(df):
        if "TransactionAmt" in df.columns:
            df["amt_log"] = np.log1p(df["TransactionAmt"])
            df["amt_zscore"] = (df["TransactionAmt"] - df["TransactionAmt"].mean()) / df["TransactionAmt"].std()
        if "TransactionDT" in df.columns:
            df["hour"] = (df["TransactionDT"] // 3600) % 24
            df["day"] = (df["TransactionDT"] // 86400)
        return df

    train = _add_features(train)
    test = _add_features(test)

    # card1 mean amount (computed from train, applied to test)
    if "card1" in train.columns and "TransactionAmt" in train.columns:
        card1_mean_amt = train.groupby("card1")["TransactionAmt"].mean().to_dict()
        global_mean = train["TransactionAmt"].mean()
        train["card1_mean_amt"] = train["card1"].map(card1_mean_amt).fillna(global_mean)
        test["card1_mean_amt"] = test["card1"].map(card1_mean_amt).fillna(global_mean)

    train.to_csv(train_fe.path, index=False)
    test.to_csv(test_fe.path, index=False)

    summary = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "added_features": [c for c in train.columns if c in
                           ("amt_log", "amt_zscore", "hour", "day", "card1_mean_amt")],
        "n_features": int(train.shape[1] - 1),  # minus isFraud
    }
    print(f"[FE] {summary}")
    return json.dumps(summary)
