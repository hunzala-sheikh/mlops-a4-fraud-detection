"""A4 Task 7: Time-based drift simulation.

We split data by TransactionDT (earlier 80% as train, later 20% as test, then
inject *new* fraud patterns and a feature-importance shift in the test
window). Compare model trained on early-only vs early+late re-training.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_local import _ingest, _validate, _feature_engineer, _train, _evaluate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def split_with_drift(df: pd.DataFrame, *, drift_strength: float = 0.5, random_state: int = 42):
    """Split by TransactionDT and inject new fraud patterns into the test slice."""
    rng = np.random.default_rng(random_state)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Inject NEW fraud pattern into test:
    # - shift V2 strongly upward for fraud rows
    # - bump TransactionAmt for half of test fraud cases
    fraud_mask = test["isFraud"] == 1
    n_fraud_test = int(fraud_mask.sum())
    if n_fraud_test:
        if "V2" in test.columns:
            test.loc[fraud_mask, "V2"] = test.loc[fraud_mask, "V2"].fillna(0) + drift_strength * 1.0
        # Mutate TransactionAmt
        test.loc[fraud_mask, "TransactionAmt"] = (
            test.loc[fraud_mask, "TransactionAmt"] * (1 + drift_strength * rng.uniform(0, 1, n_fraud_test))
        )
    return train, test


def _prep_for_model(train_df, test_df):
    """Same numeric prep as preprocess_data, but no resampling."""
    y_train = train_df["isFraud"].values.astype(int)
    y_test = test_df["isFraud"].values.astype(int)

    Xtr = train_df.drop(columns=["isFraud"]).copy()
    Xte = test_df.drop(columns=["isFraud"]).copy()

    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = Xtr.select_dtypes(include=["object"]).columns.tolist()
    for c in num_cols:
        med = Xtr[c].median()
        Xtr[c] = Xtr[c].fillna(med)
        Xte[c] = Xte[c].fillna(med)
    for c in cat_cols:
        mode_val = Xtr[c].mode().iloc[0] if not Xtr[c].mode().empty else "MISSING"
        Xtr[c] = Xtr[c].fillna(mode_val)
        Xte[c] = Xte[c].fillna(mode_val)

    high_card = [c for c in cat_cols if Xtr[c].nunique() > 30]
    low_card = [c for c in cat_cols if c not in high_card]
    for c in high_card:
        gm = y_train.mean()
        agg = pd.DataFrame({"y": y_train, "x": Xtr[c].values}).groupby("x")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + 10.0 * gm) / (agg["count"] + 10.0)
        Xtr[c + "_te"] = Xtr[c].map(enc).fillna(gm)
        Xte[c + "_te"] = Xte[c].map(enc).fillna(gm)
        Xtr = Xtr.drop(columns=[c]); Xte = Xte.drop(columns=[c])
    if low_card:
        Xtr = pd.get_dummies(Xtr, columns=low_card, dummy_na=False, drop_first=True)
        Xte = pd.get_dummies(Xte, columns=low_card, dummy_na=False, drop_first=True)
        # align columns
        for c in Xtr.columns:
            if c not in Xte.columns:
                Xte[c] = 0
        Xte = Xte[Xtr.columns]

    Xtr = Xtr.select_dtypes(include=[np.number])
    Xte = Xte[Xtr.columns]
    Xtr = _feature_engineer(Xtr)
    Xte = _feature_engineer(Xte[Xtr.columns.intersection(Xte.columns).tolist()] if False else Xte)
    common = [c for c in Xtr.columns if c in Xte.columns]
    return Xtr[common], Xte[common], y_train, y_test


def main():
    df = _validate(_ingest())
    rows = []
    for drift in [0.0, 0.3, 0.6, 1.0]:
        train_df, test_df = split_with_drift(df, drift_strength=drift)
        Xtr, Xte, ytr, yte = _prep_for_model(train_df, test_df)
        model, _ = _train(Xtr, ytr, model_type="xgboost",
                          cost_sensitive=False, fn_penalty=1.0)
        s = _evaluate(model, Xte, yte)
        s["drift_strength"] = drift
        rows.append(s)
        print(f"drift={drift:.1f}  recall={s['recall']:.4f}  auc={s['auc_roc']:.4f}  cost=${s['business_cost_usd']:.0f}")

    out = RES / "task7_drift.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
