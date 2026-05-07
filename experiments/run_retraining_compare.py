"""A4 Task 8: Intelligent retraining strategy comparison.

Three strategies, each retraining over 5 monthly windows of (synthetic) data:
  - threshold:  retrain only when AUC drops > 0.03 from baseline
  - periodic:   retrain every month
  - hybrid:     retrain monthly OR on threshold violation, whichever first

Compare:
  - performance (mean AUC over 5 windows)
  - cost (compute cost = retrains x train_time)
  - stability (std AUC)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_local import _ingest, _validate, _feature_engineer, _train, _evaluate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
RETRAIN_COST_PER_RUN = 5.0   # synthetic compute-credits per retrain


def make_windows(df: pd.DataFrame, n_windows: int = 5):
    """Split by TransactionDT into 5 sliding windows (one 'month' each)."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    edges = np.linspace(0, len(df), n_windows + 1, dtype=int)
    return [df.iloc[edges[i]:edges[i + 1]].reset_index(drop=True) for i in range(n_windows)]


def _prep(train_df, test_df):
    y_tr = train_df["isFraud"].values.astype(int)
    y_te = test_df["isFraud"].values.astype(int)
    Xtr = train_df.drop(columns=["isFraud"]).copy()
    Xte = test_df.drop(columns=["isFraud"]).copy()
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat = Xtr.select_dtypes(include=["object"]).columns.tolist()
    for c in num:
        med = Xtr[c].median()
        Xtr[c] = Xtr[c].fillna(med); Xte[c] = Xte[c].fillna(med)
    for c in cat:
        mode_v = Xtr[c].mode().iloc[0] if not Xtr[c].mode().empty else "MISSING"
        Xtr[c] = Xtr[c].fillna(mode_v); Xte[c] = Xte[c].fillna(mode_v)
    high = [c for c in cat if Xtr[c].nunique() > 30]
    low = [c for c in cat if c not in high]
    for c in high:
        gm = y_tr.mean()
        agg = pd.DataFrame({"y": y_tr, "x": Xtr[c].values}).groupby("x")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + 10 * gm) / (agg["count"] + 10)
        Xtr[c + "_te"] = Xtr[c].map(enc).fillna(gm)
        Xte[c + "_te"] = Xte[c].map(enc).fillna(gm)
        Xtr = Xtr.drop(columns=[c]); Xte = Xte.drop(columns=[c])
    if low:
        Xtr = pd.get_dummies(Xtr, columns=low, dummy_na=False, drop_first=True)
        Xte = pd.get_dummies(Xte, columns=low, dummy_na=False, drop_first=True)
        for c in Xtr.columns:
            if c not in Xte.columns:
                Xte[c] = 0
        Xte = Xte[Xtr.columns]
    Xtr = Xtr.select_dtypes(include=[np.number]); Xte = Xte[Xtr.columns]
    Xtr = _feature_engineer(Xtr); Xte = _feature_engineer(Xte)
    common = [c for c in Xtr.columns if c in Xte.columns]
    return Xtr[common], Xte[common], y_tr, y_te


def simulate(strategy: str, windows, *, threshold_drop=0.03):
    """Walk forward over windows[1:], retrain per strategy, score on next window."""
    aucs, retrained_at, total_compute_s = [], [], 0.0
    base_train = windows[0]
    cur_model = None
    baseline_auc = None
    cost_credits = 0.0

    for i in range(1, len(windows)):
        # decide whether to retrain
        retrain = False
        if cur_model is None:
            retrain = True
            reason = "initial"
        elif strategy == "periodic":
            retrain = True
            reason = "periodic"
        elif strategy == "threshold":
            # measure on previous window first
            Xtr, Xte, ytr, yte = _prep(base_train, windows[i - 1])
            s = _evaluate(cur_model, Xte, yte)
            if baseline_auc is None: baseline_auc = s["auc_roc"]
            if (baseline_auc - s["auc_roc"]) > threshold_drop:
                retrain = True; reason = f"threshold drop {baseline_auc - s['auc_roc']:.4f}"
            else:
                reason = "no drop"
        elif strategy == "hybrid":
            # periodic every 2 windows OR on threshold drop
            Xtr, Xte, ytr, yte = _prep(base_train, windows[i - 1])
            s = _evaluate(cur_model, Xte, yte)
            if baseline_auc is None: baseline_auc = s["auc_roc"]
            if (baseline_auc - s["auc_roc"]) > threshold_drop or i % 2 == 0:
                retrain = True
                reason = "hybrid"
            else:
                reason = "skip"

        if retrain:
            t0 = time.time()
            Xtr, Xte, ytr, yte = _prep(base_train, windows[i])
            cur_model, _ = _train(Xtr, ytr, model_type="xgboost",
                                  cost_sensitive=False, fn_penalty=1.0,
                                  n_estimators=100, max_depth=5)
            total_compute_s += time.time() - t0
            cost_credits += RETRAIN_COST_PER_RUN
            retrained_at.append({"window": i, "reason": reason})
            # rebase
            base_train = windows[i]
            baseline_auc = None
        else:
            # need test data prepped against the current base_train (from when last retrained)
            Xtr, Xte, ytr, yte = _prep(base_train, windows[i])

        s = _evaluate(cur_model, Xte, yte)
        aucs.append(s["auc_roc"])

    return {
        "strategy": strategy,
        "n_retrains": len(retrained_at),
        "cost_credits": cost_credits,
        "compute_seconds": round(total_compute_s, 2),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "auc_per_window": [float(a) for a in aucs],
        "retrained_at": retrained_at,
    }


def main():
    df = _validate(_ingest())
    windows = make_windows(df, n_windows=5)
    print(f"5 windows of sizes: {[len(w) for w in windows]}")

    rows = []
    for strategy in ["threshold", "periodic", "hybrid"]:
        r = simulate(strategy, windows)
        rows.append(r)
        print(f"{strategy:10s} mean_auc={r['mean_auc']:.4f} std={r['std_auc']:.4f} "
              f"retrains={r['n_retrains']} cost={r['cost_credits']:.0f}credits "
              f"compute={r['compute_seconds']:.1f}s")
    out = RES / "task8_retraining.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
