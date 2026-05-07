"""Run the full pipeline locally (without KFP) for fast iteration.

Used by experiments/run_imbalance_compare.py, run_cost_sensitive_compare.py,
run_models_compare.py, run_drift_simulation.py - all share the same
ingest+preprocess+FE+train+eval logic but vary the parameters.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)


HERE = Path(__file__).resolve().parents[1]
DATA_PATH = HERE / "data" / "fraud.csv"


def _ingest() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _validate(df: pd.DataFrame, max_missing_pct: float = 30.0) -> pd.DataFrame:
    miss_pct = (df.isna().mean() * 100)
    bad = miss_pct[miss_pct > max_missing_pct]
    if not bad.empty:
        raise RuntimeError(f"validation failed: cols over {max_missing_pct}% missing: {bad.to_dict()}")
    return df


def _preprocess(df: pd.DataFrame, *, test_size=0.2, imbalance_strategy="none",
                random_state=42, time_based_split=False):
    from sklearn.model_selection import train_test_split

    y = df["isFraud"].values.astype(int)
    X = df.drop(columns=["isFraud"]).copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "MISSING")

    high_card = [c for c in cat_cols if X[c].nunique() > 30]
    low_card = [c for c in cat_cols if c not in high_card]
    for c in high_card:
        global_mean = y.mean()
        agg = pd.DataFrame({"y": y, "x": X[c].values}).groupby("x")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + 10.0 * global_mean) / (agg["count"] + 10.0)
        X[c + "_te"] = X[c].map(enc).fillna(global_mean)
        X = X.drop(columns=[c])
    if low_card:
        X = pd.get_dummies(X, columns=low_card, dummy_na=False, drop_first=True)

    if time_based_split and "TransactionDT" in X.columns:
        order = X["TransactionDT"].argsort().values
        cut = int(len(X) * (1 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test[X_train.columns]

    if imbalance_strategy == "smote":
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, sampling_strategy=0.30)
        X_train_arr, y_train = sm.fit_resample(X_train.values, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
    elif imbalance_strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=random_state, sampling_strategy=0.30)
        X_train_arr, y_train = rus.fit_resample(X_train.values, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)

    return X_train, X_test, y_train, y_test


def _feature_engineer(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()
    if "TransactionAmt" in df.columns:
        df["amt_log"] = np.log1p(df["TransactionAmt"])
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600) % 24
    return df


def _train(X_train, y_train, *, model_type, cost_sensitive, fn_penalty,
           n_estimators=200, max_depth=6, random_state=42):
    pos = max(int((y_train == 1).sum()), 1)
    neg = (y_train == 0).sum()
    base_pos_weight = neg / pos
    pos_weight = base_pos_weight * fn_penalty if cost_sensitive else 1.0

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            random_state=random_state, n_jobs=1,
            scale_pos_weight=pos_weight, eval_metric="auc", tree_method="hist",
        ).fit(X_train, y_train)
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            random_state=random_state, n_jobs=1,
            scale_pos_weight=pos_weight, verbose=-1,
        ).fit(X_train, y_train)
    elif model_type == "rf_fs":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE
        base = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1)
        sel = RFE(base, n_features_to_select=min(20, X_train.shape[1])).fit(X_train, y_train)
        Xs = sel.transform(X_train)
        cw = {0: 1.0, 1: float(pos_weight)} if cost_sensitive else "balanced"
        m = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=1, class_weight=cw,
        ).fit(Xs, y_train)

        class _RFE:
            def __init__(self, sel, m): self.sel, self.m = sel, m
            def predict(self, X): return self.m.predict(self.sel.transform(X))
            def predict_proba(self, X): return self.m.predict_proba(self.sel.transform(X))
        model = _RFE(sel, m)
    else:
        raise ValueError(model_type)
    return model, float(pos_weight)


def _evaluate(model, X_test, y_test, *, fraud_cost=1000.0, fp_cost=50.0) -> dict[str, Any]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = (cm.ravel().tolist() if cm.size == 4 else (0, 0, 0, 0))
    return {
        "accuracy":  float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall":    float(recall_score(y_test, pred, zero_division=0)),
        "f1":        float(f1_score(y_test, pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.0,
        "ap":        float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "business_cost_usd": float(fn * fraud_cost + fp * fp_cost),
        "fraud_loss_usd": float(fn * fraud_cost),
        "false_alarm_cost_usd": float(fp * fp_cost),
    }


def run_full(*, model_type="xgboost", imbalance_strategy="none",
             cost_sensitive=False, fn_penalty=5.0, time_based_split=False,
             random_state=42, n_estimators=200, max_depth=6) -> dict[str, Any]:
    df = _ingest()
    df = _validate(df)
    Xtr, Xte, ytr, yte = _preprocess(df, imbalance_strategy=imbalance_strategy,
                                     random_state=random_state,
                                     time_based_split=time_based_split)
    Xtr = _feature_engineer(Xtr)
    Xte = _feature_engineer(Xte[Xtr.columns.tolist()[:Xte.shape[1]] if False else Xte.columns])
    # ensure same cols
    common = [c for c in Xtr.columns if c in Xte.columns]
    Xtr, Xte = Xtr[common], Xte[common]
    model, pw = _train(Xtr, ytr,
                       model_type=model_type, cost_sensitive=cost_sensitive,
                       fn_penalty=fn_penalty, n_estimators=n_estimators,
                       max_depth=max_depth, random_state=random_state)
    scores = _evaluate(model, Xte, yte)
    return {
        "config": {"model_type": model_type, "imbalance_strategy": imbalance_strategy,
                   "cost_sensitive": cost_sensitive, "fn_penalty": fn_penalty,
                   "scale_pos_weight": pw, "time_based_split": time_based_split},
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_fraud_rate": float(ytr.mean()), "test_fraud_rate": float(yte.mean()),
        **scores,
    }


if __name__ == "__main__":
    print(json.dumps(run_full(model_type="xgboost"), indent=2))
