"""Train XGBoost / LightGBM / RF+RFE on fraud.csv (Task 3), produce:
  - artifacts/deployed_model.pkl  (XGBoost, packaged with feature_order)
  - screenshots/03_confusion_matrix.png  (Task 3 requirement)
  - screenshots/03_models_summary.txt    (real numbers from this run)

The model artifact written here is generated locally and is loaded back by
our own inference Pod from a trusted ConfigMap mount; it is never loaded
from untrusted input.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fraud.csv"
ART = ROOT / "artifacts"
SHOTS = ROOT / "screenshots"

API_FEATURES = [
    "TransactionAmt", "ProductCD", "card1", "card4", "card6",
    "P_emaildomain", "DeviceType", "V1", "V2", "V3", "V4", "V5",
]


def load_df() -> pd.DataFrame:
    return pd.read_csv(DATA)


def encode(df: pd.DataFrame):
    encoders: dict[str, LabelEncoder] = {}
    df = df.copy()
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].fillna("missing")
        le = LabelEncoder().fit(df[c].astype(str))
        df[c] = le.transform(df[c].astype(str))
        encoders[c] = le
    for c in df.select_dtypes("number").columns:
        df[c] = df[c].fillna(df[c].median())
    return df, encoders


def train_xgb(X_tr, y_tr):
    from xgboost import XGBClassifier
    pos = max(int(y_tr.sum()), 1); neg = len(y_tr) - pos
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        random_state=42, n_jobs=4, eval_metric="auc",
        scale_pos_weight=neg / pos, tree_method="hist",
    ).fit(X_tr, y_tr)


def train_lgbm(X_tr, y_tr):
    from lightgbm import LGBMClassifier
    pos = max(int(y_tr.sum()), 1); neg = len(y_tr) - pos
    return LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        random_state=42, n_jobs=4, verbose=-1,
        scale_pos_weight=neg / pos,
    ).fit(X_tr, y_tr)


class _RFEPipeline:
    def __init__(self, sel, mdl):
        self.sel, self.mdl = sel, mdl
    def predict(self, X): return self.mdl.predict(self.sel.transform(X))
    def predict_proba(self, X): return self.mdl.predict_proba(self.sel.transform(X))


def train_rf_fs(X_tr, y_tr):
    base = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=4)
    sel = RFE(base, n_features_to_select=min(20, X_tr.shape[1])).fit(X_tr, y_tr)
    final = RandomForestClassifier(
        n_estimators=300, max_depth=14, random_state=42,
        n_jobs=4, class_weight="balanced",
    ).fit(sel.transform(X_tr), y_tr)
    return _RFEPipeline(sel, final)


def metric_dict(y, yp, yp_proba):
    return {
        "precision": precision_score(y, yp, zero_division=0),
        "recall": recall_score(y, yp, zero_division=0),
        "f1": f1_score(y, yp, zero_division=0),
        "auc_roc": roc_auc_score(y, yp_proba),
        "ap": average_precision_score(y, yp_proba),
    }


def main() -> None:
    print("[load]", DATA)
    df_raw = load_df()
    df, encoders = encode(df_raw)
    y = df["isFraud"].astype(int).values
    feature_cols = [c for c in df.columns if c != "isFraud"]
    X = df[feature_cols].values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    print(f"[split] train={len(y_tr)} test={len(y_te)} fraud_rate={y.mean():.4f}")

    trainers = {
        "XGBoost": train_xgb,
        "LightGBM": train_lgbm,
        "RF + RFE (hybrid)": train_rf_fs,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Task 3 - Confusion Matrix per model (test split)", fontsize=14)
    rows = []
    for ax, (name, fn) in zip(axes, trainers.items()):
        print(f"[train] {name}")
        m = fn(X_tr, y_tr)
        yp = m.predict(X_te)
        ypp = m.predict_proba(X_te)[:, 1]
        cm = confusion_matrix(y_te, yp, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        d = metric_dict(y_te, yp, ypp)
        ax.set_title(
            f"{name}\nP={d['precision']:.3f}  R={d['recall']:.3f}  "
            f"F1={d['f1']:.3f}\nAUC-ROC={d['auc_roc']:.3f}  AP={d['ap']:.3f}",
            fontsize=10)
        rows.append((name, d, cm))

    plt.tight_layout()
    SHOTS.mkdir(parents=True, exist_ok=True)
    out = SHOTS / "03_confusion_matrix.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[save] {out}")

    lines = ["=== A4 Task 3 - Model complexity comparison (real run on fraud.csv) ===", ""]
    lines.append("| model              | precision | recall | f1    | auc_roc | ap    |  TN  | FP | FN |  TP |")
    lines.append("|--------------------|-----------|--------|-------|---------|-------|------|----|----|-----|")
    for name, d, cm in rows:
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        lines.append(
            f"| {name:<18} | {d['precision']:.3f}     | {d['recall']:.3f}  | "
            f"{d['f1']:.3f} | {d['auc_roc']:.3f}   | {d['ap']:.3f} | "
            f"{tn:>4} | {fp:>2} | {fn:>2} | {tp:>3} |")
    lines.append("")
    lines.append("Confusion matrix is computed for the fraud class on the held-out 20% test")
    lines.append("split (n_test = " + str(len(y_te)) + ").")
    (SHOTS / "03_models_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"[save] {SHOTS / '03_models_summary.txt'}")

    # Inference-ready stub model trained on ONLY the API fields.
    print("[train] inference-stub model on", len(API_FEATURES), "API fields")
    api_X = df[API_FEATURES].values.astype(float)
    api_X_tr, _, api_y_tr, _ = train_test_split(
        api_X, y, test_size=0.20, stratify=y, random_state=42)
    api_model = train_xgb(api_X_tr, api_y_tr)
    bundle = {
        "model": api_model,
        "model_type": "xgboost",
        "feature_order": API_FEATURES,
        "encoders": {k: list(v.classes_) for k, v in encoders.items() if k in API_FEATURES},
        "version": "real-run-2026-05-07",
    }
    ART.mkdir(parents=True, exist_ok=True)
    out_pkl = ART / "deployed_model.pkl"
    joblib.dump(bundle, out_pkl)
    print(f"[save] {out_pkl}  ({out_pkl.stat().st_size//1024} KB)")

    summary = {
        "data_rows": int(len(df)),
        "test_rows": int(len(y_te)),
        "fraud_rate": float(y.mean()),
        "models": {name: d for name, d, _ in rows},
        "api_model_features": API_FEATURES,
    }
    (ART / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
