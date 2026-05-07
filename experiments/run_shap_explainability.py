"""A4 Task 9: Explainability with SHAP.

Trains an XGBoost model on the fraud dataset, then:
  - Global feature importance bar chart (SHAP mean |value|)
  - Beeswarm plot (sign + magnitude per feature)
  - Per-instance waterfall plot for one fraud-flagged row (answer: "why is this predicted fraud?")
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from run_local import _ingest, _validate, _feature_engineer, _train, _evaluate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)


def _prep_full(df):
    y = df["isFraud"].values.astype(int)
    X = df.drop(columns=["isFraud"]).copy()
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    for c in num:
        X[c] = X[c].fillna(X[c].median())
    for c in cat:
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "MISSING")
    high = [c for c in cat if X[c].nunique() > 30]
    low = [c for c in cat if c not in high]
    for c in high:
        gm = y.mean()
        agg = pd.DataFrame({"y": y, "x": X[c].values}).groupby("x")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + 10 * gm) / (agg["count"] + 10)
        X[c + "_te"] = X[c].map(enc).fillna(gm)
        X = X.drop(columns=[c])
    if low:
        X = pd.get_dummies(X, columns=low, dummy_na=False, drop_first=True)
    X = X.select_dtypes(include=[np.number])
    X = _feature_engineer(X)
    return X, y


def main():
    df = _validate(_ingest())
    X, y = _prep_full(df)
    # train/test split (stratified)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model, _ = _train(Xtr, ytr, model_type="xgboost", cost_sensitive=False, fn_penalty=1.0,
                      n_estimators=100, max_depth=5)
    # use a sample for SHAP (5000 rows is enough for stable estimates)
    sample = Xte.sample(n=min(5000, len(Xte)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # 1) Global feature importance (bar)
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP - global feature importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIG / "task9_shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 2) Beeswarm
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.title("SHAP - beeswarm (sign + magnitude)")
    plt.tight_layout()
    plt.savefig(FIG / "task9_shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 3) Single-instance waterfall (why predicted fraud?)
    proba = model.predict_proba(sample.values)[:, 1]
    flagged_idx = int(np.argmax(proba))
    explanation = shap.Explanation(
        values=shap_values[flagged_idx],
        base_values=explainer.expected_value,
        data=sample.iloc[flagged_idx].values,
        feature_names=sample.columns.tolist(),
    )
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"Why is row {sample.index[flagged_idx]} predicted fraud (proba={proba[flagged_idx]:.3f})?")
    plt.tight_layout()
    plt.savefig(FIG / "task9_shap_waterfall.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 4) Top features summary as JSON
    importances = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    summary = {
        "n_samples_explained": int(len(sample)),
        "top_15_features": importances.head(15).to_dict(orient="records"),
        "highest_proba_row": {
            "index": int(sample.index[flagged_idx]),
            "predicted_proba": float(proba[flagged_idx]),
            "actual_label": int(yte[Xte.index.get_loc(sample.index[flagged_idx])]) if sample.index[flagged_idx] in Xte.index else -1,
            "top_5_drivers": [
                {"feature": str(f), "shap_value": float(v)}
                for f, v in sorted(zip(sample.columns, shap_values[flagged_idx]),
                                   key=lambda kv: abs(kv[1]), reverse=True)[:5]
            ],
        },
    }
    (RES / "task9_shap.json").write_text(json.dumps(summary, indent=2))
    print(f"top features:\n{importances.head(10).to_string(index=False)}")
    print(f"\nWhy is row {sample.index[flagged_idx]} flagged fraud (proba={proba[flagged_idx]:.3f}):")
    for d in summary["highest_proba_row"]["top_5_drivers"]:
        print(f"  {d['feature']:30s} shap={d['shap_value']:+.4f}")
    print(f"\nwrote {RES / 'task9_shap.json'}")
    print(f"figures: {FIG}/task9_shap_*.png")


if __name__ == "__main__":
    main()
