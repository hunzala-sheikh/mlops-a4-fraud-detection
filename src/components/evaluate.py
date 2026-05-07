"""Component 6: Model Evaluation.

Per A4 Task 3 (mandatory metrics) and Task 4 (cost analysis):
  - Precision, Recall, F1, AUC-ROC
  - Confusion matrix for fraud class (rendered in KFP UI)
  - Business cost: estimated fraud-loss + false-alarm cost
"""
from kfp import dsl
from kfp.dsl import HTML, ClassificationMetrics, Dataset, Input, Metrics, Model, Output


@dsl.component(base_image="mlops-a4-base:v1")
def evaluate_model(
    test_fe: Input[Dataset],
    model_in: Input[Model],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    report_html: Output[HTML],
    fraud_cost_per_case: float = 1000.0,    # avg loss when fraud is missed
    fp_cost_per_case: float = 50.0,         # cost of a wrongly flagged transaction
) -> str:
    import base64
    import io
    import json
    import pickle

    import matplotlib
    import numpy as np
    import pandas as pd
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    df = pd.read_csv(test_fe.path)
    y = df["isFraud"].values.astype(int)
    X = df.drop(columns=["isFraud"]).values
    with open(model_in.path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    scores = {
        "accuracy":  float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.0,
        "ap":        float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "business_cost_usd": float(fn * fraud_cost_per_case + fp * fp_cost_per_case),
        "fraud_loss_usd": float(fn * fraud_cost_per_case),
        "false_alarm_cost_usd": float(fp * fp_cost_per_case),
    }
    for k, v in scores.items():
        if isinstance(v, float):
            metrics.log_metric(k, v)

    # Confusion matrix artifact
    if cm.size == 4:
        classification_metrics.log_confusion_matrix(["legit", "fraud"], cm.tolist())

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["legit", "fraud"], yticklabels=["legit", "fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {bundle.get('model_type', '?')}")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    html = (
        f"<!doctype html><html><body style='font-family:sans-serif'>"
        f"<h2>Fraud-detection Eval -- {bundle.get('model_type','?')} "
        f"{'(cost-sensitive)' if bundle.get('cost_sensitive') else '(standard)'}</h2>"
        f"<p><b>Hyperparams:</b> {bundle.get('params')}</p>"
        f"<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        + "".join(f"<tr><td><b>{k}</b></td><td>{v:.4f}</td></tr>" if isinstance(v, float)
                  else f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in scores.items())
        + f"</table><br/><img src='data:image/png;base64,{png_b64}'/>"
          f"<p><b>Business cost:</b> ${scores['business_cost_usd']:.2f} "
          f"(fraud_loss={scores['fraud_loss_usd']:.2f} + "
          f"false_alarm={scores['false_alarm_cost_usd']:.2f})</p>"
          f"</body></html>"
    )
    with open(report_html.path, "w") as f:
        f.write(html)

    print(f"[EVAL] {json.dumps(scores)}")
    return json.dumps(scores)
