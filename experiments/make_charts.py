"""Generate report figures from results/*.json"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def task2():
    data = json.loads((RES / "task2_imbalance.json").read_text())
    names = [d["config"]["imbalance_strategy"] for d in data]
    rec = [d["recall"] for d in data]
    f1 = [d["f1"] for d in data]
    auc = [d["auc_roc"] for d in data]
    ap = [d["ap"] for d in data]
    cost = [d["business_cost_usd"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    w = 0.2
    axes[0].bar(x - 1.5 * w, rec, w, label="recall")
    axes[0].bar(x - 0.5 * w, f1, w, label="f1")
    axes[0].bar(x + 0.5 * w, auc, w, label="auc-roc")
    axes[0].bar(x + 1.5 * w, ap, w, label="avg-prec")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_title("Imbalance handling - quality metrics")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, cost, color="firebrick")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_title("Imbalance handling - business cost (USD)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "task2_imbalance.png", dpi=120)
    plt.close(fig)


def task3():
    data = json.loads((RES / "task3_models.json").read_text())
    names = [d["config"]["model_type"] for d in data]
    metrics = ["precision", "recall", "f1", "auc_roc", "ap"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    w = 0.16
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * w, [d[m] for d in data], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Task 3 - Model complexity comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "task3_models.png", dpi=120)
    plt.close(fig)


def task4():
    data = json.loads((RES / "task4_cost_sensitive.json").read_text())
    names = [d["config_name"] for d in data]
    fraud_loss = [d["fraud_loss_usd"] for d in data]
    fa_cost = [d["false_alarm_cost_usd"] for d in data]
    total = [d["business_cost_usd"] for d in data]
    recall = [d["recall"] for d in data]
    precision = [d["precision"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    axes[0].bar(x, fraud_loss, label="fraud loss (FN x $1000)", color="firebrick")
    axes[0].bar(x, fa_cost, bottom=fraud_loss, label="false alarm (FP x $50)", color="steelblue")
    axes[0].plot(x, total, "k--o", label="total")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_title("Cost-sensitive - business impact")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(x, recall, "g-o", label="recall")
    axes[1].plot(x, precision, "r-o", label="precision")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_title("Cost-sensitive - precision/recall tradeoff")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "task4_cost_sensitive.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    for fn in [task2, task3, task4]:
        try:
            fn()
            print(f"OK {fn.__name__}")
        except Exception as e:
            print(f"SKIP {fn.__name__}: {e}")


def task7():
    data = json.loads((RES / "task7_drift.json").read_text())
    drift_strengths = [d["drift_strength"] for d in data]
    recall = [d["recall"] for d in data]
    auc = [d["auc_roc"] for d in data]
    cost = [d["business_cost_usd"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(drift_strengths, recall, "g-o", label="recall")
    axes[0].plot(drift_strengths, auc, "b-o", label="auc-roc")
    axes[0].set_xlabel("drift strength"); axes[0].set_title("Task 7 - Time-based drift impact")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(drift_strengths, cost, width=0.15, color="firebrick")
    axes[1].set_xlabel("drift strength"); axes[1].set_ylabel("business cost (USD)")
    axes[1].set_title("Task 7 - Cost vs drift")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "task7_drift.png", dpi=120); plt.close(fig)


def task8():
    data = json.loads((RES / "task8_retraining.json").read_text())
    names = [d["strategy"] for d in data]
    mean_auc = [d["mean_auc"] for d in data]
    std_auc = [d["std_auc"] for d in data]
    n_retrains = [d["n_retrains"] for d in data]
    cost = [d["cost_credits"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(names, mean_auc, yerr=std_auc, capsize=8, color="seagreen")
    axes[0].set_title("Task 8 - Mean AUC ± std")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, n_retrains, color="steelblue")
    axes[1].set_title("# of retrains over 4 windows"); axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(names, cost, color="firebrick")
    axes[2].set_title("Compute cost (credits)"); axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "task8_retraining.png", dpi=120); plt.close(fig)


if __name__ != "__main__":
    pass
