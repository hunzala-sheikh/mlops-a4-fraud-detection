"""A4 Task 2: Compare class-imbalance handling strategies.

Strategies: none, smote, undersample, class_weight (via cost_sensitive=False
but with imbalance_strategy=class_weight which is implemented in _train via
scale_pos_weight set to base ratio without fn_penalty multiplier).

We use XGBoost as the workhorse model (fastest convergence on this dataset).
"""
import json
from pathlib import Path

from run_local import run_full  # type: ignore

OUT = Path(__file__).resolve().parent.parent / "results" / "task2_imbalance.json"

def main():
    results = []
    for strategy in ["none", "smote", "undersample", "class_weight"]:
        # class_weight via scale_pos_weight = base ratio (cost_sensitive=False, fn_penalty=1)
        if strategy == "class_weight":
            r = run_full(model_type="xgboost", imbalance_strategy="none",
                         cost_sensitive=True, fn_penalty=1.0)  # weight = base ratio only
            r["config"]["imbalance_strategy"] = "class_weight"
        else:
            r = run_full(model_type="xgboost", imbalance_strategy=strategy,
                         cost_sensitive=False)
        results.append(r)
        print(f"strategy={strategy} recall={r['recall']:.4f} f1={r['f1']:.4f} auc={r['auc_roc']:.4f} cost=${r['business_cost_usd']:.0f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
