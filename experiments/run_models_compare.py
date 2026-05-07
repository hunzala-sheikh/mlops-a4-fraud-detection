"""A4 Task 3: Model complexity comparison.

Train each of XGBoost, LightGBM, RF+RFE-hybrid on the same data and compare:
  - precision, recall, F1, AUC-ROC, average precision
  - confusion matrix
  - business cost
"""
import json
from pathlib import Path

from run_local import run_full

OUT = Path(__file__).resolve().parent.parent / "results" / "task3_models.json"

def main():
    results = []
    for mt in ["xgboost", "lightgbm", "rf_fs"]:
        r = run_full(model_type=mt, imbalance_strategy="none",
                     cost_sensitive=False)
        results.append(r)
        print(f"{mt:10s} prec={r['precision']:.4f} rec={r['recall']:.4f} "
              f"f1={r['f1']:.4f} auc={r['auc_roc']:.4f} ap={r['ap']:.4f} "
              f"cost=${r['business_cost_usd']:.0f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
