"""A4 Task 4: Cost-sensitive learning comparison.

Compare standard training vs cost-sensitive training (FN penalty).
Business impact: fraud_loss = fn * $1000, false_alarm = fp * $50.
"""
import json
from pathlib import Path

from run_local import run_full

OUT = Path(__file__).resolve().parent.parent / "results" / "task4_cost_sensitive.json"

def main():
    results = []
    configs = [
        {"name": "standard",          "cost_sensitive": False, "fn_penalty": 1.0},
        {"name": "cost_sens_x3",      "cost_sensitive": True,  "fn_penalty": 3.0},
        {"name": "cost_sens_x5",      "cost_sensitive": True,  "fn_penalty": 5.0},
        {"name": "cost_sens_x10",     "cost_sensitive": True,  "fn_penalty": 10.0},
    ]
    for cfg in configs:
        r = run_full(model_type="xgboost", imbalance_strategy="none",
                     cost_sensitive=cfg["cost_sensitive"], fn_penalty=cfg["fn_penalty"])
        r["config_name"] = cfg["name"]
        results.append(r)
        print(f"{cfg['name']:18s} TP={r['tp']:4d} FP={r['fp']:4d} FN={r['fn']:4d} "
              f"recall={r['recall']:.4f} prec={r['precision']:.4f} "
              f"fraud_loss=${r['fraud_loss_usd']:.0f} fa_cost=${r['false_alarm_cost_usd']:.0f} "
              f"total=${r['business_cost_usd']:.0f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
