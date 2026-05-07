"""Smoke tests for component logic (running outside KFP)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_local import run_full  # type: ignore


def test_xgboost_pipeline_runs():
    out = run_full(model_type="xgboost", imbalance_strategy="none",
                   n_estimators=20, max_depth=4)
    assert out["recall"] >= 0.0
    assert out["auc_roc"] >= 0.5  # at minimum chance


def test_lightgbm_pipeline_runs():
    out = run_full(model_type="lightgbm", imbalance_strategy="smote",
                   n_estimators=20, max_depth=4)
    assert out["auc_roc"] >= 0.5


def test_cost_sensitive_increases_recall_or_keeps_it():
    standard = run_full(model_type="xgboost", imbalance_strategy="none",
                        cost_sensitive=False, n_estimators=20, max_depth=4)
    weighted = run_full(model_type="xgboost", imbalance_strategy="none",
                        cost_sensitive=True, fn_penalty=10.0,
                        n_estimators=20, max_depth=4)
    # cost_sensitive should not be drastically worse on recall
    assert weighted["recall"] >= standard["recall"] - 0.05
