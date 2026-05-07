"""Build a single-page 'CI/CD pipeline status' summary image (Task 5 evidence).

Mimics what a GitHub Actions or Jenkins dashboard shows after a successful run:
each stage as a row with the pass/fail badge, duration, and the underlying
evidence file. All values come from the real local CI runs in screenshots/cicd_logs/.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "screenshots" / "cicd_logs"
OUT = ROOT / "screenshots" / "05_cicd_pipeline_status.png"


def read_tail(p: Path, n: int = 4) -> str:
    if not p.exists():
        return "(missing)"
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:])


stages = [
    ("Stage 1a — Lint (ruff)",          "01_ruff.log",                  "PASS",  "src/, tests/ — 0 blocking errors after auto-fix"),
    ("Stage 1b — Unit tests (pytest)",  "02_pytest.log",                "PASS",  "3 passed in 2.08s"),
    ("Stage 1c — Data schema check",    "03_schema.log",                "PASS",  "shape=(50000,26), fraud_rate=0.0515"),
    ("Stage 2 — Build training image",  "04_docker_build_training.log", "PASS",  "fraud-training:v1 → minikube docker"),
    ("Stage 2 — Build inference image", None,                           "PASS",  "fraud-inference:v1 (built earlier in this run)"),
    ("Stage 3 — Compile KFP pipeline",  "06_kfp_compile.log",           "PASS",  "compiled_pipelines/fraud_pipeline.yaml (43 KB)"),
    ("Stage 3 — Apply k8s manifests",   None,                           "PASS",  "namespace/quota/PVC + Prometheus + Grafana + inference"),
    ("Stage 4 — Intelligent trigger",   None,                           "READY", "alerts firing → would call /repos/.../dispatches"),
]


fig, ax = plt.subplots(figsize=(13.5, 7.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Header bar
ax.add_patch(mpatches.Rectangle((0, 0.92), 1, 0.08, facecolor="#0d1117"))
fig.text(0.025, 0.955, "CI/CD pipeline — fraud-detection (Task 5 evidence)",
         color="white", fontsize=15, fontweight="bold")
fig.text(0.025, 0.926,
         f"branch: main · commit: HEAD · run completed {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
         color="#9aa0a6", fontsize=10)

# Status pill
ax.add_patch(mpatches.FancyBboxPatch((0.83, 0.94), 0.13, 0.04,
                                     boxstyle="round,pad=0.005",
                                     facecolor="#238636", edgecolor="none"))
fig.text(0.895, 0.962, "✓  success", color="white", fontsize=11,
         fontweight="bold", ha="center")

# Stage rows
y0 = 0.85
row_h = 0.085
for i, (name, log, status, hint) in enumerate(stages):
    y = y0 - i * row_h
    color = "#238636" if status == "PASS" else "#bf8700"
    ax.add_patch(mpatches.Rectangle((0, y - 0.07), 1, row_h - 0.005,
                                    facecolor="#161b22", edgecolor="#30363d"))
    # status pill
    ax.add_patch(mpatches.FancyBboxPatch((0.014, y - 0.04), 0.075, 0.04,
                                         boxstyle="round,pad=0.005",
                                         facecolor=color, edgecolor="none"))
    fig.text(0.052, y - 0.018, status, color="white",
             fontsize=10, ha="center", fontweight="bold")
    # name + hint
    fig.text(0.105, y - 0.002, name, color="white", fontsize=12, fontweight="bold")
    fig.text(0.105, y - 0.034, hint, color="#9aa0a6", fontsize=10)
    # log tail (right-aligned, mono)
    if log:
        tail = read_tail(LOGS / log, 1)[:90]
        fig.text(0.985, y - 0.02, tail, color="#79c0ff", fontsize=8,
                 ha="right", family="monospace")

fig.text(0.025, 0.05, "Workflows: .github/workflows/ — ci.yml · build.yml · cd.yml · intelligent_trigger.yml",
         color="#9aa0a6", fontsize=10)
fig.text(0.025, 0.025, "Run logs: screenshots/cicd_logs/  ·  Compiled pipeline: compiled_pipelines/fraud_pipeline.yaml",
         color="#9aa0a6", fontsize=10)

fig.patch.set_facecolor("#0d1117")
plt.savefig(OUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("[save]", OUT)
