"""Render the actual fraud_pipeline.py DAG as a high-quality PNG using
Graphviz, including the conditional-deploy guard and per-step retry counts.

This reflects the real pipeline structure that fraud_pipeline.py compiles
into when submitted to KFP - same components, same edges, same retries.
"""
from pathlib import Path

import graphviz

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "01_kfp_pipeline_graph.png"

g = graphviz.Digraph(
    "fraud_pipeline",
    format="png",
    graph_attr={"rankdir": "TB", "bgcolor": "#0e1116", "fontname": "Helvetica",
                "label": "fraud-pipeline (A4 — Task 1)\\n7 components · per-step retries · conditional deploy",
                "labelloc": "t", "fontcolor": "white", "fontsize": "16", "pad": "0.5"},
    node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica",
               "color": "#3a3f48", "fontcolor": "white", "fontsize": "12", "margin": "0.18,0.10"},
    edge_attr={"color": "#888", "fontname": "Helvetica", "fontsize": "10", "fontcolor": "#bbb"},
)

C_COMP = "#1f6feb"   # blue: code component
C_DATA = "#d29922"   # amber: artifact
C_COND = "#a371f7"   # purple: conditional gate
C_DEP  = "#3fb950"   # green: deploy

# ---- nodes (mirroring fraud_pipeline.py) ----
g.node("ingest", "ingest-data\\n(retry=2)", fillcolor=C_COMP)
g.node("raw", "raw_data", fillcolor=C_DATA, shape="folder")

g.node("validate", "validate-data\\n(retry=2)", fillcolor=C_COMP)
g.node("validated", "validated", fillcolor=C_DATA, shape="folder")

g.node("preprocess",
       "preprocess-data\\n(missing values, encoding,\\nimbalance: SMOTE/undersample/class_weight)\\n(retry=2)",
       fillcolor=C_COMP)
g.node("train_data", "train_data", fillcolor=C_DATA, shape="folder")
g.node("test_data",  "test_data",  fillcolor=C_DATA, shape="folder")

g.node("fe", "feature-engineer\\n(target/freq encoding, scaling)\\n(retry=2)", fillcolor=C_COMP)
g.node("train_fe", "train_fe", fillcolor=C_DATA, shape="folder")
g.node("test_fe",  "test_fe",  fillcolor=C_DATA, shape="folder")

g.node("train",
       "train-model\\n(XGBoost / LightGBM / RF+RFE)\\nTask 4: cost-sensitive option\\n(retry=2)",
       fillcolor=C_COMP)
g.node("model_out", "model_out", fillcolor=C_DATA, shape="folder")

g.node("eval",
       "evaluate-model\\nP / R / F1 / AUC-ROC / confusion matrix\\n(retry=1)",
       fillcolor=C_COMP)
g.node("metrics", "metrics", fillcolor=C_DATA, shape="folder")

g.node("cond", "if AUC-ROC >= auc_threshold (default 0.85)", shape="diamond",
       style="filled", fillcolor=C_COND)
g.node("deploy", "deploy-model\\n(conditional)\\nrolls forward KServe InferenceService",
       fillcolor=C_DEP)

# ---- edges ----
g.edge("ingest", "raw")
g.edge("raw", "validate")
g.edge("validate", "validated")
g.edge("validated", "preprocess")
g.edge("preprocess", "train_data")
g.edge("preprocess", "test_data")
g.edge("train_data", "fe")
g.edge("test_data", "fe")
g.edge("fe", "train_fe")
g.edge("fe", "test_fe")
g.edge("train_fe", "train")
g.edge("train", "model_out")
g.edge("test_fe", "eval")
g.edge("model_out", "eval")
g.edge("eval", "metrics")
g.edge("metrics", "cond")
g.edge("cond", "deploy", label=" yes (AUC ok) ")
g.edge("model_out", "deploy", style="dashed")

OUT.parent.mkdir(parents=True, exist_ok=True)
out_no_ext = str(OUT.with_suffix(""))
g.render(filename=out_no_ext, cleanup=True)  # writes 01_kfp_pipeline_graph.png
print("[save]", OUT)
