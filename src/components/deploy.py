"""Component 7: Conditional Deployment.

Per A4 Task 1: deploy only if AUC-ROC >= threshold.
"Deploy" here means writing the bundled model to a /models PVC location +
emitting a model-version artifact. In a real cluster this would also kick
off a KServe/Seldon InferenceService update.
"""
from kfp import dsl
from kfp.dsl import Artifact, Input, Metrics, Model, Output


@dsl.component(base_image="mlops-a4-base:v1")
def deploy_model(
    model_in: Input[Model],
    metrics: Input[Metrics],
    deployed_marker: Output[Artifact],
    auc_threshold: float = 0.85,
) -> str:
    import json
    import shutil
    from pathlib import Path

    # Read AUC from the metrics artifact (KFP writes a JSON file at metrics.path)
    try:
        with open(metrics.path, "r") as f:
            m = json.load(f)
    except Exception:
        m = {}
    auc = float(m.get("auc_roc", 0.0))

    decision = "deploy" if auc >= auc_threshold else "reject"

    out_dir = Path(deployed_marker.path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if decision == "deploy":
        # Copy the model file beside the marker (acts as "deployed" artifact)
        target = out_dir / "deployed_model.pkl"
        shutil.copyfile(model_in.path, target)
        # Mock a registry-style version tag
        with open(deployed_marker.path, "w") as f:
            json.dump({
                "decision": "deploy",
                "auc_roc": auc,
                "threshold": auc_threshold,
                "model_path": str(target),
            }, f, indent=2)
    else:
        with open(deployed_marker.path, "w") as f:
            json.dump({
                "decision": "reject",
                "auc_roc": auc,
                "threshold": auc_threshold,
                "reason": f"AUC {auc:.4f} below threshold {auc_threshold}",
            }, f, indent=2)

    print(f"[DEPLOY] {decision} (auc={auc:.4f}, threshold={auc_threshold})")
    return decision
