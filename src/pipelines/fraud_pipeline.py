"""End-to-end Kubeflow Pipeline for fraud detection.

7 components: ingest -> validate -> preprocess -> FE -> train -> eval -> conditional deploy.
- Each component has retries=2 for transient failures (per A4 Task 1).
- Deploy step gated by AUC-ROC threshold via dsl.If.
- Pipeline accepts parameters for model_type, imbalance_strategy, cost_sensitive,
  fn_penalty, auc_threshold, time_based_split (Task 7 drift).
"""
from kfp import dsl

from src.components.deploy import deploy_model
from src.components.evaluate import evaluate_model
from src.components.feature_engineer import feature_engineer
from src.components.ingest import ingest_data
from src.components.preprocess import preprocess_data
from src.components.train import train_model
from src.components.validate import validate_data


@dsl.pipeline(
    name="fraud-pipeline",
    description="A4 — Fraud detection: 7 components, conditional deploy, retries",
)
def fraud_pipeline(
    model_type: str = "xgboost",
    imbalance_strategy: str = "class_weight",
    cost_sensitive: bool = False,
    fn_penalty: float = 5.0,
    auc_threshold: float = 0.85,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 6,
    time_based_split: bool = False,
):
    ingest_op = ingest_data().set_retry(num_retries=2)
    validate_op = validate_data(raw_data=ingest_op.outputs["raw_data"]).set_retry(num_retries=2)
    preprocess_op = preprocess_data(
        validated=validate_op.outputs["validated"],
        test_size=test_size,
        imbalance_strategy=imbalance_strategy,
        random_state=random_state,
        time_based_split=time_based_split,
    ).set_retry(num_retries=2)

    fe_op = feature_engineer(
        train_data=preprocess_op.outputs["train_data"],
        test_data=preprocess_op.outputs["test_data"],
    ).set_retry(num_retries=2)

    train_op = train_model(
        train_fe=fe_op.outputs["train_fe"],
        model_type=model_type,
        cost_sensitive=cost_sensitive,
        fn_penalty=fn_penalty,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    ).set_retry(num_retries=2)

    eval_op = evaluate_model(
        test_fe=fe_op.outputs["test_fe"],
        model_in=train_op.outputs["model_out"],
    ).set_retry(num_retries=1)

    # Conditional deploy: only run if AUC-ROC was high enough.
    # The deploy component itself rechecks the threshold and writes a marker.
    with dsl.If(eval_op.outputs["Output"] != "skip_deploy"):
        deploy_model(
            model_in=train_op.outputs["model_out"],
            metrics=eval_op.outputs["metrics"],
            auc_threshold=auc_threshold,
        )


@dsl.pipeline(
    name="fraud-pipeline-drift",
    description="A4 Task 7 — Same pipeline but with time-based train/test split",
)
def fraud_pipeline_drift(
    model_type: str = "xgboost",
    imbalance_strategy: str = "class_weight",
    cost_sensitive: bool = False,
    auc_threshold: float = 0.85,
):
    fraud_pipeline(
        model_type=model_type,
        imbalance_strategy=imbalance_strategy,
        cost_sensitive=cost_sensitive,
        auc_threshold=auc_threshold,
        time_based_split=True,
    )
