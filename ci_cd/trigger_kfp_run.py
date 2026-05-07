"""Submit a fraud-pipeline run to a remote KFP cluster.

Used by .github/workflows/cd.yml. Reads host + bearer token from env or args.
"""
import argparse
import os
import sys

import kfp


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("KFP_HOST"))
    p.add_argument("--token", default=os.environ.get("KFP_AUTH_TOKEN"))
    p.add_argument("--pipeline-name", default="fraud_pipeline")
    p.add_argument("--experiment-name", default="cd-runs")
    p.add_argument("--model", default="xgboost")
    p.add_argument("--auc-threshold", type=float, default=0.85)
    p.add_argument("--imbalance", default="class_weight")
    args = p.parse_args()

    if not args.host:
        print("ERROR: --host or $KFP_HOST required", file=sys.stderr)
        return 2

    client = kfp.Client(host=args.host,
                        existing_token=args.token if args.token else None)

    exp = client.create_experiment(name=args.experiment_name)
    pipelines = client.list_pipelines(page_size=100).pipelines or []
    target = next((p for p in pipelines if p.display_name == args.pipeline_name), None)
    if target is None:
        print(f"ERROR: pipeline '{args.pipeline_name}' not found on {args.host}", file=sys.stderr)
        return 1

    versions = client.list_pipeline_versions(target.pipeline_id).pipeline_versions or []
    version_id = versions[0].pipeline_version_id

    run = client.run_pipeline(
        experiment_id=exp.experiment_id,
        job_name=f"cd-{args.model}-auc{args.auc_threshold}",
        version_id=version_id,
        params={
            "model_type": args.model,
            "imbalance_strategy": args.imbalance,
            "auc_threshold": args.auc_threshold,
        },
    )
    print(f"submitted: run_id={run.run_id}")
    print(run.run_id)  # stdout for downstream wait_and_check_auc.py
    return 0


if __name__ == "__main__":
    sys.exit(main())
