"""Compile all A4 pipelines to YAML.

Run from repo root:
    python -m src.compile_all
"""
from pathlib import Path

from kfp import compiler

from src.pipelines import fraud_pipeline, fraud_pipeline_drift

OUT = Path(__file__).resolve().parents[1] / "compiled_pipelines"
OUT.mkdir(exist_ok=True)

PIPELINES = [
    (fraud_pipeline, "fraud_pipeline.yaml"),
    (fraud_pipeline_drift, "fraud_pipeline_drift.yaml"),
]


def main() -> None:
    for pipe, fname in PIPELINES:
        dest = OUT / fname
        compiler.Compiler().compile(pipeline_func=pipe, package_path=str(dest))
        print(f"compiled -> {dest}")


if __name__ == "__main__":
    main()
