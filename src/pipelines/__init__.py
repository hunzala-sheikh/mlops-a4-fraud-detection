"""KFP v2 pipeline definitions.

fraud_pipeline:           full 7-component pipeline with conditional deploy + retries
fraud_pipeline_drift:     time-based drift simulation variant (Task 7)
imbalance_compare:        runs fraud_pipeline with each imbalance strategy (Task 2)
"""
from .fraud_pipeline import fraud_pipeline, fraud_pipeline_drift
