"""Fraud-detection inference API.

Endpoints
---------
GET  /healthz          -> liveness
POST /predict          -> single transaction -> fraud probability
GET  /metrics          -> Prometheus exposition (system + model + data)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# ---------- Prometheus metrics (Task 6 - System & Model & Data) ----------
REQUESTS = Counter(
    "fraud_api_requests_total", "Total prediction requests", ["status"])
LATENCY = Histogram(
    "fraud_api_latency_seconds", "Prediction latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0))
PREDICTIONS_FRAUD = Counter(
    "fraud_api_predictions_fraud_total", "Predictions classified as fraud")
PREDICTIONS_LEGIT = Counter(
    "fraud_api_predictions_legit_total", "Predictions classified as legitimate")
PROBA = Histogram(
    "fraud_api_predicted_probability", "Distribution of predicted fraud probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0))
MISSING_PER_REQ = Histogram(
    "fraud_api_missing_features", "# of missing fields in incoming requests",
    buckets=(0, 1, 2, 3, 5, 10))
MODEL_INFO = Gauge(
    "fraud_api_model_info", "Model metadata", ["model_type", "version"])

# Confusion-matrix style counters (require expected_label on the request).
# Used by Grafana to compute FPR, precision, recall in real time.
LABEL_TP = Counter("fraud_api_label_tp_total", "True positives (correctly flagged fraud)")
LABEL_FP = Counter("fraud_api_label_fp_total", "False positives (legit flagged as fraud)")
LABEL_TN = Counter("fraud_api_label_tn_total", "True negatives (legit correctly let through)")
LABEL_FN = Counter("fraud_api_label_fn_total", "False negatives (fraud missed)")

# Feature drift (Population Stability Index) per feature, set by a background
# task or by a /set_psi admin call from the drift component.
FEATURE_PSI = Gauge(
    "fraud_api_feature_psi", "Population stability index per feature", ["feature"])

# Input anomaly counter - increments when the request payload violates expected
# ranges (e.g. negative TransactionAmt, unknown ProductCD).
INPUT_ANOMALIES = Counter(
    "fraud_api_input_anomalies_total", "Input data anomalies", ["kind"])


# ---------- App ----------
app = FastAPI(title="Fraud Detection API", version="1.0.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/deployed_model.pkl")
MODEL_BUNDLE: dict[str, Any] | None = None


def _load_model():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"model not found at {path}")
        # joblib is the standard way to load sklearn/XGBoost artifacts and is
        # only ever called against a path inside the container image.
        MODEL_BUNDLE = joblib.load(path)
        MODEL_INFO.labels(
            model_type=MODEL_BUNDLE.get("model_type", "unknown"),
            version=os.environ.get("MODEL_VERSION", "dev"),
        ).set(1)
    return MODEL_BUNDLE


class TransactionRequest(BaseModel):
    """All-optional fields - the model handles missing values upstream."""
    TransactionAmt: float | None = None
    ProductCD: str | None = None
    card1: int | None = None
    card4: str | None = None
    card6: str | None = None
    P_emaildomain: str | None = None
    DeviceType: str | None = None
    V1: float | None = None
    V2: float | None = None
    V3: float | None = None
    V4: float | None = None
    V5: float | None = None
    # Optional ground-truth label from the load-gen / shadow-eval traffic.
    # When provided, the API increments TP/FP/TN/FN counters so Grafana can
    # plot real precision/recall/FPR.
    expected_label: int | None = None


class PSIUpdate(BaseModel):
    feature: str
    value: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest, threshold: float = 0.5):
    start = time.time()
    try:
        bundle = _load_model()
        model = bundle["model"]
        # Pull out the optional expected_label before feeding to the model.
        row = req.model_dump()
        expected_label = row.pop("expected_label", None)
        missing = sum(1 for v in row.values() if v is None)
        MISSING_PER_REQ.observe(missing)
        # Cheap sanity checks => anomaly counter (Task 6 input anomalies panel).
        amt = row.get("TransactionAmt")
        if amt is not None and amt < 0:
            INPUT_ANOMALIES.labels(kind="negative_amount").inc()
        if row.get("ProductCD") is not None and row["ProductCD"] not in {"H", "W", "C", "S", "R"}:
            INPUT_ANOMALIES.labels(kind="unknown_product_cd").inc()
        # Fill missing numerics with 0.0 (model was trained with imputation)
        x = np.array([[v if isinstance(v, (int, float)) and v is not None else 0.0
                       for v in row.values()]])
        proba = float(model.predict_proba(x)[0, 1])
        is_fraud = bool(proba >= threshold)

        if is_fraud:
            PREDICTIONS_FRAUD.inc()
        else:
            PREDICTIONS_LEGIT.inc()
        PROBA.observe(proba)
        REQUESTS.labels(status="ok").inc()

        # Confusion-matrix counters when ground truth is available.
        if expected_label is not None:
            if is_fraud and expected_label == 1:
                LABEL_TP.inc()
            elif is_fraud and expected_label == 0:
                LABEL_FP.inc()
            elif not is_fraud and expected_label == 1:
                LABEL_FN.inc()
            else:
                LABEL_TN.inc()

        return PredictionResponse(
            fraud_probability=proba, is_fraud=is_fraud,
            threshold=threshold,
            model_version=os.environ.get("MODEL_VERSION", "dev"),
        )
    except FileNotFoundError as e:
        REQUESTS.labels(status="model_missing").inc()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        LATENCY.observe(time.time() - start)


@app.post("/admin/set_psi")
def set_psi(update: PSIUpdate):
    """Update the per-feature PSI gauge (called by the drift component)."""
    FEATURE_PSI.labels(feature=update.feature).set(update.value)
    return {"ok": True, "feature": update.feature, "value": update.value}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
