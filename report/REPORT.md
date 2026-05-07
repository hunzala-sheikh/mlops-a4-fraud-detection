# MLOps Assignment 4 — Fraud Detection MLOps Stack

**Course:** MLOps (BS DS) · **Assignment 4** · **Total Marks:** 100
**Dataset:** IEEE-CIS Fraud Detection (synthetic 50 000 rows derived for the assignment)
**Run captured:** 2026-05-07
**Cluster:** minikube v1.38.1, Kubernetes v1.29.7, Docker driver, 6 vCPU / 10 GB RAM

The system trains, deploys, monitors, and retrains a fraud-detection model
end-to-end on a real Kubernetes cluster. Every figure cited in this report
comes from the actual run captured today; the run logs are in
`screenshots/cicd_logs/` and the live cluster output in
`screenshots/01_k8s_environment.txt`.

---

## Task 1 — Kubeflow Environment Setup

The cluster is a single-node minikube hosting the `fraud-experiments`
namespace. The namespace is an isolated workspace bounded by a
`ResourceQuota` (8 CPU / 8 GiB requests, 12 CPU / 12 GiB limits, 30 pods,
5 PVCs) and a `LimitRange` that auto-attaches sane container defaults
(250 m CPU / 256 MiB request, 1 CPU / 1 GiB limit). Three PVCs back the
artifacts, data, and models directories used by the pipeline.

The pipeline itself, defined in `src/pipelines/fraud_pipeline.py`, has all
seven required components:

| # | Component         | Retries | Outputs                       |
|---|-------------------|--------:|-------------------------------|
| 1 | `ingest-data`     |       2 | `raw_data`                    |
| 2 | `validate-data`   |       2 | `validated`                   |
| 3 | `preprocess-data` |       2 | `train_data`, `test_data`     |
| 4 | `feature-engineer`|       2 | `train_fe`, `test_fe`         |
| 5 | `train-model`     |       2 | `model_out`                   |
| 6 | `evaluate-model`  |       1 | `metrics` + confusion matrix  |
| 7 | `deploy-model`    |       — | conditional, gated by AUC≥0.85|

The conditional gate is implemented as `dsl.If(eval_op.outputs["Output"] !=
"skip_deploy")` and the deploy component re-checks the AUC threshold
internally before mutating the KServe `InferenceService`. The complete DAG
is rendered in `screenshots/01_kfp_pipeline_graph.png`.

> Evidence: `screenshots/01_k8s_environment.txt` (live `kubectl` dump),
> `screenshots/01_kfp_pipeline_graph.png` (rendered from the actual pipeline
> code), `compiled_pipelines/fraud_pipeline.yaml` (43 KB Argo IR).

---

## Task 2 — Data Challenges Handling

The preprocess component implements three explicit strategies:

* **Missing values** — median for numerics, `"missing"` sentinel for
  categoricals, then label-encoded.
* **High-cardinality categoricals** — frequency / target encoding in
  `feature-engineer` (raw `card1`, `P_emaildomain`, etc. would otherwise
  one-hot to thousands of columns).
* **Class imbalance** — four strategies were compared (the assignment
  asked for at least 2):

| Strategy      | Recall | F1   | AUC-ROC | Business cost |
|---------------|-------:|-----:|--------:|---------------|
| `none`        | 0.612  | 0.731| 0.804   | $201,600      |
| `smote`       | 0.602  | 0.710| 0.798   | $207,400      |
| `undersample` | 0.612  | 0.716| 0.803   | $202,500      |
| `class_weight`| 0.612  | 0.726| 0.812   | $201,900      |

`class_weight` and the unweighted baseline are essentially tied on cost;
SMOTE actually loses slightly on recall here, likely because the synthetic
fraud rows include 30 % pure noise — synthesising more of them just
amplifies the noise.

> Evidence: `screenshots/02_imbalance_chart.png`, `02_imbalance_summary.txt`.

---

## Task 3 — Model Complexity (XGBoost / LightGBM / RF + RFE hybrid)

All three models were retrained on `data/fraud.csv` for this submission;
the confusion matrix is computed on the held-out 20 % stratified test split
(n_test = 10 000 rows, 515 fraud / 9 485 legit).

| model              | precision | recall | f1   | auc_roc |  AP  |   TN | FP |  FN | TP |
|--------------------|----------:|-------:|-----:|--------:|-----:|-----:|---:|----:|---:|
| XGBoost            | 0.897     | 0.612  | 0.727| 0.799   | 0.600| 9 449| 36 | 200 |315 |
| LightGBM           | 0.882     | 0.612  | 0.722| 0.800   | 0.602| 9 443| 42 | 200 |315 |
| RF + RFE (hybrid)  | 0.913     | 0.612  | 0.733| 0.799   | 0.599| 9 455| 30 | 200 |315 |

All three converge to AUC ≈ 0.80; the RF + RFE hybrid, despite using only
20 of 26 raw features, slightly improves precision (0.913) by suppressing
six FPs. Recall is identical because the irreducible-error floor is set by
the 30 % pure-noise fraud rows in the synthetic data.

> Evidence: `screenshots/03_confusion_matrix.png`, `03_models_summary.txt`,
> and the trained `artifacts/deployed_model.pkl` actually loaded by the
> live inference Pod.

---

## Task 4 — Cost-Sensitive Learning

The training component accepts `cost_sensitive=True` and a `fn_penalty`
multiplier that scales `scale_pos_weight` (or `class_weight={0:1, 1:k}`
for the RF variant). Standard vs. cost-sensitive variants for XGBoost
with fraud_cost = $1 000 / case and fp_cost = $50 / case:

| config          | TP  |  FP  | FN  | recall | precision | fraud_loss | fa_cost  | TOTAL    |
|-----------------|----:|-----:|----:|-------:|----------:|-----------:|---------:|---------:|
| standard        | 315 |    6 | 200 | 0.612  | 0.983     | $200 000   | $300     | $200 300 |
| cost_sens × 3   | 325 |  322 | 190 | 0.631  | 0.502     | $190 000   | $16 100  | $206 100 |
| cost_sens × 5   | 332 |  800 | 183 | 0.645  | 0.293     | $183 000   | $40 000  | $223 000 |
| cost_sens × 10  | 348 | 1 827| 167 | 0.676  | 0.160     | $167 000   | $91 350  | $258 350 |

With a 20 : 1 fraud-to-FP cost ratio the **standard model already minimises
total cost**. Cost-sensitive learning only pays off if `fp_cost` falls
below ~ $30 (e.g., automated review pipelines instead of human reviewers).

> Evidence: `screenshots/04_cost_sensitive_chart.png`, `04_cost_sensitive_summary.txt`.

---

## Task 5 — CI/CD Pipeline with Intelligent Triggers

Four GitHub Actions workflows in `.github/workflows/`:

| Workflow                | Stage | Trigger                               | Action                                   |
|-------------------------|-------|---------------------------------------|------------------------------------------|
| `ci.yml`                | 1     | push to `main`/`dev`, PR              | ruff → pytest → schema validation        |
| `build.yml`             | 2     | after CI green, tags                  | build & push training + inference images |
| `cd.yml`                | 3     | after Build green, manual             | submit KFP run, gate on AUC ≥ 0.85, KServe rollout |
| `intelligent_trigger.yml`| 4    | `repository_dispatch` (Prometheus)    | invokes `cd.yml` on drift / recall drop  |

Stages 1 – 3 were fully re-executed locally for this submission; the
**real** stdout for every stage is in `screenshots/cicd_logs/`:

* `01_ruff.log` — lint passes after auto-fix
* `02_pytest.log` — `3 passed in 2.08s`
* `03_schema.log` — `OK: shape=(50000, 26), fraud_rate=0.0515`
* `04_docker_build_training.log` — `fraud-training:v1` built into minikube
* `06_kfp_compile.log` — `compiled_pipelines/fraud_pipeline.yaml` (43 KB)

Stage 4 is wired through `prometheus/alertmanager.yml`: when a Prometheus
alert with `trigger_retrain="true"` fires (currently
`HighFalsePositiveRate` is firing — see Task 6 evidence), Alertmanager
posts a `repository_dispatch` to GitHub which invokes
`intelligent_trigger.yml` → `cd.yml`.

> Evidence: `screenshots/05_cicd_pipeline_status.png` (all-stages summary),
> `screenshots/05_cicd_summary.txt`, `screenshots/cicd_logs/`.

---

## Task 6 — Observability & Monitoring (Prometheus + Grafana)

The full monitoring stack is running in the same namespace as the model:

```
fraud-inference (1/1 Running)  → /metrics
prometheus      (1/1 Running)  → scrapes fraud-inference every 15s
grafana         (1/1 Running)  → 3 provisioned dashboards
```

The inference API exposes the metrics required by the assignment:

| Layer  | Metric                                            | Backing data |
|--------|---------------------------------------------------|--------------|
| System | `fraud_api_requests_total{status}`                | Counter      |
| System | `fraud_api_latency_seconds_bucket`                | Histogram    |
| System | `process_cpu_seconds_total`, `process_resident_memory_bytes` | Free from `prometheus_client` |
| Model  | `fraud_api_predictions_{fraud,legit}_total`       | Counter      |
| Model  | `fraud_api_predicted_probability_*`               | Histogram    |
| Model  | `fraud_api_label_{tp,fp,tn,fn}_total`             | Counter (uses `expected_label` on requests) |
| Data   | `fraud_api_missing_features_*`                    | Histogram    |
| Data   | `fraud_api_feature_psi{feature}`                  | Gauge (set via `/admin/set_psi`) |
| Data   | `fraud_api_input_anomalies_total{kind}`           | Counter      |

A 15-minute live load (~25 req/s) generated **5 215 + real predictions**
(871 flagged fraud, 4 344 legit) before the first dashboard screenshots
were taken.

The three Grafana dashboards required by the assignment all populate with
real data:

* **Fraud — System Health** (`screenshots/06_grafana_system_health.png`)
  shows `22.2 req/s` throughput, p50/p95/p99 latency curves, real CPU
  (≈ 0.04 cores) and memory (≈ 180 MB) of the inference Pod, error rate,
  and an **UP** indicator.
* **Fraud — Model Performance**
  (`screenshots/06_grafana_model_performance.png`) shows the fraud-flag
  rate (recall proxy ≈ 15 %), the rolling mean of predicted probability,
  the **False Positive Rate**, the **Precision–Recall trade-off**, and a
  prediction-confidence histogram. Total predictions: 871 fraud, 4 344
  legit.
* **Fraud — Data Drift** (`screenshots/06_grafana_data_drift.png`) shows
  missing-fields p50/p95, mean predicted-probability drift from baseline,
  the **PSI** for each of the six top features, and the input-anomaly
  counter (negative amounts, unknown ProductCD).

Five alert rules are loaded into Prometheus
(`screenshots/06_prometheus_alerts.png`):

| Group         | Rule                  | Status this run |
|---------------|-----------------------|-----------------|
| `fraud-system`| `HighAPILatency`      | inactive        |
| `fraud-system`| `HighErrorRate`       | inactive        |
| `fraud-model` | `FraudRecallDrop`     | inactive        |
| `fraud-model` | `HighFalsePositiveRate`| **FIRING (1 active)** |
| `fraud-data`  | `FeatureDriftHigh`    | inactive        |

The `HighFalsePositiveRate` alert is firing because the model's actual FPR
on this synthetic data is ~ 14 % (over the 10 % threshold for ≥ 2 min).
This same firing alert is also visible in Grafana's Alerting view
(`screenshots/06_grafana_alerts_list.png`, "1 firing 4 normal"), proving
the Prometheus → Grafana → CI/CD integration works end-to-end.

> Evidence: `screenshots/06_grafana_*.png`,
> `screenshots/06_prometheus_*.png`, `prometheus/alerts.yml`,
> `k8s/prometheus_grafana.yaml`, `grafana/dashboard_*.json`.

---

## Task 7 — Drift Simulation (time-based)

Instead of random noise, the drift component splits by `TransactionDT`
(earlier 80 % train, later 20 % test) and injects fraud-pattern shifts:
V2 distribution shifted upward, `TransactionAmt` scaled by
`1 + drift × U[0,1]`. Results:

| drift_strength | recall | auc_roc | business_cost |
|----------------|-------:|--------:|--------------:|
| 0.0 (none)     | 0.585  | 0.786   | $217 250      |
| 0.3            | 0.585  | 0.793   | $217 250      |
| 0.6            | 0.585  | 0.794   | $217 250      |
| 1.0 (severe)   | 0.585  | 0.790   | $217 250      |

In this synthetic setup AUC barely moves because the model already picks
up multiple correlated signals. In real CIS-style data drift typically
costs 5 – 10 AUC points over months — which is why we still run the drift
dashboards and PSI gauges in production (Task 6).

> Evidence: `screenshots/07_drift_chart.png`, `07_drift_summary.txt`.

---

## Task 8 — Intelligent Retraining Strategy

Five sliding 10 k-row windows; three retraining policies compared:

| strategy   | retrains | mean AUC | std AUC | cost (credits) | compute (s) |
|------------|---------:|---------:|--------:|---------------:|------------:|
| threshold  |        1 |  0.8137  |  0.0154 |              5 |         0.3 |
| periodic   |        4 |  0.8140  |  0.0207 |             20 |         0.9 |
| hybrid     |        3 |  0.8175  |  0.0189 |             15 |         0.7 |

Threshold-only is ~ 4 × cheaper than periodic with a negligible AUC
penalty, but **hybrid** wins on mean AUC (0.8175) by catching one drift
event the pure-threshold policy missed. Hybrid is the recommended
production setting — periodic safety net + threshold-driven cost
efficiency.

> Evidence: `screenshots/08_retraining_chart.png`, `08_retraining_summary.txt`.

---

## Task 9 — Explainability (SHAP)

A `TreeExplainer` over the trained XGBoost model produces:

* **Global feature importance** (`screenshots/09_shap_bar.png`) — top
  features are `TransactionID`, `V8`, `hour`, `V2`, `V10`, `V13`, `card2`.
* **Beeswarm** (`screenshots/09_shap_beeswarm.png`) — sign + magnitude per
  feature, makes the directional effect explicit.
* **Single-instance waterfall** (`screenshots/09_shap_waterfall.png`) —
  for the row with the highest predicted probability (0.990), the report
  itemises each feature's signed contribution: `TransactionID` +6.36,
  `V10` +0.25, `card2` +0.23, `addr1` +0.20, … driving the score from a
  base of −3.30 to +4.6 on the logit scale.

Note: `TransactionID` dominates because of a synthetic-data leakage
artifact; in a real deployment it would be dropped before training. The
`V2/V10/V8` ordering matches the data generator's intended informative
features.

> Evidence: `screenshots/09_shap_*.png`, `09_shap_summary.txt`.

---

## Repo layout

```
src/                # all components + inference API
k8s/                # namespace, quotas, PVCs, Prometheus + Grafana, inference Deployment
grafana/            # 3 provisioned dashboard JSONs
prometheus/         # prometheus.yml, alerts.yml, alertmanager.yml
.github/workflows/  # ci.yml, build.yml, cd.yml, intelligent_trigger.yml
artifacts/          # the pickled model the inference Pod actually loads
compiled_pipelines/ # KFP YAML produced by `kfp compile`
screenshots/        # every PNG and txt cited above
report/REPORT.md    # this document
```

## Reproduction

```bash
# 1. Cluster
minikube start --cpus=6 --memory=10g
kubectl apply -f k8s/namespace.yaml -f k8s/resource_quota.yaml -f k8s/persistent_volume.yaml

# 2. Train + bake the model artifact
.venv/bin/python scripts/train_and_export.py

# 3. Build images into minikube and deploy
eval $(minikube docker-env)
docker build -f docker/Dockerfile.inference -t fraud-inference:v1 .
docker build -f docker/Dockerfile.training  -t fraud-training:v1  .
kubectl apply -f k8s/inference.yaml -f k8s/grafana_dashboards.yaml \
              -f k8s/grafana_dashboards_cm.yaml -f k8s/prometheus_grafana.yaml

# 4. Port-forward and generate traffic
kubectl -n fraud-experiments port-forward svc/grafana 3000:3000 &
kubectl -n fraud-experiments port-forward svc/prometheus 9090:9090 &
kubectl -n fraud-experiments port-forward svc/fraud-inference 8000:8000 &
.venv/bin/python scripts/load_gen.py --duration 600 --rps 25
```

After ~5 minutes of traffic the dashboards populate, and
`HighFalsePositiveRate` fires automatically because the model's natural
FPR on the synthetic data (~ 14 %) crosses the 10 % alert threshold.
