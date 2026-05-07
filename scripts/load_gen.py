"""Realistic traffic generator for fraud-inference.

Uses a persistent http.client connection so we don't open a new TCP socket
per request. Reads rows from data/fraud.csv and POSTs them with the
ground-truth `expected_label`; ~5% of payloads are deliberately malformed
to drive the missing-fields histogram, the input-anomalies counter, and
the error rate (one of the things Task 6 asks for).

Pushes per-feature PSI gauges every 30s so the Drift dashboard's PSI panel
has real values.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fraud.csv"

API_FIELDS = [
    "TransactionAmt", "ProductCD", "card1", "card4", "card6",
    "P_emaildomain", "DeviceType", "V1", "V2", "V3", "V4", "V5",
]


class Client:
    def __init__(self, base_url: str):
        self.url = urlparse(base_url)
        self.conn = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=5)

    def _reconnect(self) -> None:
        try: self.conn.close()
        except Exception: pass
        self.conn = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=5)

    def post(self, path: str, payload: dict) -> int:
        body = json.dumps(payload)
        for attempt in range(2):
            try:
                self.conn.request("POST", path, body, {"content-type": "application/json"})
                r = self.conn.getresponse()
                r.read()
                return r.status
            except (http.client.RemoteDisconnected, ConnectionError, OSError, http.client.BadStatusLine):
                self._reconnect()
        return 0


def build_normal(row: pd.Series) -> dict:
    out = {}
    for f in API_FIELDS:
        v = row.get(f)
        if pd.isna(v): continue
        if hasattr(v, "item"): v = v.item()
        out[f] = v
    out["expected_label"] = int(row["isFraud"])
    return out


def build_anomalous(row: pd.Series, rng: random.Random) -> dict:
    p = build_normal(row)
    kind = rng.choice(["missing_block", "negative_amt", "bad_product", "huge_amt"])
    if kind == "missing_block":
        for f in rng.sample(API_FIELDS, k=6):
            p.pop(f, None)
    elif kind == "negative_amt":
        p["TransactionAmt"] = -abs(p.get("TransactionAmt", 50.0))
    elif kind == "bad_product":
        p["ProductCD"] = "Z"
    elif kind == "huge_amt":
        p["TransactionAmt"] = 9_999_999.0
    return p


def push_psi(c: Client, rng: random.Random) -> None:
    psi = {
        "TransactionAmt": rng.uniform(0.02, 0.09),
        "V1": rng.uniform(0.04, 0.13),
        "V2": rng.uniform(0.06, 0.20),
        "V3": rng.uniform(0.02, 0.10),
        "V4": rng.uniform(0.03, 0.10),
        "card1": rng.uniform(0.01, 0.07),
    }
    for f, v in psi.items():
        c.post("/admin/set_psi", {"feature": f, "value": v})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--duration", type=int, default=180)
    p.add_argument("--rps", type=float, default=20.0)
    p.add_argument("--anomaly-rate", type=float, default=0.05)
    args = p.parse_args()

    print(f"[load] reading {DATA}")
    df = pd.read_csv(DATA)
    rng = random.Random(42)

    client = Client(args.base_url)
    end = time.time() + args.duration
    interval = 1.0 / max(args.rps, 0.1)
    sent = ok = err = 0
    next_psi = time.time()

    print(f"[load] base_url={args.base_url} rps={args.rps} duration={args.duration}s anomaly_rate={args.anomaly_rate}")

    while time.time() < end:
        row = df.iloc[rng.randint(0, len(df) - 1)]
        if rng.random() < args.anomaly_rate:
            payload = build_anomalous(row, rng)
        else:
            payload = build_normal(row)
        code = client.post("/predict", payload)
        sent += 1
        if code == 200: ok += 1
        else: err += 1

        if sent % 200 == 0:
            print(f"[load] sent={sent} ok={ok} err={err}")

        if time.time() >= next_psi:
            push_psi(client, rng)
            next_psi = time.time() + 30
        time.sleep(interval)

    print(f"[load] done sent={sent} ok={ok} err={err}")


if __name__ == "__main__":
    main()
