"""Wait for the most recent KFP run to finish; fail if AUC < threshold.

Reads run_id from stdin (piped from trigger_kfp_run.py).
"""
import argparse
import os
import sys
import time

import kfp


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("KFP_HOST"))
    p.add_argument("--token", default=os.environ.get("KFP_AUTH_TOKEN"))
    p.add_argument("--auc-threshold", type=float, default=0.85)
    p.add_argument("--timeout-min", type=int, default=30)
    p.add_argument("--run-id", default=None,
                   help="run_id (default: read last line of stdin)")
    args = p.parse_args()

    run_id = args.run_id or sys.stdin.read().strip().splitlines()[-1]
    if not run_id:
        print("ERROR: no run_id", file=sys.stderr)
        return 2

    client = kfp.Client(host=args.host,
                        existing_token=args.token if args.token else None)

    deadline = time.time() + args.timeout_min * 60
    while time.time() < deadline:
        run = client.get_run(run_id=run_id).run
        state = (run.state or "").upper()
        print(f"run {run_id[:8]} state={state}")
        if state in ("SUCCEEDED", "FAILED", "ERROR", "CANCELED"):
            break
        time.sleep(15)

    if state != "SUCCEEDED":
        print(f"ERROR: run {state}", file=sys.stderr)
        return 1
    print("KFP run succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
