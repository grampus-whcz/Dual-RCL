#!/usr/bin/env python3
"""
Run CCF AIOps experiments on a single cloudbed with a specific method.
Generates per-event reports and evaluation logs.

Usage:
  python run_ccf_aiops_batch.py --method localexpert --cloudbed cloudbed-3 --prefix 0320c
  python run_ccf_aiops_batch.py --method dualchannel --cloudbed cloudbed-2 --prefix 0320b
  python run_ccf_aiops_batch.py --method tvdig --cloudbed cloudbed-3 --prefix 0320c
  python run_ccf_aiops_batch.py --method localexpert --cloudbed cloudbed --prefix 0501t --date 2022-05-01 --gt-json
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

PYTHON = "/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python"
GT_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"
GT_TEST_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"

METHOD_ARGS = {
    "localexpert": ["--skip-multivariate"],
    "dualchannel": ["--anomaly-method", "tranad", "--anomaly-epochs", "3"],
    "tvdig": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
              "--rca-method", "tvdig", "--tvdig-model", "./tvdig_checkpoint_ccf_v3"],
    "hybrid": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
               "--rca-method", "hybrid", "--tvdig-model", "./tvdig_checkpoint_ccf_v3"],
}


def load_events_csv(gt_file):
    """Load ground truth from CSV format."""
    import pandas as pd
    df = pd.read_csv(gt_file)
    events = []
    for _, row in df.iterrows():
        ts = datetime.datetime.fromtimestamp(row["timestamp"])
        events.append({
            "time": ts.strftime("%H:%M"),
            "level": row["level"],
            "cmdb_id": row["cmdb_id"],
            "failure_type": row["failure_type"],
        })
    return events


def load_events_json(gt_file):
    """Load ground truth from JSON format (test data)."""
    with open(gt_file, "r") as f:
        gt = json.load(f)
    events = []
    for i in range(len(gt["timestamp"])):
        ts = datetime.datetime.fromtimestamp(gt["timestamp"][i])
        events.append({
            "time": ts.strftime("%H:%M"),
            "level": gt["level"][i],
            "cmdb_id": gt["cmdb_id"][i],
            "failure_type": gt["failure_type"][i],
        })
    return events


def main():
    parser = argparse.ArgumentParser(description="Run CCF AIOps batch experiments")
    parser.add_argument("--method", required=True, choices=["localexpert", "dualchannel", "tvdig", "hybrid"])
    parser.add_argument("--cloudbed", required=True, help="e.g., cloudbed-3, cloudbed")
    parser.add_argument("--prefix", required=True, help="e.g., 0320c, 0501t")
    parser.add_argument("--date", default="2022-03-20", help="Date string for task")
    parser.add_argument("--gt-json", action="store_true", help="Use JSON ground truth format")
    parser.add_argument("--gt-file", default=None, help="Override GT file path")
    parser.add_argument("--model", default="ollama-qwen3-14b")
    parser.add_argument("--report-dir", default=None,
                        help="Report dir (auto-generated if not set)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limit number of events to process")
    args = parser.parse_args()

    if args.report_dir is None:
        args.report_dir = f"reports/Report_ccf_aiops/{args.method}_{args.prefix}"

    os.makedirs(args.report_dir, exist_ok=True)

    # Load ground truth
    if args.gt_file:
        gt_file = args.gt_file
    elif args.gt_json:
        date_str = args.date.replace("-", "")
        gt_file = os.path.join(GT_TEST_DIR, f"groundtruth-{args.date}.json")
    else:
        cb_num = args.cloudbed.split("-")[-1]
        gt_file = os.path.join(GT_DIR, f"groundtruth-k8s-{cb_num}-{args.date}.csv")

    print(f"Loading GT from: {gt_file}")
    if args.gt_json:
        events = load_events_json(gt_file)
    else:
        events = load_events_csv(gt_file)

    print(f"Found {len(events)} events")

    if args.max_events:
        events = events[:args.max_events]
        print(f"Limited to {len(events)} events")

    # Build task date string
    date_parts = args.date.split("-")
    task_date = f"{date_parts[0]}/{date_parts[1]}/{date_parts[2]}"

    extra_args = METHOD_ARGS[args.method]
    success = 0
    fail = 0

    for i, ev in enumerate(events):
        time_str = ev["time"]
        case_name = f"{args.method}_{args.prefix}_{time_str.replace(':', '')}"
        task = (f"At {task_date} {time_str} have exceptions in the microservices system. "
                f"What are these exceptions? Please output an exception analysis.")

        cmd = [
            PYTHON, "-u", "run.py",
            "--task", task,
            "--name", case_name,
            "--model", args.model,
            "--ollama-url", "http://localhost:11434/v1",
            "--dataset", "ccf_aiops",
            "--date-prefix", args.prefix,
            "--report-dir", args.report_dir,
        ] + extra_args

        print(f"\n[{i+1}/{len(events)}] {time_str} | {ev['cmdb_id']} | {ev['failure_type']}")
        print(f"  cmd: {' '.join(cmd[:8])}...")

        try:
            result = subprocess.run(cmd, capture_output=False, timeout=1800)
            if result.returncode == 0:
                success += 1
                print(f"  [OK] {case_name}")
            else:
                fail += 1
                print(f"  [FAIL] {case_name} (rc={result.returncode})")
        except subprocess.TimeoutExpired:
            fail += 1
            print(f"  [TIMEOUT] {case_name}")
        except Exception as e:
            fail += 1
            print(f"  [ERROR] {case_name}: {e}")

    print(f"\n{'='*60}")
    print(f"  Method: {args.method} | Prefix: {args.prefix}")
    print(f"  Total: {len(events)} | Success: {success} | Failed: {fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
