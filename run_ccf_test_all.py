#!/usr/bin/env python3
"""Run a method on ALL 5 CCF AIOps test dates (2022-05-01/03/05/07/09).

Usage:
  python run_ccf_test_all.py --method localexpert
  python run_ccf_test_all.py --method hybrid
"""
import argparse, datetime, json, os, subprocess

PYTHON = "/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python"
GT_TEST_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
TEST_DATES = ["2022-05-01", "2022-05-03", "2022-05-05", "2022-05-07", "2022-05-09"]
TEST_PREFIXES = ["0501t", "0503t", "0505t", "0507t", "0509t"]

METHOD_ARGS = {
    "localexpert": ["--skip-multivariate"],
    "dualchannel": ["--anomaly-method", "tranad", "--anomaly-epochs", "3"],
    "tvdig": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
              "--rca-method", "tvdig", "--tvdig-model", "./tvdig_checkpoint_ccf_v3"],
    "hybrid": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
               "--rca-method", "hybrid", "--tvdig-model", "./tvdig_checkpoint_ccf_v3"],
}


def load_events_json(date_str):
    with open(os.path.join(GT_TEST_DIR, f"groundtruth-{date_str}.json")) as f:
        gt = json.load(f)
    events = []
    for i in range(len(gt["timestamp"])):
        ts = datetime.datetime.fromtimestamp(int(gt["timestamp"][i]))
        events.append((ts.strftime("%H:%M"), gt["cmdb_id"][i], gt["level"][i], gt["failure_type"][i]))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["localexpert", "dualchannel", "tvdig", "hybrid"])
    ap.add_argument("--model", default="ollama-qwen3-14b")
    args = ap.parse_args()
    extra = METHOD_ARGS[args.method]
    total_ok = total_fail = 0

    for date_str, prefix in zip(TEST_DATES, TEST_PREFIXES):
        report_dir = f"reports/Report_ccf_aiops/{args.method}_{prefix}"
        os.makedirs(report_dir, exist_ok=True)
        events = load_events_json(date_str)
        date_parts = date_str.split("-")
        task_date = f"{date_parts[0]}/{date_parts[1]}/{date_parts[2]}"
        ok = fail = 0
        for i, (time_str, cmdb, level, fault) in enumerate(events):
            case_name = f"{args.method}_{prefix}_{time_str.replace(':','')}"
            task = (f"At {task_date} {time_str} have exceptions in the microservices system. "
                    f"What are these exceptions? Please output an exception analysis.")
            cmd = [PYTHON, "-u", "run.py", "--task", task, "--name", case_name,
                   "--model", args.model, "--ollama-url", "http://localhost:11434/v1",
                   "--dataset", "ccf_aiops", "--date-prefix", prefix, "--report-dir", report_dir] + extra
            print(f"  [{date_str} {i+1}/{len(events)}] {time_str} | {cmdb} | {fault}")
            try:
                r = subprocess.run(cmd, capture_output=False, timeout=1800)
                if r.returncode == 0: ok += 1
                else: fail += 1
            except subprocess.TimeoutExpired: fail += 1
            except Exception: fail += 1
        total_ok += ok; total_fail += fail
        print(f">>> {date_str} ({prefix}): {ok} ok, {fail} fail")
    print(f"\n=== {args.method} ALL TEST DATES: {total_ok} ok, {total_fail} fail ===")


if __name__ == "__main__":
    main()
