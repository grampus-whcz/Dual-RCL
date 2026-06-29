#!/usr/bin/env python3
"""Fast evaluation for CCF AIOps testall — uses grep to extract only prediction lines
(huge logs are 600MB+, full read is too slow)."""
import subprocess, re, os, json
from pathlib import Path
import numpy as np

GT_TEST_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
TEST_DATES = ["2022-05-01", "2022-05-03", "2022-05-05", "2022-05-07", "2022-05-09"]


def load_gt(date_str):
    with open(os.path.join(GT_TEST_DIR, f"groundtruth-{date_str}.json")) as f:
        gt = json.load(f)
    return [(gt["timestamp"][i], gt["cmdb_id"][i], gt["level"][i], gt["failure_type"][i])
            for i in range(len(gt["timestamp"]))]


def gt_candidates(cmdb, level):
    c = {cmdb}
    if level == "pod":
        c.add(re.sub(r"-\d+$", "", cmdb))
    elif level == "service":
        c.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
    return c


def extract_preds_fast(logfile):
    """Use grep to extract prediction-relevant lines, then parse."""
    if not os.path.exists(logfile):
        return {}
    # grep only the lines we need (fast on huge files)
    patterns = [
        r'task_parsed',
        r'MEPFL top-5 root services',
        r'TVDiag root services',
        r'Fused root services',
    ]
    cmd = ["grep", "-aE", "|".join(patterns), logfile]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return {}
    # Parse: walk lines, keyed by time (HHMM)
    results = {}
    cur_time = None
    cur_preds = None
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_time:
                results[cur_time] = cur_preds or []
            cur_time = f"{m.group(1)}{m.group(2)}"
            cur_preds = None
            continue
        if not cur_time:
            continue
        # prediction lines (last one wins → Fused > TVDiag > MEPFL order in log)
        for pat in [r"MEPFL top-5 root services:\s*(.*)",
                    r"TVDiag root services:\s*(\[.*?\])",
                    r"Fused root services.*?:\s*(\[.*?\])"]:
            mm = re.search(pat, line)
            if mm:
                raw = mm.group(1)
                if raw.startswith("["):
                    try:
                        cur_preds = [str(p) for p in eval(raw)[:5]]
                    except Exception:
                        pass
                else:
                    cur_preds = [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)", raw) if p.strip()][:5]
                break
    if cur_time:
        results[cur_time] = cur_preds or []
    return results


def evaluate_method(logfile):
    """Aggregate A@k across all 5 test dates."""
    preds_by_time = extract_preds_fast(logfile)
    n_pred = a1 = a3 = a5 = 0
    total = 0
    for date_str in TEST_DATES:
        for ts, cmdb, level, fault in load_gt(date_str):
            from datetime import datetime
            hhmm = datetime.fromtimestamp(int(ts)).strftime("%H%M")
            total += 1
            preds = preds_by_time.get(hhmm)
            if preds:
                n_pred += 1
                cands = gt_candidates(cmdb, level)
                if any(any(c in p for c in cands) for p in preds[:1]): a1 += 1
                if any(any(c in p for c in cands) for p in preds[:3]): a3 += 1
                if any(any(c in p for c in cands) for p in preds[:5]): a5 += 1
    n = n_pred if n_pred else 1
    return total, n_pred, a1/n, a3/n, a5/n


def main():
    configs = [
        ("LocaleXpert", "experiments_ccf_localexpert_testall.log"),
        ("DualChannel", "experiments_ccf_dualchannel_testall.log"),
        ("TVDiag-v3", "experiments_ccf_tvdig_v3_testall.log"),
        ("Hybrid", "experiments_ccf_hybrid_testall.log"),
    ]
    print(f"{'Method':<14} {'preds':>10} {'A@1':>7} {'A@3':>7} {'A@5':>7}")
    print("-" * 48)
    for name, logfile in configs:
        path = str(Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA") / logfile)
        total, n_pred, a1, a3, a5 = evaluate_method(path)
        print(f"{name:<14} {n_pred}/{total:>4} {a1:>6.1%} {a3:>6.1%} {a5:>6.1%}")


if __name__ == "__main__":
    main()
