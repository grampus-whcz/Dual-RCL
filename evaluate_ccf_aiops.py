#!/usr/bin/env python3
"""
CCF AIOps 评估脚本 v2 — 从 nohup 实验日志提取预测结果，计算 A@k 和延迟。

预测结果在 nohup 日志中（run.py 的 stdout），格式：
  MEPFL top-5 root services: (1)svc1,(2)svc2,...
  TVDiag root services: ['svc1', 'svc2', ...]
"""
import argparse, datetime, json, os, re
from pathlib import Path
import numpy as np

PROJECT_DIR = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_TRAIN_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"
GT_TEST_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"


def load_gt(gt_file, is_json):
    """Returns ordered list of (time_HH:MM, cmdb_id, level, fault_type) preserving GT order."""
    events = []
    if is_json:
        with open(gt_file) as f:
            gt = json.load(f)
        for i in range(len(gt["timestamp"])):
            ts = datetime.datetime.fromtimestamp(int(gt["timestamp"][i]))
            events.append((ts.strftime("%H:%M"), gt["cmdb_id"][i], gt["level"][i], gt["failure_type"][i]))
    else:
        import pandas as pd
        df = pd.read_csv(gt_file)
        for _, row in df.iterrows():
            ts = datetime.datetime.fromtimestamp(int(row["timestamp"]))
            events.append((ts.strftime("%H:%M"), row["cmdb_id"], row["level"], row["failure_type"]))
    return events


def gt_candidates(cmdb_id, level):
    """Set of candidate strings that would match a correct prediction."""
    cands = set()
    if level == "pod":
        cands.add(cmdb_id)
        base = re.sub(r"-\d+$", "", cmdb_id)
        cands.add(base)
    elif level == "service":
        cands.add(cmdb_id)
        cands.add(re.sub(r"\d+$", "", cmdb_id).rstrip("-"))
    else:  # node
        cands.add(cmdb_id)
    return cands


def extract_from_log(nohup_log):
    """Extract per-case predictions and latency from a nohup experiment log.

    Returns dict: time_HH:MM (zero-padded HHMM) -> {preds, latency, ok}.
    """
    if not os.path.exists(nohup_log):
        return {}
    with open(nohup_log, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    results = {}
    # Split into cases by "task_parsed" or case boundaries
    # Pattern: task parsed time -> followed by MEPFL/TVDiag prediction
    # Find all case blocks
    case_re = re.compile(
        r'\[(?:EVAL\] )?\{[^}]*"type": "task_parsed"[^}]*"time": "(\d{2})-(\d{2})"[^}]*\}'
        r'.*?(?=\[(?:EVAL\] )?\{[^}]*"type": "task_parsed"|\Z)',
        re.DOTALL,
    )
    # Simpler: iterate task_parsed lines, find next prediction+latency
    lines = content.split("\n")
    cur_time = None
    cur_preds = None
    cur_latency = None
    for line in lines:
        m = re.search(r'"type": "task_parsed"[^}]*"time": "(\d{2})-(\d{2})"', line)
        if m:
            # Save previous
            if cur_time is not None:
                key = f"{cur_time[0]}{cur_time[1]}"
                results[key] = {"preds": cur_preds or [], "latency": cur_latency}
            cur_time = (m.group(1), m.group(2))
            cur_preds = None
            cur_latency = None
            continue
        m = re.search(r"MEPFL top-5 root services:\s*(.*)", line)
        if m and cur_time:
            raw = m.group(1)
            preds = re.findall(r"\(\d+\)([^,(]+)", raw)
            cur_preds = [p.strip() for p in preds if p.strip()]
            continue
        m = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
        if m and cur_time:
            try:
                preds = eval(m.group(1))
                cur_preds = [str(p) for p in preds[:5]]
            except Exception:
                pass
            continue
        # Hybrid mode: fused root services is the FINAL prediction (overrides MEPFL/TVDiag)
        m = re.search(r"Fused root services \(causal re-ranked\):\s*(\[.*?\])", line)
        if m and cur_time:
            try:
                preds = eval(m.group(1))
                cur_preds = [str(p) for p in preds[:5]]
            except Exception:
                pass
            continue
        m = re.search(r'"type": "e2e_latency"[^}]*"duration_s": ([\d.]+)', line)
        if m and cur_time:
            cur_latency = float(m.group(1))
            continue
    # Save last
    if cur_time is not None:
        key = f"{cur_time[0]}{cur_time[1]}"
        results[key] = {"preds": cur_preds or [], "latency": cur_latency}
    return results


def evaluate(nohup_log, gt_file, is_json):
    events = load_gt(gt_file, is_json)
    parsed = extract_from_log(nohup_log)

    n_total = len(events)
    n_parsed = 0
    n_pred = 0
    a1 = a3 = a5 = 0
    latencies = []
    details = []

    for time_hhmm, cmdb, level, fault in events:
        key = time_hhmm.replace(":", "")
        if key not in parsed:
            continue
        info = parsed[key]
        preds = info["preds"]
        latency = info["latency"]
        n_parsed += 1
        if preds:
            n_pred += 1
            cands = gt_candidates(cmdb, level)
            top1 = preds[:1]
            top3 = preds[:3]
            top5 = preds[:5]
            h1 = any(any(c in p for c in cands) for p in top1)
            h3 = any(any(c in p for c in cands) for p in top3)
            h5 = any(any(c in p for c in cands) for p in top5)
            if h1: a1 += 1
            if h3: a3 += 1
            if h5: a5 += 1
            details.append({"time": time_hhmm, "gt": f"{cmdb}({level})", "fault": fault,
                            "preds": preds, "h1": h1, "h3": h3, "h5": h5})
        if latency:
            latencies.append(latency)

    n = n_pred if n_pred > 0 else 1
    return {
        "total": n_total, "parsed": n_parsed, "with_preds": n_pred,
        "A@1": a1/n, "A@3": a3/n, "A@5": a5/n,
        "a1": a1, "a3": a3, "a5": a5,
        "mean_latency": float(np.mean(latencies)) if latencies else None,
        "median_latency": float(np.median(latencies)) if latencies else None,
        "details": details,
    }


def evaluate_multi(log_gt_pairs):
    """Aggregate evaluation across multiple (nohup_log, gt_file) pairs.

    Each pair contributes its events; A@k computed over the union.
    """
    n_pred = a1 = a3 = a5 = 0
    lats = []
    total = 0
    for nohup_log, gt_file in log_gt_pairs:
        if not os.path.exists(nohup_log) or not os.path.exists(gt_file):
            continue
        parsed = extract_from_log(nohup_log)
        gt_map = load_gt(gt_file, True)
        total += len(gt_map)
        for hhmm, cmdb, level, fault in gt_map:
            key = hhmm.replace(":", "")
            if key not in parsed:
                continue
            info = parsed[key]
            if info["preds"]:
                n_pred += 1
                preds = info["preds"]
                cands = gt_candidates(cmdb, level)
                if any(any(c in p for c in cands) for p in preds[:1]): a1 += 1
                if any(any(c in p for c in cands) for p in preds[:3]): a3 += 1
                if any(any(c in p for c in cands) for p in preds[:5]): a5 += 1
            if info["latency"]:
                lats.append(info["latency"])
    n = n_pred if n_pred else 1
    return {
        "total": total, "with_preds": n_pred,
        "A@1": a1/n, "A@3": a3/n, "A@5": a5/n,
        "mean_latency": float(np.mean(lats)) if lats else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    TEST_DATES = ["2022-05-01", "2022-05-03", "2022-05-05", "2022-05-07", "2022-05-09"]
    TEST_PREFIXES = ["0501t", "0503t", "0505t", "0507t", "0509t"]

    # Per-date evaluation (single log per method per date)
    per_date_configs = []
    for method_key, logfile_tmpl in [
        ("LocaleXpert", "experiments_ccf_localexpert_{p}.log"),
        ("DualChannel", "experiments_ccf_dualchannel_{p}.log"),
        ("TVDiag-v3", "experiments_ccf_tvdig_v3_{p}.log"),
        ("Hybrid", "experiments_ccf_hybrid_{p}.log"),
    ]:
        per_date_configs.append((method_key, logfile_tmpl))

    all_res = {}
    # Primary: aggregate the "testall" single-log-per-method runs (all 5 dates in one log)
    testall_configs = [
        ("LocaleXpert", "experiments_ccf_localexpert_testall.log"),
        ("DualChannel", "experiments_ccf_dualchannel_testall.log"),
        ("TVDiag-v3", "experiments_ccf_tvdig_v3_testall.log"),
        ("Hybrid", "experiments_ccf_hybrid_testall.log"),
    ]
    for method_key, logfile in testall_configs:
        pairs = []
        for date_str in TEST_DATES:
            gt = os.path.join(GT_TEST_DIR, f"groundtruth-{date_str}.json")
            pairs.append((str(PROJECT_DIR / logfile), gt))
        try:
            r = evaluate_multi(pairs)
            all_res[f"{method_key}_test-all5"] = r
            lat = f", lat={r['mean_latency']:.0f}s" if r['mean_latency'] else ""
            print(f"{method_key} (all 5 dates): preds={r['with_preds']}/{r['total']}, "
                  f"A@1={r['A@1']:.3f} A@3={r['A@3']:.3f} A@5={r['A@5']:.3f}{lat}")
        except Exception as e:
            print(f"  [ERROR] {method_key}: {e}")
    return  # primary path complete

    # Also keep single-date (0501t) and cloudbed-3 validation for reference
    single_configs = [
        ("LocaleXpert", "0501t", "cloudbed", "2022-05-01", True, "experiments_ccf_localexpert_0501t.log"),
        ("DualChannel", "0501t", "cloudbed", "2022-05-01", True, "experiments_ccf_dualchannel_0501t.log"),
        ("TVDiag-v3", "0501t", "cloudbed", "2022-05-01", True, "experiments_ccf_tvdig_v3_0501t.log"),
        ("Hybrid", "0501t", "cloudbed", "2022-05-01", True, "experiments_ccf_hybrid_0501t.log"),
    ]
    for method, prefix, cloudbed, date_str, is_json, logfile in single_configs:
        nohup = PROJECT_DIR / logfile
        if not nohup.exists():
            continue
        gt = os.path.join(GT_TEST_DIR, f"groundtruth-{date_str}.json")
        try:
            r = evaluate(str(nohup), gt, is_json)
            all_res[f"{method}_{prefix}"] = r
        except Exception:
            pass

    # Write report
    lines = ["# CCF AIOps 评估结果 (v2)\n", f"> {datetime.datetime.now()}\n"]
    lines.append("## 对比表\n")
    lines.append("| 方法 | 数据集 | 解析数 | 有预测 | A@1 | A@3 | A@5 | 延迟均值(s) |")
    lines.append("|------|--------|--------|--------|-----|-----|-----|-------------|")
    for name, r in all_res.items():
        lat = f"{r['mean_latency']:.0f}" if r['mean_latency'] else "-"
        lines.append(f"| {name} | - | {r['parsed']}/{r['total']} | {r['with_preds']} | "
                     f"{r['A@1']:.4f} | {r['A@3']:.4f} | {r['A@5']:.4f} | {lat} |")
    out = PROJECT_DIR / "CCF_AIOps_Evaluation_Results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {out}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
