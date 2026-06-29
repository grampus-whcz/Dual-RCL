#!/usr/bin/env python3
"""Evaluate GAIA experiments (TVDiag-v3 / Hybrid-v2) — A@k from nohup logs.

GAIA GT: fault_injection_list_*.pkl → {time: service}
Predictions: parse nohup log for MEPFL/TVDiag/Fused root services.
"""
import argparse, os, re, pickle, datetime
from pathlib import Path
import numpy as np

PROJECT_DIR = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_PKL_DIR = PROJECT_DIR / "Datasets/GAIA/fault_injection_tracerank"


def load_gaia_gt(date_str):
    """Returns dict: HH:MM -> service_name."""
    pkl = GT_PKL_DIR / f"fault_injection_list_{date_str}.pkl"
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    gt = {}
    for fi in data:
        if isinstance(fi, dict) and "time" in fi and "service" in fi:
            t = fi["time"].strftime("%H:%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
            gt[t] = fi["service"]
    return gt


def extract_from_log(nohup_log):
    """Returns dict: HHMM -> {preds, latency}."""
    if not os.path.exists(nohup_log):
        return {}
    with open(nohup_log, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    results = {}
    lines = content.split("\n")
    cur_time = None
    cur_preds = None
    cur_latency = None
    for line in lines:
        m = re.search(r'"type": "task_parsed"[^}]*"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_time is not None:
                results[f"{cur_time[0]}{cur_time[1]}"] = {"preds": cur_preds or [], "latency": cur_latency}
            cur_time = (m.group(1), m.group(2))
            cur_preds = cur_latency = None
            continue
        m = re.search(r"MEPFL top-5 root services:\s*(.*)", line)
        if m and cur_time:
            cur_preds = [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)", m.group(1)) if p.strip()]
            continue
        m = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
        if m and cur_time:
            try: cur_preds = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
            continue
        m = re.search(r"Fused root services \(confidence-vote\):\s*(\[.*?\])", line)
        if m and cur_time:
            try: cur_preds = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
            continue
        m = re.search(r'"type": "e2e_latency"[^}]*"duration_s": ([\d.]+)', line)
        if m and cur_time:
            cur_latency = float(m.group(1))
    if cur_time is not None:
        results[f"{cur_time[0]}{cur_time[1]}"] = {"preds": cur_preds or [], "latency": cur_latency}
    return results


def evaluate(nohup_log, gt_map):
    parsed = extract_from_log(nohup_log)
    n_pred = a1 = a3 = a5 = 0
    lats = []
    for hhmm, svc in gt_map.items():
        key = hhmm.replace(":", "")
        if key not in parsed:
            continue
        info = parsed[key]
        if info["preds"]:
            n_pred += 1
            top = info["preds"]
            if any(svc in p for p in top[:1]): a1 += 1
            if any(svc in p for p in top[:3]): a3 += 1
            if any(svc in p for p in top[:5]): a5 += 1
        if info["latency"]:
            lats.append(info["latency"])
    n = n_pred if n_pred else 1
    return {
        "with_preds": n_pred, "A@1": a1/n, "A@3": a3/n, "A@5": a5/n,
        "mean_latency": float(np.mean(lats)) if lats else None,
    }


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    gt_map = load_gaia_gt("2021-07-01")

    configs = [
        ("LocaleXpert", "experiments_localexpert.log"),
        ("DualChannel", "experiments_dualchannel.log"),
        ("TVDiag-v3", "experiments_gaia_tvdig_0701.log"),
        ("Hybrid-v2", "experiments_gaia_hybrid_0701.log"),
    ]
    print(f"{'Method':<14} {'preds':>6} {'A@1':>7} {'A@3':>7} {'A@5':>7} {'lat':>6}")
    print("-" * 50)
    results = {}
    for name, logfile in configs:
        path = PROJECT_DIR / logfile
        if not path.exists():
            print(f"{name:<14}  (not found)")
            continue
        try:
            r = evaluate(str(path), gt_map)
            results[name] = r
            lat = f"{r['mean_latency']:.0f}" if r['mean_latency'] else "-"
            print(f"{name:<14} {r['with_preds']:>6} {r['A@1']:>6.1%} {r['A@3']:>6.1%} {r['A@5']:>6.1%} {lat:>5}s")
        except Exception as e:
            print(f"{name:<14}  ERROR: {e}")

    # Save
    lines = ["# GAIA 评估结果\n", f"> {datetime.datetime.now()}\n",
             "| 方法 | preds | A@1 | A@3 | A@5 | 延迟 |", "|------|-------|-----|-----|-----|------|"]
    for name, r in results.items():
        lat = f"{r['mean_latency']:.0f}s" if r['mean_latency'] else "-"
        lines.append(f"| {name} | {r['with_preds']} | {r['A@1']:.1%} | {r['A@3']:.1%} | {r['A@5']:.1%} | {lat} |")
    (PROJECT_DIR / "GAIA_Evaluation_Results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
