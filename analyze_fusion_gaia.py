#!/usr/bin/env python3
"""Honest decomposition for GAIA: does OUR fusion help, or is TVDiag doing the work?
Same apples-to-apples comparison on incidents where fusion ran."""
import subprocess, re, os, pickle
from pathlib import Path
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_PKL = PROJ / "Datasets/GAIA/fault_injection_tracerank/fault_injection_list_2021-07-01.pkl"


def load_gaia_gt():
    with open(GT_PKL, "rb") as f:
        data = pickle.load(f)
    gt = {}
    for fi in data:
        if isinstance(fi, dict) and "time" in fi and "service" in fi:
            t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
            gt[t] = fi["service"]
    return gt


def grep_extract(logfile, patterns):
    cmd = ["grep", "-aE", "|".join(patterns), logfile]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    except subprocess.TimeoutExpired:
        return ""


def parse_hybrid(logfile):
    out = grep_extract(logfile, [r'task_parsed', r'TVDiag root services', r'Fused root services', r'Causal walk failed'])
    results = {}
    cur = None; cur_tvdig = None; cur_fused = None; was_fb = False
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur: results[cur] = (cur_tvdig, cur_fused, was_fb)
            cur = f"{m.group(1)}{m.group(2)}"; cur_tvdig = cur_fused = None; was_fb = False
            continue
        if "Causal walk failed" in line or "TVDiag-only fallback" in line:
            was_fb = True
        m = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
        if m and cur:
            try: cur_tvdig = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
        m = re.search(r"Fused root services.*?:\s*(\[.*?\])", line)
        if m and cur:
            try: cur_fused = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
    if cur: results[cur] = (cur_tvdig, cur_fused, was_fb)
    return results


def main():
    gt = load_gaia_gt()
    hybrid = parse_hybrid(str(PROJ / "experiments_gaia_hybrid_0701.log"))

    fusion_inc = []   # (tvdig_preds, fused_preds, svc)
    fallback_inc = []
    for hhmm, (tvdig, fused, was_fb) in hybrid.items():
        if hhmm not in gt: continue
        svc = gt[hhmm]
        if was_fb or fused is None:
            fallback_inc.append((tvdig, svc))
        else:
            fusion_inc.append((tvdig, fused, svc))

    def acc(pl, k):
        return sum(1 for p, s in pl if p and any(s in pp for pp in p[:k])) / max(len(pl), 1)

    print(f"=== GAIA 分解 ===")
    print(f"  fallback（崩溃→纯TVDiag）: {len(fallback_inc)} 事件")
    print(f"  fusion （双通道成功→融合） : {len(fusion_inc)} 事件")
    print()
    print(f"=== 仅看融合运行的 {len(fusion_inc)} 事件（同事件对比）===")
    tv = [(t, s) for t, f, s in fusion_inc]
    fu = [(f, s) for t, f, s in fusion_inc]
    print(f"{'':32} {'A@1':>7} {'A@3':>7} {'A@5':>7}")
    print(f"{'TVDiag-standalone (同事件)':32} {acc(tv,1):>6.1%} {acc(tv,3):>6.1%} {acc(tv,5):>6.1%}")
    print(f"{'Hybrid-fused (我们的融合)':32} {acc(fu,1):>6.1%} {acc(fu,3):>6.1%} {acc(fu,5):>6.1%}")
    print()
    print(f"=== 融合净效应（Hybrid - TVDiag）===")
    print(f"  A@1: {acc(fu,1)-acc(tv,1):+.1%}")
    print(f"  A@3: {acc(fu,3)-acc(tv,3):+.1%}")
    print(f"  A@5: {acc(fu,5)-acc(tv,5):+.1%}")


if __name__ == "__main__":
    main()
