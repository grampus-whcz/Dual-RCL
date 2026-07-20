#!/usr/bin/env python3
"""
Experiments 1-3: Robust conflict-scenario isolation using two-pass grep.
"""
import os, re, json, subprocess, pickle
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_DIR = PROJ / "Datasets/GAIA/fault_injection_tracerank"
CCF_GT = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"


def grep_lines(logfile, pattern):
    """Single-pattern grep, returns matching lines."""
    try:
        r = subprocess.run(["grep", "-a", pattern, logfile],
                          capture_output=True, text=True, timeout=300)
        return r.stdout.strip().split("\n") if r.stdout.strip() else []
    except:
        return []


def parse_events(logfiles):
    """Two-pass parse: (1) task_parsed → event boundaries, (2) predictions + scenarios."""
    events = {}  # {hhmm: {preds, scenario, latency}}

    for lf in logfiles:
        if not os.path.exists(lf): continue

        # Pass 1: task_parsed lines → ordered list of event keys
        tp_lines = grep_lines(lf, '"type": "task_parsed"')
        event_keys = []
        for line in tp_lines:
            m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
            if m: event_keys.append(f"{m.group(1)}{m.group(2)}")

        # Pass 2: predictions (last MEPFL/TVDiag/Fused per event)
        pred_lines = grep_lines(lf, 'root services')
        preds_by_pos = {}
        for line in pred_lines:
            for pat in [r"Fused root services.*?:\s*(\[.*?\])",
                        r"TVDiag root services:\s*(\[.*?\])",
                        r"MEPFL top-5 root services:\s*(.*)"]:
                m = re.search(pat, line)
                if m:
                    raw = m.group(1)
                    try:
                        if raw.startswith("["):
                            preds = [str(p) for p in eval(raw)[:5]]
                        else:
                            preds = [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)", raw) if p.strip()][:5]
                        # Store with line position for later assignment
                        preds_by_pos[len(preds_by_pos)] = preds
                    except: pass
                    break

        # Pass 3: conflict scenarios
        cs_lines = grep_lines(lf, "Conflict scenario:")
        scenarios = []
        for line in cs_lines:
            m = re.search(r'Conflict scenario:\s*(\w+)', line)
            if m: scenarios.append(m.group(1))

        # Pass 4: latencies
        lat_lines = grep_lines(lf, "Phase 9.*completed in")
        lats = []
        for line in lat_lines:
            m = re.search(r'completed in\s+([\d.]+)s', line)
            if m: lats.append(float(m.group(1)))
        # Also try e2e_latency
        e2e_lines = grep_lines(lf, '"type": "e2e_latency"')
        for line in e2e_lines:
            m = re.search(r'"duration_s":\s*([\d.]+)', line)
            if m: lats.append(float(m.group(1)))

        # Assign by position (each event has exactly 1 prediction, 0-1 scenario, 0-1 latency)
        for i, key in enumerate(event_keys):
            events[key] = {
                'preds': preds_by_pos.get(i, None),
                'scenario': scenarios[i] if i < len(scenarios) else 'unknown',
                'latency': lats[i] if i < len(lats) else None,
            }

    return events


def load_gaia_gt():
    gt = {}
    for d in ["2021-07-04", "2021-07-05", "2021-07-06"]:
        with open(GT_DIR / f"fault_injection_list_{d}.pkl", "rb") as f:
            data = pickle.load(f)
        for fi in data:
            if isinstance(fi, dict) and "time" in fi and "service" in fi:
                t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
                gt[t] = fi["service"]
    return gt


def load_ccf_gt():
    gt = {}
    for d in ["2022-05-01","2022-05-03","2022-05-05","2022-05-07","2022-05-09"]:
        with open(os.path.join(CCF_GT, f"groundtruth-{d}.json")) as f:
            g = json.load(f)
        for i in range(len(g["timestamp"])):
            hhmm = datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            cmdb = g["cmdb_id"][i]; level = g["level"][i]
            cands = {cmdb}
            if level == "pod": cands.add(re.sub(r"-\d+$", "", cmdb))
            elif level == "service": cands.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
            gt[hhmm] = (cmdb, cands)
    return gt


def compute_ak(preds, gt_val, is_gaia=True):
    if not preds: return False, False, False
    if is_gaia:
        svc = gt_val
        return (any(svc in p for p in preds[:1]),
                any(svc in p for p in preds[:3]),
                any(svc in p for p in preds[:5]))
    else:
        _, cands = gt_val
        return (any(any(c in p for c in cands) for p in preds[:1]),
                any(any(c in p for c in cands) for p in preds[:3]),
                any(any(c in p for c in cands) for p in preds[:5]))


def main():
    os.chdir(PROJ)

    # ===== Experiment 1: GAIA conflict-scenario isolation =====
    print("=" * 70)
    print("  Experiment 1: GAIA Conflict-Scenario Isolation (LE vs DC)")
    print("=" * 70)

    gt = load_gaia_gt()
    dc = parse_events([f"logs/experiments_gaia_dualchannel_{d}.log" for d in "0704 0705 0706".split()])
    le = parse_events([f"logs/experiments_gaia_localexpert_{d}.log" for d in "0704 0705 0706".split()])

    scenarios = defaultdict(list)
    for hhmm in dc:
        if hhmm in gt:
            scenarios[dc[hhmm]['scenario']].append(hhmm)

    print(f"\n  Scenario distribution ({sum(len(v) for v in scenarios.values())} events):")
    for sc, evs in sorted(scenarios.items(), key=lambda x: -len(x[1])):
        print(f"    {sc}: {len(evs)}")

    print(f"\n  {'Scenario':<30} {'N':>4} | {'LE A@1':>7} {'LE A@3':>7} {'LE A@5':>7} | {'DC A@1':>7} {'DC A@3':>7} {'DC A@5':>7}")
    print("  " + "-" * 90)
    for sc in ['consistent', 'both_anom_different_root', 'multi_anom_single_normal']:
        evs = scenarios.get(sc, [])
        if not evs: continue
        le_h = [0,0,0]; dc_h = [0,0,0]; n_le = n_dc = 0
        for hhmm in evs:
            svc = gt[hhmm]
            lp = le.get(hhmm, {}).get('preds')
            if lp: n_le += 1; h = compute_ak(lp, svc); le_h = [a+b for a,b in zip(le_h, h)]
            dp = dc.get(hhmm, {}).get('preds')
            if dp: n_dc += 1; h = compute_ak(dp, svc); dc_h = [a+b for a,b in zip(dc_h, h)]
        ln = max(n_le,1); dn = max(n_dc,1)
        print(f"  {sc:<30} {len(evs):>4} | {le_h[0]/ln:>6.1%} {le_h[1]/ln:>6.1%} {le_h[2]/ln:>6.1%} | {dc_h[0]/dn:>6.1%} {dc_h[1]/dn:>6.1%} {dc_h[2]/dn:>6.1%}")

    # ===== Experiment 2: CCF conflict-scenario isolation =====
    print(f"\n{'=' * 70}")
    print("  Experiment 2: CCF AIOps Conflict-Scenario Isolation (LE vs TV)")
    print("=" * 70)

    ccf_gt = load_ccf_gt()
    ccf_dc = parse_events([f"logs/experiments_ccf_dualchannel_{p}.log" for p in "0501t 0503t 0505t 0507t 0509t".split()])
    ccf_tv = parse_events([f"logs/experiments_ccf_tvdig_{p}.log" for p in "0501t 0503t 0505t 0507t 0509t".split()])
    ccf_le = parse_events([f"logs/experiments_ccf_localexpert_{p}.log" for p in "0501t 0503t 0505t 0507t 0509t".split()])

    ccf_sc = defaultdict(list)
    for hhmm in ccf_dc:
        if hhmm in ccf_gt:
            ccf_sc[ccf_dc[hhmm]['scenario']].append(hhmm)

    print(f"\n  Scenario distribution ({sum(len(v) for v in ccf_sc.values())} events):")
    for sc, evs in sorted(ccf_sc.items(), key=lambda x: -len(x[1])):
        print(f"    {sc}: {len(evs)}")

    print(f"\n  {'Scenario':<30} {'N':>4} | {'LE A@1':>7} {'LE A@3':>7} {'LE A@5':>7} | {'TV A@1':>7} {'TV A@3':>7} {'TV A@5':>7}")
    print("  " + "-" * 90)
    for sc in sorted(ccf_sc.keys(), key=lambda s: -len(ccf_sc[s])):
        evs = ccf_sc[sc]
        le_h = [0,0,0]; tv_h = [0,0,0]; n_le = n_tv = 0
        for hhmm in evs:
            gv = ccf_gt[hhmm]
            lp = ccf_le.get(hhmm, {}).get('preds')
            if lp: n_le += 1; h = compute_ak(lp, gv, False); le_h = [a+b for a,b in zip(le_h, h)]
            tp = ccf_tv.get(hhmm, {}).get('preds')
            if tp: n_tv += 1; h = compute_ak(tp, gv, False); tv_h = [a+b for a,b in zip(tv_h, h)]
        ln = max(n_le,1); tn = max(n_tv,1)
        print(f"  {sc:<30} {len(evs):>4} | {le_h[0]/ln:>6.1%} {le_h[1]/ln:>6.1%} {le_h[2]/ln:>6.1%} | {tv_h[0]/tn:>6.1%} {tv_h[1]/tn:>6.1%} {tv_h[2]/tn:>6.1%}")

    # ===== Experiment 3: Latency =====
    print(f"\n{'=' * 70}")
    print("  Experiment 3: Latency Breakdown (GAIA)")
    print("=" * 70)
    for method in ["localexpert", "dualchannel", "tvdig", "hybrid"]:
        data = parse_events([f"logs/experiments_gaia_{method}_{d}.log" for d in "0704 0705 0706".split()])
        lats = [v['latency'] for v in data.values() if v.get('latency')]
        if lats:
            arr = np.array(lats)
            print(f"  {method:<14}: mean={np.mean(arr):.0f}s, median={np.median(arr):.0f}s, p95={np.percentile(arr,95):.0f}s, n={len(arr)}")

    print(f"\n{'=' * 70}")
    print("  ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
