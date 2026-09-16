"""
Offline fusion replay for the GAIA embedding-upgrade experiment (C3).

Phase A (expensive, once per checkpoint set): for every logged GAIA hybrid case,
re-assemble the online inputs (microcause metrics, tracerca spans, log dir) and
run the TVDiag localizer with each candidate checkpoint. Results cached to pkl.

Phase B (instant, offline): splice the new TVDiag rankings into the logged
per-case fusion inputs (MEPFL ranking, univariate metric string, conflict
scenario, proxy gamma) and re-run confidence_vote_fusion. Reports:
  F1  localizer fidelity (old checkpoint vs logged tvdig_services)
  F2  fusion replay fidelity (logged inputs vs logged fused ranking)
  F3  Hybrid AC@1/3/5 with each candidate checkpoint (frozen paper accounting)
  F4  standalone TVDiag AC@1/3/5 per checkpoint

Usage:
  python replay_hybrid_gaia.py --collect --checkpoints models/tvdig_checkpoint models/tvdig_checkpoint_gaia_v1_idf_s1 ...
  python replay_hybrid_gaia.py --score --checkpoints ...
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(PROJ, "Datasets/GAIA/fault_injection_tracerank")
CACHE_PATH = os.path.join(PROJ, "replay_gaia_localizer_cache.pkl")

LOG_SET = [f"logs/experiments_gaia_hybrid_{d}{s}.log"
           for d in ("0704", "0705", "0706") for s in ("", "_resume")]

CKPT_ALIASES = {
    "old": "models/tvdig_checkpoint",
}


# ---------------------------------------------------------------------
# Logged-case parsing
# ---------------------------------------------------------------------

def parse_hybrid_logs():
    """Per inner-case (date|HHMM): fused (last), tvdig (first), mepfl, uni-metrics, scenario."""
    cases = {}
    for path in LOG_SET:
        p = os.path.join(PROJ, path)
        if not os.path.exists(p):
            continue
        cur = None
        for line in open(p, errors="ignore"):
            line = line.strip()
            m = re.search(r'"type": "task_parsed", "date": "(\w+)", "time": "(\d{2})-(\d{2})"', line)
            if m:
                cur = cases.setdefault(f"{m.group(1)}|{m.group(2)}{m.group(3)}", {})
                continue
            if cur is None:
                continue
            mm = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
            if mm:
                try:
                    cur.setdefault("tvdig", eval(mm.group(1)))
                except Exception:
                    pass
                continue
            mm = re.search(r"Fused root services(?:\s*\(confidence-vote\))?:\s*(\[.*?\])", line)
            if mm:
                try:
                    cur["fused"] = eval(mm.group(1))
                except Exception:
                    pass
                continue
            if "MEPFL top-5 root services:" in line:
                cur["mepfl"] = [t.strip() for _, t in re.findall(r"\((\d)\)([^,\.]+)", line)]
                continue
            mm = re.search(r'"type": "root_metrics", "channel": "univariate",.*?"metrics": "(.*?)"\}', line)
            if mm:
                cur["uni_metrics"] = mm.group(1)
                continue
            mm = re.search(r"Conflict scenario:\s*(\w+)", line)
            if mm:
                cur.setdefault("scenario", mm.group(1))
    return cases


def load_gt_paper():
    """Frozen paper accounting: HHMM bare key, last-write-wins, single service."""
    gt = {}
    for d in ("2021-07-04", "2021-07-05", "2021-07-06"):
        with open(os.path.join(GT_DIR, f"fault_injection_list_{d}.pkl"), "rb") as f:
            for fi in pickle.load(f):
                if isinstance(fi, dict) and "time" in fi and "service" in fi:
                    gt[fi["time"].strftime("%H%M")] = fi["service"]
    return gt


def hit(preds, service, k):
    if not preds:
        return False
    return any(service in x for x in preds[:k])


# ---------------------------------------------------------------------
# Phase A: localizer collection
# ---------------------------------------------------------------------

def collect(checkpoints):
    from failure_localization import create_localizer as create_rca_localizer
    from util_funcs.loaddata import load
    # run.py import provides find_closest_file + the fusion function (same env)
    sys.path.insert(0, PROJ)
    import run as run_mod
    find_closest_file = run_mod.find_closest_file

    cases = parse_hybrid_logs()
    keys = sorted(cases)
    print(f"[collect] {len(keys)} logged cases, checkpoints: {checkpoints}")

    localizers = {tag: create_rca_localizer(method="tvdig", model_dir=os.path.join(PROJ, ck))
                  for tag, ck in checkpoints.items()}
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"[collect] resume: {len(cache)} cached case results")

    import contextlib, io
    failed = 0
    for ci, key in enumerate(keys):
        if key in cache and all(tag in cache[key] for tag in checkpoints):
            continue
        date, hhmm = key.split("|")
        time_result = f"{hhmm[:2]}-{hhmm[2:]}"
        try:
            mc_dirs = os.listdir(f"{date}_microcause")
            mc_file = find_closest_file(mc_dirs, time_result)
            dataa, data_head = load(f"{date}_microcause/{mc_file}", normalize=False,
                                    zero_fill_method="prevlatter", aggre_delta=1, verbose=False)
            tr_dirs = os.listdir(f"{date}_tracerca")
            tr_file = find_closest_file(tr_dirs, time_result)
            log_dirs = os.listdir(f"{date}_log_fault")
            log_file = find_closest_file(log_dirs, time_result)
            n_init = int(0.5 * len(dataa))
            mepfl = cases[key].get("mepfl", [])
            task = {
                "metric_data": dataa,
                "data_head": data_head,
                "n_init": n_init,
                "trace_data_path": f"./{date}_tracerca/{tr_file}",
                "log_dir": f"{date}_log_fault/{log_file}" if log_file else "",
                "root_services": mepfl[:5],
            }
            cache[key] = cache.get(key, {})
            with contextlib.redirect_stdout(io.StringIO()):
                for tag, loc in localizers.items():
                    if tag in cache[key]:
                        continue
                    res = loc.localize(task)
                    cache[key][tag] = {
                        "services": list(res.raw_root_services),
                        "scores": np.asarray(res.raw_root_scores).tolist(),
                    }
        except Exception as e:
            failed += 1
            cache[key] = cache.get(key, {})
            cache[key]["__error__"] = f"{type(e).__name__}: {e}"
        if (ci + 1) % 50 == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
            print(f"[collect] {ci+1}/{len(keys)} done (failed={failed})")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"[collect] finished: {len(cache)} cases, failed={failed}")


# ---------------------------------------------------------------------
# Phase B: offline scoring
# ---------------------------------------------------------------------

def gamma_proxy(data_head, uni_metrics_str):
    """Descending-score proxy for the univariate gamma vector (not logged)."""
    names = [t.strip() for _, t in re.findall(r"\((\d)\)([^,\.]+)", uni_metrics_str or "")]
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]
    gamma = np.zeros(len(data_head))
    for r, name in enumerate(names[:5]):
        if name in data_head:
            gamma[data_head.index(name)] = weights[r]
    return gamma


def score(checkpoints):
    sys.path.insert(0, PROJ)
    import run as run_mod
    fusion = run_mod.confidence_vote_fusion
    data_head_gaia = None  # per-case data_head needed; rebuild via loader cache

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    cases = parse_hybrid_logs()
    gt = load_gt_paper()

    # frozen accounting: fold inner cases to HHMM, last occurrence wins
    order = sorted(cases)
    fold = {}
    for k in order:
        hhmm = k.split("|")[1]
        fold[hhmm] = k

    def score_lists(get_pred):
        a = {1: 0, 3: 0, 5: 0}
        n = 0
        for hhmm, key in fold.items():
            p = get_pred(key)
            if not p or hhmm not in gt:
                continue
            n += 1
            for k in (1, 3, 5):
                a[k] += hit(p, gt[hhmm], k)
        return n, a

    # --- F1: localizer fidelity (old checkpoint) ---
    match5 = match1 = total = 0
    for key in order:
        c, ck = cases[key], cache.get(key, {})
        if "tvdig" not in c or "old" not in ck:
            continue
        total += 1
        logged, new = c["tvdig"], ck["old"]["services"]
        if logged[:5] == new[:5]:
            match5 += 1
        if logged[:1] == new[:1]:
            match1 += 1
    print(f"F1 localizer fidelity (old ckpt): top-5 exact {match5}/{total} = {match5/max(total,1):.0%}, "
          f"top-1 {match1}/{total} = {match1/max(total,1):.0%}")

    # --- F2/F3/F4: need data_head per case; rebuild once and cache heads ---
    heads_path = os.path.join(PROJ, "replay_gaia_data_heads.pkl")
    if os.path.exists(heads_path):
        with open(heads_path, "rb") as f:
            heads = pickle.load(f)
    else:
        heads = {}
    missing = [k for k in order if k not in heads and "__error__" not in cache.get(k, {})]
    if missing:
        from util_funcs.loaddata import load
        import contextlib, io
        import run as _r
        for key in missing:
            date, hhmm = key.split("|")
            try:
                mc_file = _r.find_closest_file(os.listdir(f"{date}_microcause"), f"{hhmm[:2]}-{hhmm[2:]}")
                with contextlib.redirect_stdout(io.StringIO()):
                    _, data_head = load(f"{date}_microcause/{mc_file}", normalize=False,
                                        zero_fill_method="prevlatter", aggre_delta=1, verbose=False)
                heads[key] = list(data_head)
            except Exception as e:
                heads[key] = None
        with open(heads_path, "wb") as f:
            pickle.dump(heads, f)

    for tag, ck_name in checkpoints.items():
        f2_m5 = f2_m1 = f2_tot = 0
        n, a = 0, {1: 0, 3: 0, 5: 0}
        ns, as_ = 0, {1: 0, 3: 0, 5: 0}
        for hhmm, key in fold.items():
            c, ck = cases[key], cache.get(key, {})
            if tag not in ck or c.get("fused") is None or "old" not in ck:
                continue
            head = heads.get(key)
            if not head:
                continue
            gamma = gamma_proxy(head, c.get("uni_metrics", ""))
            scen = c.get("scenario")
            mepfl = c.get("mepfl", [])
            # F2: replay with OLD tvdig ranking
            f_old, _, _ = fusion(tvdig_root_services=ck["old"]["services"],
                                 mepfl_root_services=mepfl[:5],
                                 dual_root_metrics=c.get("uni_metrics", ""),
                                 data_head=head, gamma=gamma, scenario=scen, top_k=5)
            old_logged = c.get("fused", [])
            if old_logged:
                f2_tot += 1
                if f_old[:5] == old_logged[:5]:
                    f2_m5 += 1
                if f_old[:1] == old_logged[:1]:
                    f2_m1 += 1
            # F3: splice candidate ranking
            f_new, _, _ = fusion(tvdig_root_services=ck[tag]["services"],
                                 mepfl_root_services=mepfl[:5],
                                 dual_root_metrics=c.get("uni_metrics", ""),
                                 data_head=head, gamma=gamma, scenario=scen, top_k=5)
            if hhmm in gt:
                n += 1
                for k in (1, 3, 5):
                    a[k] += hit(f_new, gt[hhmm], k)
            # F4: standalone
            if hhmm in gt:
                ns += 1
                for k in (1, 3, 5):
                    as_[k] += hit(ck[tag]["services"], gt[hhmm], k)
        print(f"[{tag}] F2 fusion fidelity: top-5 {f2_m5}/{f2_tot} = {f2_m5/max(f2_tot,1):.0%}, "
              f"top-1 {f2_m1}/{f2_tot} = {f2_m1/max(f2_tot,1):.0%}")
        print(f"[{tag}] F3 Hybrid  (N={n}): AC@1={a[1]/n:.1%} AC@3={a[3]/n:.1%} AC@5={a[5]/n:.1%}")
        print(f"[{tag}] F4 standalone (N={ns}): AC@1={as_[1]/ns:.1%} AC@3={as_[3]/ns:.1%} AC@5={as_[5]/ns:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--checkpoints", nargs="+", help="tag=path entries; 'old' alias predefined")
    args = ap.parse_args()

    checkpoints = {"old": CKPT_ALIASES["old"]}
    for item in (args.checkpoints or []):
        tag, _, path = item.partition("=")
        checkpoints[tag] = path

    if args.collect:
        collect(checkpoints)
    if args.score:
        score(checkpoints)


if __name__ == "__main__":
    main()
