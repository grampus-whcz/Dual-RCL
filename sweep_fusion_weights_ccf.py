"""
Hyperparameter sensitivity sweep for confidence-vote fusion (C3 / ICASSP fig).

Offline replay over the logged CCF anchorfix cases: for each sweep point we
re-run confidence_vote_fusion with one weight perturbed from its default and
score AC@1/3/5 under the frozen paper accounting. No pipeline rerun needed —
the fusion is a pure function of the logged per-case inputs.

Sweeps:
  w_D   (dual-channel confirmation)  0.1..0.9 step 0.1   default 0.3
  w_T   (MEPFL trace confirmation)   0.1..0.7 step 0.1   default 0.4
  w_c   (causal gamma)               0.1..0.5 step 0.1   default 0.3
  alpha (consensus bonus)            0.0..0.8 step 0.1   default 0.4

Usage:
  python sweep_fusion_weights_ccf.py --collect   # parse logs + rebuild data heads
  python sweep_fusion_weights_ccf.py --sweep     # run the sweeps, write JSON
"""

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ = os.path.dirname(os.path.abspath(__file__))
HEADS_PATH = os.path.join(PROJ, "replay_ccf_data_heads.pkl")
OUT_JSON = os.path.join(PROJ, "tex_ICASSP", "param_sensitivity_results.json")

TEST_DATES = {"0501t": "2022-05-01", "0503t": "2022-05-03", "0505t": "2022-05-05",
              "0507t": "2022-05-07", "0509t": "2022-05-09"}
VAL_DATES = {"0320b_anchorfix": "cloudbed-2", "0320c_anchorfix": "cloudbed-3"}
CCF_GT_TEST = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
CCF_GT_VAL = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"

DEFAULTS = {"w_causal": 0.3, "w_mepfl": 0.4, "w_dual": 0.3, "alpha": 0.4}
SWEEPS = {
    "w_dual": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "w_mepfl": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "w_causal": [0.1, 0.2, 0.3, 0.4, 0.5],
    "alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}


# ---------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------

def parse_log(path):
    cases = {}
    cur = None
    for line in open(path, errors="ignore"):
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
        mm = re.search(r"Fused root services(?:\s*\(confidence-vote\))?:\s*(\[.*?\])", line)
        if mm:
            try:
                cur["fused"] = eval(mm.group(1))
            except Exception:
                pass
    return cases


def load_cases():
    cases = {}
    for pref in list(TEST_DATES) + list(VAL_DATES):
        path = os.path.join(PROJ, f"logs/experiments_ccf_hybrid_{pref}.log")
        if not os.path.exists(path):
            continue
        for key, c in parse_log(path).items():
            cases[f"{pref}|{key}"] = c
    return cases


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------

def load_test_gt_paper():
    """Frozen paper accounting: HHMM bare key, last-write-wins over the 5 dates."""
    gt = {}
    for d in TEST_DATES.values():
        g = json.load(open(f"{CCF_GT_TEST}/groundtruth-{d}.json"))
        for i in range(len(g["timestamp"])):
            hhmm = datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            cmdb, level = g["cmdb_id"][i], g["level"][i]
            cands = {cmdb}
            if level == "pod":
                cands.add(re.sub(r"-\d+$", "", cmdb))
            elif level == "service":
                cands.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
            gt[hhmm] = cands
    return gt


def load_val_gt():
    import pandas as pd
    gt = {}
    for pref, cb in VAL_DATES.items():
        cb_num = cb.split("-")[-1]
        df = pd.read_csv(f"{CCF_GT_VAL}/groundtruth-k8s-{cb_num}-2022-03-20.csv")
        for _, row in df.iterrows():
            hhmm = datetime.fromtimestamp(row["timestamp"]).strftime("%H%M")
            cmdb, level = row["cmdb_id"], row["level"]
            cands = {cmdb}
            if level == "pod":
                cands.add(re.sub(r"-\d+$", "", cmdb))
            elif level == "service":
                cands.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
            gt.setdefault((pref, hhmm), set()).update(cands)
    return gt


# ---------------------------------------------------------------------
# data_head reconstruction (CCF normalization path of run.py)
# ---------------------------------------------------------------------

def rebuild_heads(cases):
    if os.path.exists(HEADS_PATH):
        with open(HEADS_PATH, "rb") as f:
            heads = pickle.load(f)
        print(f"[heads] resume: {len(heads)} cached")
    else:
        heads = {}
    import contextlib
    import io
    sys.path.insert(0, PROJ)
    import run as run_mod
    from util_funcs.loaddata import load

    missing = []
    for key, c in cases.items():
        pref, inner = key.split("|", 1)
        if c.get("tvdig") and heads.get(key) is None:
            missing.append(key)
    print(f"[heads] rebuilding {len(missing)} case heads")
    for i, key in enumerate(missing):
        pref, inner = key.split("|", 1)
        # validation log names carry an "_anchorfix" suffix that data dirs lack
        dir_pref = pref.replace("_anchorfix", "")
        hhmm = inner[-4:] if len(inner) >= 4 else inner
        try:
            mc_file = run_mod.find_closest_file(os.listdir(f"{dir_pref}_microcause"),
                                                f"{hhmm[:2]}-{hhmm[2:]}")
            with contextlib.redirect_stdout(io.StringIO()):
                dataa, data_head = load(f"{dir_pref}_microcause/{mc_file}", normalize=False,
                                        zero_fill_method="prevlatter", aggre_delta=1, verbose=False)
            # CCF normalization (run.py): remove constant columns + z-score + clip
            means = np.mean(dataa, axis=0, keepdims=True)
            stds = np.std(dataa, axis=0, keepdims=True)
            keep = (stds > 0).flatten()
            data_head = [data_head[j] for j in range(len(data_head)) if keep[j]]
            heads[key] = list(data_head)
        except Exception as e:
            heads[key] = None
            print(f"[heads] {key} failed: {type(e).__name__}: {e}")
        if (i + 1) % 25 == 0:
            with open(HEADS_PATH, "wb") as f:
                pickle.dump(heads, f)
            print(f"[heads] {i+1}/{len(missing)}")
    with open(HEADS_PATH, "wb") as f:
        pickle.dump(heads, f)
    print(f"[heads] done: {sum(1 for v in heads.values() if v)} valid / {len(heads)}")
    return heads


# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

def gamma_proxy(data_head, uni_metrics_str):
    names = [t.strip() for _, t in re.findall(r"\((\d)\)([^,\.]+)", uni_metrics_str or "")]
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]
    gamma = np.zeros(len(data_head))
    for r, name in enumerate(names[:5]):
        if name in data_head:
            gamma[data_head.index(name)] = weights[r]
    return gamma


def hit(preds, cands, k):
    if not preds:
        return False
    return any(any(c in x for c in cands) for x in preds[:k])


def sweep(cases, heads):
    sys.path.insert(0, PROJ)
    import run as run_mod
    fusion = run_mod.confidence_vote_fusion

    test_gt = load_test_gt_paper()
    val_gt = load_val_gt()

    # frozen paper accounting: fold inner cases to bare HHMM, last occurrence wins
    test_order = [k for k in sorted(cases) if not k.split("|", 1)[0].startswith("0320")]
    test_fold = {}
    for k in test_order:
        test_fold[k.rsplit("|", 1)[1][-4:]] = k

    def run_point(**kw):
        params = dict(DEFAULTS)
        params.update(kw)
        res = {"test": {1: 0, 3: 0, 5: 0}, "val": {1: 0, 3: 0, 5: 0}}
        n_test = n_val = 0
        f2_m1 = f2_m5 = f2_tot = 0
        # test
        fused_at_hhmm = {}
        for hhmm, key in test_fold.items():
            c, head = cases[key], heads.get(key)
            if not c.get("tvdig") or not head:
                continue
            gamma = gamma_proxy(head, c.get("uni_metrics", ""))
            f, _, _ = fusion(tvdig_root_services=c["tvdig"],
                             mepfl_root_services=c.get("mepfl", [])[:5],
                             dual_root_metrics=c.get("uni_metrics", ""),
                             data_head=head, gamma=gamma,
                             scenario=c.get("scenario"), top_k=5, **params)
            fused_at_hhmm[hhmm] = f
            if hhmm in test_gt:
                n_test += 1
                for k in (1, 3, 5):
                    res["test"][k] += hit(f, test_gt[hhmm], k)
            # fidelity at defaults vs logged fused
            if not any(kw) and c.get("fused"):
                f2_tot += 1
                if f[:1] == c["fused"][:1]:
                    f2_m1 += 1
                if f[:5] == c["fused"][:5]:
                    f2_m5 += 1
        # validation
        for key, c in sorted(cases.items()):
            pref, inner = key.split("|", 1)
            if not pref.startswith("0320"):
                continue
            head = heads.get(key)
            if not c.get("tvdig") or not head:
                continue
            hhmm = inner[-4:]
            gtc = val_gt.get((pref, hhmm))
            if not gtc:
                continue
            gamma = gamma_proxy(head, c.get("uni_metrics", ""))
            f, _, _ = fusion(tvdig_root_services=c["tvdig"],
                             mepfl_root_services=c.get("mepfl", [])[:5],
                             dual_root_metrics=c.get("uni_metrics", ""),
                             data_head=head, gamma=gamma,
                             scenario=c.get("scenario"), top_k=5, **params)
            n_val += 1
            for k in (1, 3, 5):
                res["val"][k] += hit(f, gtc, k)
        out = {"params": params,
               "test": {f"AC@{k}": (round(res["test"][k] / n_test * 100, 1) if n_test else None)
                        for k in (1, 3, 5)},
               "val": {f"AC@{k}": (round(res["val"][k] / n_val * 100, 1) if n_val else None)
                       for k in (1, 3, 5)}}
        fid = {"top1": round(f2_m1 / max(f2_tot, 1) * 100, 1),
               "top5": round(f2_m5 / max(f2_tot, 1) * 100, 1), "n": f2_tot} if not any(kw) else None
        return out, (n_test, n_val), fid

    results = {"defaults": {}, "sweeps": {}}
    d, (nt, nv), fid = run_point()
    d["fidelity_vs_logged"] = fid
    d["n_test"], d["n_val"] = nt, nv
    results["defaults"] = d
    print(f"defaults: {d}")

    for param, values in SWEEPS.items():
        results["sweeps"][param] = []
        for v in values:
            pt, _, _ = run_point(**{param: v})
            print(f"{param}={v}: test {pt['test']}  val {pt['val']}")
            results["sweeps"][param].append({"value": v, **pt})

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSweep results written to {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="parse logs + rebuild data heads")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    cases = load_cases()
    n_ok = sum(1 for c in cases.values() if c.get("tvdig"))
    print(f"[cases] {len(cases)} logged cases ({n_ok} with tvdig rankings)")
    if args.collect:
        rebuild_heads(cases)
    if args.sweep:
        heads = rebuild_heads(cases) if not os.path.exists(HEADS_PATH) else None
        if heads is None:
            with open(HEADS_PATH, "rb") as f:
                heads = pickle.load(f)
        sweep(cases, heads)


if __name__ == "__main__":
    main()
