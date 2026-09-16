"""
Confidence-signal audit for margin-gated anchor protection (C3 / Phase 4).

Fits a scalar temperature on the held-out train-tail slice (root-assignment NLL
under softmax(root_logits / T)), then computes the AUC of each candidate
confidence signal for predicting "argmax node == root" on the test split.

Signals:
  s1  raw logit margin (top1-top2)/(top1-min)      — the current q_tvdig style
  s2  probability margin at fitted T
  s3  (1 - probability entropy) at fitted T
  s4  top-1 probability at fitted T
  s5  cross-modal node-embedding agreement of the top-1 node
  s6  FTI head max probability

Usage:
  python -m failure_localization.audit_confidence --dump models/tvdig_checkpoint_gaia_base/root_logits_dump.pkl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def softmax(x, T=1.0):
    z = x / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_temperature(records):
    """Grid + refine search of T minimizing root-assignment NLL on train-tail."""
    def nll(T):
        total = 0.0
        for r in records:
            p = softmax(r["root_logits"], T)
            total -= np.log(max(p[r["root_idx"]], 1e-12))
        return total / len(records)

    grid = np.exp(np.linspace(np.log(0.05), np.log(50.0), 120))
    vals = [nll(T) for T in grid]
    best = grid[int(np.argmin(vals))]
    fine = np.linspace(best / 2, best * 2, 200)
    best = fine[int(np.argmin([nll(T) for T in fine]))]
    return float(best), float(nll(best))


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def signals(record, T):
    logits = np.asarray(record["root_logits"], dtype=float)
    order = np.argsort(logits)[::-1]
    top1, top2 = logits[order[0]], logits[order[1]] if len(order) > 1 else logits[order[0]] - 1.0
    lo = logits.min()
    s1 = (top1 - top2) / max(top1 - lo, 1e-9)

    p = softmax(logits, T)
    order_p = np.argsort(p)[::-1]
    p1, p2 = p[order_p[0]], p[order_p[1]] if len(order_p) > 1 else p[order_p[0]] - p[order_p[0]]
    s2 = (p1 - p2) / max(p1 - p.min(), 1e-9)

    ent = -np.sum(p * np.log(np.clip(p, 1e-12, None))) / np.log(len(p))
    s3 = 1.0 - ent
    s4 = p1

    es = record.get("es") or {}
    if len(es) == 3:
        vecs = []
        for mod, arr in es.items():
            v = arr[order[0]]
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
        agree = np.mean([float(np.dot(vecs[i], vecs[j]))
                         for i in range(3) for j in range(i + 1, 3)])
        s5 = agree
    else:
        s5 = float("nan")

    tl = record.get("type_logits")
    s6 = (float(np.exp(tl.max() - tl.max())) if tl is None else
          float(np.exp(tl)[np.argmax(tl)] / np.sum(np.exp(tl)))) if tl is not None else float("nan")
    if tl is not None:
        z = tl - tl.max()
        e = np.exp(z)
        s6 = float(e.max() / e.sum())

    return {"s1_logit_margin": s1, "s2_prob_margin": s2, "s3_conf_entropy": s3,
            "s4_top1_prob": s4, "s5_crossmodal_agree": s5, "s6_fti_prob": s6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    with open(args.dump, "rb") as f:
        import pickle
        records = pickle.load(f)
    train_recs = [r for r in records if r["split"] == "train"]
    test_recs = [r for r in records if r["split"] == "test"]

    T, train_nll = fit_temperature(train_recs)
    print(f"Fitted temperature T = {T:.3f} (train-tail NLL {train_nll:.4f}, n={len(train_recs)})")

    labels = [int(np.argmax(np.asarray(r["root_logits"])) == r["root_idx"]) for r in test_recs]
    print(f"Test: n={len(test_recs)}, top-1 accuracy={np.mean(labels):.1%}\n")

    sig_names = None
    table = {}
    for r in test_recs:
        sig = signals(r, T)
        sig_names = list(sig)
        for k, v in sig.items():
            table.setdefault(k, []).append(v)

    print(f"{'signal':<22}{'AUC(test)':>10}")
    results = {}
    for k in sig_names:
        vals = table[k]
        a = auc(vals, labels)
        results[k] = a
        print(f"{k:<22}{a:>10.3f}")

    out = os.path.splitext(args.dump)[0] + "_audit.json"
    with open(out, "w") as f:
        json.dump({"temperature": T, "train_tail_nll": train_nll,
                   "test_acc": float(np.mean(labels)), "auc": results}, f, indent=2)
    print(f"\nAudit written to {out}")


if __name__ == "__main__":
    main()
