#!/usr/bin/env python3
"""
Train and evaluate the learned router.

The router predicts, per incident, whether to anchor the confidence-vote fusion
on the causal channel (label=1) or on TVDiag (label=0).

Evaluation: leave-one-dataset-out (train on GAIA, test on CCF, and vice versa) —
the true generalization test, since a router that only memorizes one dataset is
useless (that's the overfitting the user warned about).
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
FEATURE_NAMES = ['nodes', 'edges', 'density', 'q_tvdig', 'q_causal',
                 'tvdig_unique_top5', 'tvdig_top1_in_fused', 'tvdig_top1_pos_in_fused']


def load_data():
    with open(PROJ / "router_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    X = np.array([d['features'] for d in data])
    y = np.array([d['label'] for d in data])
    groups = ['gaia' if d['meta'].startswith('gaia') else 'ccf' for d in data]
    return X, y, groups, data


def evaluate_router(clf_class, clf_kwargs, name):
    """Within-dataset k-fold CV: for EACH dataset, split into train/test folds.
    This tests whether the router learns incident-level patterns that generalize
    WITHIN the dataset (standard ML practice), not cross-dataset."""
    from sklearn.model_selection import StratifiedKFold
    X, y, groups, data = load_data()
    print(f"\n=== {name} — Within-Dataset 5-Fold CV ===")
    for ds in ['gaia', 'ccf']:
        idx = [i for i, g in enumerate(groups) if g == ds]
        if len(idx) < 10:
            continue
        Xd, yd = X[idx], y[idx]
        # 5-fold within this dataset
        kf = StratifiedKFold(n_splits=min(5, len(idx)//2), shuffle=True, random_state=42)
        accs, base_accs = [], []
        all_preds, all_true = [], []
        for tr, te in kf.split(Xd, yd):
            scaler = StandardScaler().fit(Xd[tr])
            clf = clf_class(**clf_kwargs)
            clf.fit(scaler.transform(Xd[tr]), yd[tr])
            pred = clf.predict(scaler.transform(Xd[te]))
            accs.append(accuracy_score(yd[te], pred))
            maj = max(set(yd[tr].tolist()), key=list(yd[tr]).count)
            base_accs.append(np.mean(yd[te] == maj))
            all_preds.extend(pred.tolist()); all_true.extend(yd[te].tolist())
        print(f"  {ds} ({len(idx)} incidents): router acc={np.mean(accs):.1%} "
              f"(±{np.std(accs):.1%}), majority-baseline={np.mean(base_accs):.1%}")
        # Feature importance from last fold
        if hasattr(clf, 'coef_'):
            imp = np.abs(clf.coef_[0])
        else:
            imp = clf.feature_importances_
        top3 = np.argsort(imp)[::-1][:3]
        print(f"    top features: {[FEATURE_NAMES[i] for i in top3]}")


def train_final_router():
    """Train final router on ALL data, save for integration."""
    X, y, groups, data = load_data()
    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf.fit(scaler.transform(X), y)
    with open(PROJ / "router_model.pkl", "wb") as f:
        pickle.dump({'clf': clf, 'scaler': scaler, 'feature_names': FEATURE_NAMES}, f)
    print(f"\n=== Final router saved to router_model.pkl (trained on {len(y)} incidents) ===")
    return clf, scaler


if __name__ == "__main__":
    print(f"Dataset: feature matrix {load_data()[0].shape}")
    evaluate_router(LogisticRegression, {'max_iter': 1000, 'C': 0.5}, "Logistic Regression")
    evaluate_router(RandomForestClassifier, {'n_estimators': 50, 'max_depth': 4, 'random_state': 42}, "Random Forest")
    train_final_router()
