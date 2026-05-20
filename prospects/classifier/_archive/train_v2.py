"""
prospects/classifier/_archive/train_v2.py
==================================

Random-year-sample training for the prospect classifier.

Sampling protocol (per user spec):
  1. For each prospect, sample one random "as-of" year from their active
     years (or draft-year+~5 if no stats).
  2. Build a feature vector from pedigree + stats in the last 3 years ending
     at the as-of year (-1 imputation for missing).
  3. Split players into 90% train / 5% val / 5% test by player ID.
  4. Train one HistGradientBoosting classifier per CareerEvent on the 90%.
     Report metrics on val and test.

Each player contributes exactly one row to exactly one split.

Usage:
    python -m prospects.classifier.train_v2 \\
        [--db prospects.db] [--out models/event_classifiers_v0.2_windowed.pkl]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from prospects.classifier.model import EventClassifier, save_models, _ConstantClassifier
from prospects.features.windowed import (
    FEATURE_NAMES,
    N_FEATURES,
    build_training_dataset,
    y_for_event,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION_V2 = "v0.2-hgb-windowed"


def _fit_event(
    X: np.ndarray,
    y: np.ndarray,
    event: CareerEvent,
    n_bootstraps: int,
    seed: int,
) -> EventClassifier:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_pos = int(y.sum())
    if n_pos < 5 or n_pos > n - 5:
        base = max(n_pos / max(n, 1), 1e-4)
        return EventClassifier(event=event,
                               members=[_ConstantClassifier(base)],
                               n_train=n, n_pos=n_pos)
    members = []
    for k in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        if yb.sum() == 0:
            pos_idx = np.where(y == 1)[0][:5]
            Xb = np.vstack([Xb, X[pos_idx]])
            yb = np.concatenate([yb, np.ones(len(pos_idx), dtype=np.int8)])
        base = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, random_state=seed + k,
        )
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        try:
            clf.fit(Xb, yb)
            members.append(clf)
        except Exception:
            continue
    return EventClassifier(event=event, members=members,
                           n_train=n, n_pos=n_pos)


def _metrics(y_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, float, float]:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return (float("nan"),) * 3
    try:
        auc = roc_auc_score(y_true, p_pred)
    except Exception:
        auc = float("nan")
    brier = brier_score_loss(y_true, p_pred)
    try:
        ll = log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6))
    except Exception:
        ll = float("nan")
    return auc, brier, ll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--out", default="models/event_classifiers_v0.2_windowed.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Building windowed training dataset from {args.db}...")
    X, pids, years, joined = build_training_dataset(db, seed=args.seed)
    n = X.shape[0]
    print(f"  {n} players × {X.shape[1]} features (={N_FEATURES})")
    if n == 0:
        raise SystemExit("No labeled prospects. Run outcomes phase first.")

    # Stat coverage report
    stat_cols = [FEATURE_NAMES.index(f"pa_yT"),
                 FEATURE_NAMES.index(f"ip_yT")]
    has_stats = ((X[:, stat_cols] != -1.0).any(axis=1)).mean()
    print(f"  fraction of players with any stats in their as-of year: {has_stats:.1%}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(round(args.val_frac * n))
    n_test = int(round(args.test_frac * n))
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]

    print(f"  split: train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    print()

    X_tr = X[train_idx]
    X_va = X[val_idx]
    X_te = X[test_idx]
    j_tr = [joined[i] for i in train_idx]
    j_va = [joined[i] for i in val_idx]
    j_te = [joined[i] for i in test_idx]

    print(f"{'Event':<22} {'pos_tr':>7} {'pos_va':>7} {'pos_te':>7}  "
          f"{'AUC_va':>7} {'AUC_te':>7}  {'Brier_te':>9}  {'LL_te':>7}")
    print("-" * 92)

    final_models: dict = {}
    for event in CareerEvent.all_events():
        y_tr = y_for_event(j_tr, int(event))
        y_va = y_for_event(j_va, int(event))
        y_te = y_for_event(j_te, int(event))

        clf = _fit_event(X_tr, y_tr, event, args.bootstraps, args.seed)
        final_models[event] = clf

        # Predict on val & test using bootstrap ensemble mean
        p_va = clf.predict_proba(X_va).mean(axis=1) if y_va.size else np.array([])
        p_te = clf.predict_proba(X_te).mean(axis=1) if y_te.size else np.array([])
        auc_va, _, _ = _metrics(y_va, p_va)
        auc_te, brier_te, ll_te = _metrics(y_te, p_te)

        print(f"{event.name:<22} {y_tr.sum():>7d} {y_va.sum():>7d} {y_te.sum():>7d}  "
              f"{auc_va:>7.3f} {auc_te:>7.3f}  {brier_te:>9.4f}  {ll_te:>7.3f}")

    # Save the per-event ensembles (trained on train split only — held-out
    # validity preserved). To use the full data, re-run without val/test.
    save_models(final_models, args.out)
    print(f"\nSaved {len(final_models)} models -> {args.out}")
    print(f"Model version: {MODEL_VERSION_V2}")


if __name__ == "__main__":
    main()
