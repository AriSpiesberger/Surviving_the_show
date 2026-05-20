"""
prospects/classifier/_archive/train_v4.py
==================================

Pure prospect classifier: train only on MINOR-LEAGUE snapshots.

  - Sample as-of year only from years where the player had MiLB rows.
  - Features only see MiLB-level season_stats (MLB rows masked).
  - Future-only loss (per-event mask, unchanged from v3).

This forces the model to predict from prospect-state alone. No "they're
already in MLB" leakage. Players with zero MiLB rows in DB are dropped
from training (they reappear at inference if/when stats arrive).

Usage:
    python -m prospects.classifier.train_v4 \\
        [--db prospects.db] \\
        [--out models/event_classifiers_v0.4_milb_only.pkl]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from prospects.classifier.model import save_models
from prospects.classifier.train_utils import _fit_event, _future_only_labels, _metrics
from prospects.features.windowed import (
    FEATURE_NAMES,
    N_FEATURES,
    build_training_dataset,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION_V4 = "v0.4-hgb-milb-only"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--out", default="models/event_classifiers_v0.4_milb_only.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Building MiLB-only dataset from {args.db}...")
    X, pids, years, joined = build_training_dataset(
        db, seed=args.seed, milb_only=True,
    )
    n = X.shape[0]
    print(f"  {n} players × {X.shape[1]} features (={N_FEATURES})")
    print(f"  dropped (no MiLB rows): {db.count_prospects() - n}")
    print(f"  year span sampled: {min(years)}..{max(years)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(round(args.val_frac * n))
    n_test = int(round(args.test_frac * n))
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]

    print(f"  split: train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    print()
    print(f"{'Event':<22} {'n_tr':>6} {'pos_tr':>7} {'n_te':>6} {'pos_te':>6}  "
          f"{'AUC_te':>7} {'Brier_te':>9} {'Note':<22}")
    print("-" * 95)

    final_models = {}
    for event in CareerEvent.all_events():
        mask, y_all = _future_only_labels(joined, years, event)

        X_tr = X[train_idx][mask[train_idx]]
        y_tr = y_all[train_idx][mask[train_idx]]
        X_te = X[test_idx][mask[test_idx]]
        y_te = y_all[test_idx][mask[test_idx]]

        clf, fell_back = _fit_event(X_tr, y_tr, event, args.seed, args.bootstraps)
        final_models[event] = clf

        if X_te.shape[0] == 0 or y_te.sum() == 0:
            note = "no eval positives" if y_te.size and y_te.sum() == 0 else "no eval samples"
            print(f"{event.name:<22} {X_tr.shape[0]:>6d} {int(y_tr.sum()):>7d} "
                  f"{X_te.shape[0]:>6d} {int(y_te.sum()):>6d}      n/a       n/a  {note}")
            continue

        p_te = clf.predict_proba(X_te).mean(axis=1)
        auc, brier, _ = _metrics(y_te, p_te)
        note = "base-rate fallback" if fell_back else ""
        print(f"{event.name:<22} {X_tr.shape[0]:>6d} {int(y_tr.sum()):>7d} "
              f"{X_te.shape[0]:>6d} {int(y_te.sum()):>6d}  "
              f"{auc:>7.3f} {brier:>9.4f}  {note}")

    save_models(final_models, args.out)
    print(f"\nSaved {len(final_models)} models -> {args.out}")
    print(f"Model version: {MODEL_VERSION_V4}")


if __name__ == "__main__":
    main()
