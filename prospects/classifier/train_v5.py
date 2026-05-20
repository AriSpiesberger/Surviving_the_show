"""
prospects/classifier/train_v5.py
==================================

v0.4 + selection-bias fix. Train only on draftees from years that have had
ENOUGH TIME to trigger career events.

The motivating bias: a 2023 draftee can't have triggered ESTABLISHED_MLB yet
(needs 2-4 years to reach MLB + 500 PA). Including them as "negatives" in
training teaches the model that good prospects don't reach the majors.

Default cutoff: train only on draft_year <= 2020 (i.e., 6+ years to develop
as of currentdate 2026). Trained model can still SCORE recent draftees at
inference time — they just don't pollute training labels.

Other settings unchanged from v0.4:
  - MiLB-only features
  - Future-only loss (per-event mask)
  - 90/5/5 split on the kept players

Usage:
    python -m prospects.classifier.train_v5 \\
        [--max-draft-year 2020] \\
        [--out models/event_classifiers_v0.5_pre2021_milb.pkl]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from prospects.classifier.model import save_models
from prospects.classifier.train_utils import _fit_event, _future_only_labels, _metrics
from prospects.features.windowed import (
    FEATURE_NAMES,
    N_FEATURES,
    build_training_dataset,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION_V5 = "v0.5-hgb-milb-pre2021"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--out", default="models/event_classifiers_v0.5_pre2021_milb.pkl")
    parser.add_argument("--max-draft-year", type=int, default=2020,
                        help="Train only on draftees with draft_year <= this")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Building MiLB-only dataset (draft_year <= {args.max_draft_year})...")
    X_all, pids_all, years_all, joined_all = build_training_dataset(
        db, seed=args.seed, milb_only=True,
    )
    # Filter to mature draftees
    keep = [i for i, j in enumerate(joined_all)
            if j.get("draft_year") and j["draft_year"] <= args.max_draft_year]
    X = X_all[keep]
    years = [years_all[i] for i in keep]
    joined = [joined_all[i] for i in keep]
    pids = [pids_all[i] for i in keep]

    print(f"  kept {len(joined)} (of {len(joined_all)} with MiLB rows)")
    print(f"  draft_year range in training: "
          f"{min((j['draft_year'] for j in joined), default='-')}..{args.max_draft_year}")
    print(f"  as-of year range: {min(years)}..{max(years)}")

    n = X.shape[0]
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
    print(f"Model version: {MODEL_VERSION_V5}")


if __name__ == "__main__":
    main()
