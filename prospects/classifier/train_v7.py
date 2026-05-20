"""
prospects/classifier/train_v7.py
==================================

Panel training: every pre-MLB year of every player is a training row.

For each prospect drafted in `--max-draft-year` or earlier:
  - Enumerate every season they had MiLB rows, strictly before MLB debut.
  - Build a feature vector at each such as-of year.
  - Future-only loss: per event, mask rows where the event triggered at or
    before that year; otherwise label 1 if it triggers after, 0 if never.

Split is by PLAYER (not by row) so multiple rows of the same player can't
straddle train and test (would leak).

Usage:
    python -m prospects.classifier.train_v7 \\
        [--max-draft-year 2020] \\
        [--max-as-of-year 2024] \\
        [--out models/event_classifiers_v0.7_panel.pkl]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from prospects.classifier.model import save_models
from prospects.classifier.train_utils import _fit_event, _future_only_labels, _metrics
from prospects.features.windowed import FEATURE_NAMES, N_FEATURES, build_panel_dataset
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION_V7 = "v0.7-hgb-panel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--out", default="models/event_classifiers_v0.7_panel.pkl")
    parser.add_argument("--max-draft-year", type=int, default=2020,
                        help="Train only on draftees <= this year")
    parser.add_argument("--max-as-of-year", type=int, default=2024,
                        help="Cap as-of-year to this (last fully complete season)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Building panel dataset (pre-MLB years per player, "
          f"draft <= {args.max_draft_year}, as_of <= {args.max_as_of_year})...")
    X_all, pids_all, years_all, joined_all = build_panel_dataset(
        db, require_outcome=True, max_year=args.max_as_of_year,
    )
    # Filter to mature draftees
    keep = [i for i, j in enumerate(joined_all)
            if j.get("draft_year") and j["draft_year"] <= args.max_draft_year]
    X = X_all[keep]
    pids = [pids_all[i] for i in keep]
    years = [years_all[i] for i in keep]
    joined = [joined_all[i] for i in keep]

    n_rows = X.shape[0]
    unique_players = sorted(set(pids))
    n_players = len(unique_players)
    print(f"  {n_rows:,} (player, year) rows from {n_players:,} unique players")
    print(f"  avg years/player: {n_rows / max(n_players, 1):.2f}")
    print(f"  year span: {min(years)}..{max(years)}")

    # ---- Player-grouped split ----
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_players)
    n_val = int(round(args.val_frac * n_players))
    n_test = int(round(args.test_frac * n_players))
    val_players = set(unique_players[i] for i in perm[:n_val])
    test_players = set(unique_players[i] for i in perm[n_val:n_val + n_test])
    train_players = set(unique_players[i] for i in perm[n_val + n_test:])

    split = np.array([
        "test" if p in test_players else ("val" if p in val_players else "train")
        for p in pids
    ])
    print(f"  player split: train {len(train_players):,} | "
          f"val {len(val_players):,} | test {len(test_players):,}")
    print(f"  row split:    train {(split == 'train').sum():,} | "
          f"val {(split == 'val').sum():,} | test {(split == 'test').sum():,}")
    print()
    print(f"{'Event':<22} {'n_tr':>7} {'pos_tr':>7} {'n_te':>6} {'pos_te':>6}  "
          f"{'AUC_te':>7} {'Brier_te':>9} {'Note':<22}")
    print("-" * 96)

    final_models = {}
    for event in CareerEvent.all_events():
        mask, y_all = _future_only_labels(joined, years, event)
        tr_sel = (split == "train") & mask
        te_sel = (split == "test") & mask
        X_tr, y_tr = X[tr_sel], y_all[tr_sel]
        X_te, y_te = X[te_sel], y_all[te_sel]

        clf, fell_back = _fit_event(X_tr, y_tr, event, args.seed, args.bootstraps)
        final_models[event] = clf

        if X_te.shape[0] == 0 or y_te.sum() == 0:
            note = "no eval positives" if y_te.size and y_te.sum() == 0 else "no eval samples"
            print(f"{event.name:<22} {X_tr.shape[0]:>7d} {int(y_tr.sum()):>7d} "
                  f"{X_te.shape[0]:>6d} {int(y_te.sum()):>6d}      n/a       n/a  {note}")
            continue

        p_te = clf.predict_proba(X_te).mean(axis=1)
        auc, brier, _ = _metrics(y_te, p_te)
        note = "base-rate fallback" if fell_back else ""
        print(f"{event.name:<22} {X_tr.shape[0]:>7d} {int(y_tr.sum()):>7d} "
              f"{X_te.shape[0]:>6d} {int(y_te.sum()):>6d}  "
              f"{auc:>7.3f} {brier:>9.4f}  {note}")

    save_models(final_models, args.out)
    print(f"\nSaved {len(final_models)} models -> {args.out}")
    print(f"Model version: {MODEL_VERSION_V7}")


if __name__ == "__main__":
    main()
