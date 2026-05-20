"""
prospects/classifier/_archive/train.py
==============================

CLI entry point to train all CareerEvent classifiers from the labeled DB
and save them to disk. Reports training metrics per event.

Usage:
    python -m prospects.classifier.train [--db prospects.db] [--out models.pkl]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from prospects.classifier.model import (
    MODEL_VERSION,
    _train_one_event,
    _y_from_outcomes,
    save_models,
)
from prospects.features.build import N_FEATURES, build_training_matrix
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--out", default="models/event_classifiers.pkl")
    parser.add_argument("--bootstraps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="fraction held out for evaluation metrics")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading training matrix from {args.db}...")
    X, player_ids, outcomes = build_training_matrix(db, require_outcome=True)
    print(f"  {X.shape[0]} players × {X.shape[1]} features (={N_FEATURES} expected)")
    if X.shape[0] == 0:
        raise SystemExit("No labeled players. Run outcomes phase first.")

    idx_train, idx_eval = train_test_split(
        np.arange(X.shape[0]),
        test_size=args.holdout,
        random_state=args.seed,
        shuffle=True,
    )
    X_tr, X_ev = X[idx_train], X[idx_eval]
    outc_tr = [outcomes[i] for i in idx_train]
    outc_ev = [outcomes[i] for i in idx_eval]

    print(f"  train: {len(idx_train)} | eval: {len(idx_eval)}")
    print()

    models = {}
    print(f"{'Event':<22} {'n_pos_tr':>9} {'n_pos_ev':>9} {'AUC':>7} {'Brier':>7} {'LL':>7}")
    print("-" * 65)

    for event in CareerEvent.all_events():
        y_tr = _y_from_outcomes(outc_tr, event)
        y_ev = _y_from_outcomes(outc_ev, event)

        clf = _train_one_event(
            X_tr, y_tr, event,
            n_bootstraps=args.bootstraps,
            seed=args.seed,
            verbose=False,
        )
        models[event] = clf

        if y_ev.sum() == 0 or y_ev.sum() == len(y_ev):
            print(f"{event.name:<22} {y_tr.sum():>9d} {y_ev.sum():>9d}    n/a    n/a    n/a")
            continue

        P = clf.predict_proba(X_ev)
        p_mean = P.mean(axis=1)
        try:
            auc = roc_auc_score(y_ev, p_mean)
        except Exception:
            auc = float("nan")
        brier = brier_score_loss(y_ev, p_mean)
        try:
            ll = log_loss(y_ev, np.clip(p_mean, 1e-6, 1 - 1e-6))
        except Exception:
            ll = float("nan")
        print(f"{event.name:<22} {y_tr.sum():>9d} {y_ev.sum():>9d} "
              f"{auc:>7.3f} {brier:>7.3f} {ll:>7.3f}")

    # Retrain final models on the full dataset before saving
    print("\nRetraining final models on full data...")
    final_models = {}
    for event in CareerEvent.all_events():
        y = _y_from_outcomes(outcomes, event)
        final_models[event] = _train_one_event(
            X, y, event,
            n_bootstraps=args.bootstraps,
            seed=args.seed,
            verbose=False,
        )

    save_models(final_models, args.out)
    print(f"\nSaved {len(final_models)} models -> {args.out}")
    print(f"Model version: {MODEL_VERSION}")


if __name__ == "__main__":
    main()
