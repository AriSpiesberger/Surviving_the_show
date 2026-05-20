"""
prospects/classifier/train_utils.py
==================================

Future-only training: for each (player, as-of-year) sample we know if/when
each event triggered. For event E:
  - If trigger_year(E) <= as_of_year, the event has already happened. We MASK
    this sample for event E's training and evaluation (the label is no longer
    forward-looking).
  - Else label = 1 if trigger_year(E) > as_of_year, else 0.

This way the classifier only learns to predict events that haven't yet
happened as of the prediction date — the correct semantics for prospect-card
EV.

Player splits are 90% train / 5% val / 5% test, drawn once globally. Per-event
filtering happens within each split.

Low-N events (ALL_STAR_THREE_PLUS, MAJOR_AWARD, HOF_TRAJECTORY) train with a
constant base-rate model and are flagged.

Usage:
    python -m prospects.classifier.train_utils \\
        [--db prospects.db] \\
        [--out models/event_classifiers_v0.3_future_only.pkl]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from prospects.classifier.model import EventClassifier, _ConstantClassifier, save_models
from prospects.features.windowed import (
    FEATURE_NAMES,
    N_FEATURES,
    build_training_dataset,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION_V3 = "v0.3-hgb-future-only"

# Map CareerEvent -> the trigger_year column name in career_outcomes.
EVENT_TO_TRIGGER_COL = {
    CareerEvent.TOP_100_PROSPECT: "year_top_100",
    CareerEvent.TOP_25_PROSPECT: "year_top_25",
    CareerEvent.MLB_DEBUT: "mlb_debut_year",
    CareerEvent.ESTABLISHED_MLB: "year_established_mlb",
    CareerEvent.ALL_STAR_ONCE: "year_all_star_once",
    CareerEvent.ALL_STAR_THREE_PLUS: "year_all_star_three",
    CareerEvent.MAJOR_AWARD: "year_major_award",
    CareerEvent.HOF_TRAJECTORY: "year_hof_trajectory",
}

LOW_N_THRESHOLD = 30  # in TRAIN set; below this we fall back to base rate


def _trigger_year_for(row: dict, event: CareerEvent) -> int | None:
    col = EVENT_TO_TRIGGER_COL[event]
    val = row.get(col)
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _future_only_labels(
    joined: list[dict],
    years: list[int],
    event: CareerEvent,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, y) where mask=1 means use this sample, y is 0/1.

    A sample is masked when the event already triggered at/before the as-of
    year. Among unmasked samples, y=1 iff event triggered strictly after
    as-of year.
    """
    n = len(joined)
    mask = np.zeros(n, dtype=bool)
    y = np.zeros(n, dtype=np.int8)
    for i, (row, yr) in enumerate(zip(joined, years)):
        trigger = _trigger_year_for(row, event)
        if trigger is not None and trigger <= yr:
            continue  # already happened — mask
        mask[i] = True
        y[i] = 1 if (trigger is not None and trigger > yr) else 0
    return mask, y


def _fit_event(X, y, event, seed, n_boot):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_pos = int(y.sum())
    if n_pos < LOW_N_THRESHOLD or n_pos > n - LOW_N_THRESHOLD:
        base = max(n_pos / max(n, 1), 1e-4)
        return EventClassifier(event=event, members=[_ConstantClassifier(base)],
                               n_train=n, n_pos=n_pos), True
    members = []
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        if yb.sum() == 0:
            pos = np.where(y == 1)[0][:5]
            Xb = np.vstack([Xb, X[pos]])
            yb = np.concatenate([yb, np.ones(len(pos), dtype=np.int8)])
        base = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, random_state=seed + k,
        )
        try:
            clf = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xb, yb)
            members.append(clf)
        except Exception:
            continue
    return EventClassifier(event=event, members=members,
                           n_train=n, n_pos=n_pos), False


def _metrics(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return (float("nan"),) * 3
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    brier = brier_score_loss(y, p)
    try:
        ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    except Exception:
        ll = float("nan")
    return auc, brier, ll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--out", default="models/event_classifiers_v0.3_future_only.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Building windowed dataset from {args.db}...")
    X, pids, years, joined = build_training_dataset(db, seed=args.seed)
    n = X.shape[0]
    print(f"  {n} players × {X.shape[1]} features (={N_FEATURES})")

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

        tr_mask = mask[train_idx]
        te_mask = mask[test_idx]
        X_tr = X[train_idx][tr_mask]
        y_tr = y_all[train_idx][tr_mask]
        X_te = X[test_idx][te_mask]
        y_te = y_all[test_idx][te_mask]

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
    print(f"Model version: {MODEL_VERSION_V3}")


if __name__ == "__main__":
    main()
