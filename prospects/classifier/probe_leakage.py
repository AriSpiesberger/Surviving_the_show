"""
prospects/classifier/probe_leakage.py
=======================================

Diagnostic: re-sample as-of years constrained to be STRICTLY BEFORE MLB debut
(for players who reached MLB), and retrain the v0.2 classifier head-to-head
against the unconstrained sampler. The gap in AUC isolates how much of the
"high AUC" comes from level-leakage vs. genuine signal.

Usage:
    python -m prospects.classifier.probe_leakage [--db prospects.db]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

from prospects.features.windowed import (
    FEATURE_NAMES, N_FEATURES,
    build_windowed_features, y_for_event,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def _fit(X, y, seed=0):
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return None
    base = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, random_state=seed,
    )
    return CalibratedClassifierCV(base, method="isotonic", cv=3).fit(X, y)


def _build_dataset(db, seed, mask_post_debut: bool):
    rng = np.random.default_rng(seed)
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year
            FROM prospects p
            JOIN career_outcomes o ON p.player_id = o.player_id
        """).fetchall()
        prospects = [dict(r) for r in rows]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    X_list, joined = [], []
    for pr in prospects:
        pid = pr["player_id"]
        stats = stats_by_pid.get(pid, [])
        dy = pr.get("draft_year")
        lo = (dy or 1990) - 1
        hi = (dy or 2024) + 20
        debut = pr.get("mlb_debut_year")

        years = sorted({s["season_year"] for s in stats
                        if s.get("season_year") is not None
                        and lo <= s["season_year"] <= hi})
        if mask_post_debut and debut is not None:
            years = [y for y in years if y < int(debut)]
        if not years:
            yr = (dy + int(rng.integers(0, 5))) if dy else 2017
        else:
            yr = int(rng.choice(years))

        # Also strip stats rows >= debut when masking, so feature window can't see them
        if mask_post_debut and debut is not None:
            stats_use = [s for s in stats if s.get("season_year") < int(debut)]
        else:
            stats_use = stats

        X_list.append(build_windowed_features(pr, stats_use, yr))
        joined.append(pr)
    X = np.vstack(X_list) if X_list else np.zeros((0, N_FEATURES))
    return X, joined


def _eval_split(X, joined, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_test = int(0.10 * n)
    test = perm[:n_test]; train = perm[n_test:]
    X_tr, X_te = X[train], X[test]
    j_tr = [joined[i] for i in train]
    j_te = [joined[i] for i in test]

    print(f"{'Event':<22} {'AUC':>7} {'Brier':>8} {'n_pos':>7}")
    print("-" * 50)
    for event in CareerEvent.all_events():
        y_tr = y_for_event(j_tr, int(event))
        y_te = y_for_event(j_te, int(event))
        clf = _fit(X_tr, y_tr, seed=seed)
        if clf is None or y_te.sum() == 0 or y_te.sum() == len(y_te):
            print(f"{event.name:<22}     n/a      n/a {y_te.sum():>7d}")
            continue
        p = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_te, p)
        except Exception:
            auc = float("nan")
        brier = brier_score_loss(y_te, p)
        print(f"{event.name:<22} {auc:>7.3f} {brier:>8.4f} {y_te.sum():>7d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db = ProspectDB(args.db)

    print("=" * 60)
    print("A) UNCONSTRAINED (as in v0.2): random year, all stats")
    print("=" * 60)
    Xa, ja = _build_dataset(db, args.seed, mask_post_debut=False)
    _eval_split(Xa, ja, args.seed)

    print()
    print("=" * 60)
    print("B) PRE-DEBUT ONLY: as-of year < MLB debut, stats masked likewise")
    print("=" * 60)
    Xb, jb = _build_dataset(db, args.seed, mask_post_debut=True)
    _eval_split(Xb, jb, args.seed)


if __name__ == "__main__":
    main()
