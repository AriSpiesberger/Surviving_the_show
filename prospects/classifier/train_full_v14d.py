"""Train v1.14d: 80% train / 10% lasso-fit / 10% lasso-val split, no Beta calibration.

Player-grouped split (seed=42). v14c right-censoring policy applied per
event during hazard training. NO Beta calibrators fit — the Lasso meta-
learner is expected to handle final calibration on top of raw hazard outputs.

Saved artifacts:
  models/event_classifiers_v1.14d.pkl     hazards only (no calibrators)
  models/v1.14d_lasso_fit_players.txt     player_id, one per line
  models/v1.14d_lasso_val_players.txt     player_id, one per line

Usage:
    python -m prospects.classifier.train_full_v14d \\
        --panel panel_v1.14c.npz \\
        --out models/event_classifiers_v1.14d.pkl
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from prospects.classifier.architectures.survival import (
    ELITE_KEY, EVENT_TRIGGER_COL, EXIT_KEY, MAX_OBS_YEAR, STAR_KEY,
    _BetaCalibrator, _PlattCalibrator,
    exit_labels, labels_and_eligibility,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator
sys.modules["__main__"]._BetaCalibrator = _BetaCalibrator


EVENT_POLICY = {
    "TOP_100_PROSPECT": (True, 0),
    "MLB_DEBUT": (True, 0),
    "ESTABLISHED_MLB": (True, 4),
    "ALL_STAR_ONCE": (True, 4),
    "ALL_STAR_THREE_PLUS": (True, 6),
    "MAJOR_AWARD": (True, 5),
    "HOF_TRAJECTORY": (True, 10),
    "ELITE": (True, 5),
    "STAR": (True, 4),
}


def _ename(e):
    return e.name if hasattr(e, "name") else str(e).lstrip("_")


def _train_event(X_tr, y_tr, seed):
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        min_samples_leaf=30, random_state=seed,
        early_stopping=True, n_iter_no_change=10,
        validation_fraction=0.1,
    ).fit(X_tr, y_tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_v1.14c.npz")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--out", default="models/event_classifiers_v1.14d.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lasso-fit-frac", type=float, default=0.10)
    ap.add_argument("--lasso-val-frac", type=float, default=0.10)
    ap.add_argument("--max-entry-year", type=int, default=None,
                    help="If set, restrict the player universe to those whose "
                         "entry year (= draft_year, or earliest non-MLB "
                         "season_year for IFAs) is <= this. Used by the TEST "
                         "pipeline so the 80/10/10 split has mature outcomes.")
    args = ap.parse_args()

    print("=" * 70)
    print("  v1.14d TRAIN: 80% train / 10% lasso-fit / 10% lasso-val (no Beta cal)")
    print("=" * 70)

    print(f"\nLoading panel from {args.panel}")
    with np.load(args.panel, allow_pickle=True) as d:
        X = d["X"].astype(np.float32, copy=False)
        pids = d["pids"].tolist()
        years = d["years"].tolist()
    with open(args.panel.replace(".npz", ".joined.pkl"), "rb") as fh:
        joined = pickle.load(fh)
    pids_arr = np.array(pids)
    unique_players = sorted(set(pids))
    print(f"  panel: {X.shape[0]:,} rows over "
          f"{len(unique_players):,} players, {X.shape[1]} features")

    # 80/10/10 player-grouped split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(unique_players))
    n_fit = int(round(args.lasso_fit_frac * len(unique_players)))
    n_val = int(round(args.lasso_val_frac * len(unique_players)))
    fit_players = {unique_players[i] for i in perm[:n_fit]}
    val_players = {unique_players[i] for i in perm[n_fit:n_fit + n_val]}
    train_players = {unique_players[i] for i in perm[n_fit + n_val:]}
    print(f"  split: train={len(train_players):,}  "
          f"lasso_fit={len(fit_players):,}  lasso_val={len(val_players):,}")

    train_mask = np.array([p in train_players for p in pids_arr], dtype=bool)
    print(f"  panel rows in train: {train_mask.sum():,}")

    # Stats for right-censoring
    print(f"\nLoading season_stats for right-censoring")
    db = ProspectDB(args.db)
    with db._connect() as conn:
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    del stats_rows

    # Train hazards on 80% train slice
    print(f"\n--- Training hazards on 80% train slice ---")
    print(f"{'Event':<22} {'policy':<22} {'n_train':>9} {'pos':>7}")
    print("-" * 60)
    hazards: dict = {}
    train_events = list(CareerEvent.all_events()) + [ELITE_KEY, STAR_KEY]
    for event in train_events:
        if (event not in (ELITE_KEY, STAR_KEY)
                and event not in EVENT_TRIGGER_COL):
            continue
        ename = _ename(event)
        rc, min_yrs = EVENT_POLICY.get(ename, (True, 0))
        eligible, y_all = labels_and_eligibility(
            joined, years, event,
            stats_by_pid=stats_by_pid,
            right_censor=rc,
            min_years_to_fire=min_yrs,
            max_obs_year=MAX_OBS_YEAR,
        )
        tr_mask = train_mask & eligible
        X_tr, y_tr = X[tr_mask], y_all[tr_mask]
        n_pos = int(y_tr.sum())
        policy_str = f"rc={rc},min_yrs={min_yrs}"
        if n_pos < 10 or n_pos > X_tr.shape[0] - 10:
            print(f"{ename:<22} {policy_str:<22} {X_tr.shape[0]:>9,d} "
                  f"{n_pos:>7d}  skip")
            continue
        clf = _train_event(X_tr, y_tr, seed=args.seed)
        print(f"{ename:<22} {policy_str:<22} {X_tr.shape[0]:>9,d} {n_pos:>7d}")
        hazards[event] = {
            "hazard": clf,
            "feature_names": list(FEATURE_NAMES),
        }
        del X_tr, y_tr, eligible, y_all
        gc.collect()

    # Exit hazard
    elig_e, y_e_all = exit_labels(joined, years, stats_by_pid)
    tr_e = train_mask & elig_e
    X_tr_e, y_tr_e = X[tr_e], y_e_all[tr_e]
    if int(y_tr_e.sum()) >= 10:
        clf_e = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=30, random_state=args.seed,
        ).fit(X_tr_e, y_tr_e)
        hazards[EXIT_KEY] = {
            "hazard": clf_e,
            "feature_names": list(FEATURE_NAMES),
        }
        print(f"EXIT trained on {X_tr_e.shape[0]:,} rows, "
              f"pos={int(y_tr_e.sum()):,}")

    # NO Beta calibration — Lasso handles it
    print(f"\nNO Beta calibrators fit (Lasso replaces). "
          f"Hazards emit raw probs only.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out}")

    base = args.out.replace(".pkl", "")
    with open(f"{base}_lasso_fit_players.txt", "w") as f:
        for p in sorted(fit_players): f.write(p + "\n")
    with open(f"{base}_lasso_val_players.txt", "w") as f:
        for p in sorted(val_players): f.write(p + "\n")
    print(f"Wrote lasso-fit/val player lists")


if __name__ == "__main__":
    main()
