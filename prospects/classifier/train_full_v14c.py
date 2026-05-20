"""Train v1.14c: panel includes 2021-2025 drafted players (right-censored),
90% player-grouped train + 10% direct calibration split.

Per-event right-censoring policy:
  - TOP_100_PROSPECT, MLB_DEBUT: exit-based censoring only.
  - ESTABLISHED_MLB: exit + 4-yr plausibility (don't count "not yet
                     established" rows for recent draftees as negatives).
  - STAR / ELITE: exit + 4-yr plausibility.

Calibration: 10% of players held out, scored at snap = draft_year + 2
(or first_milb_year + 2 for IFAs); Beta calibrators fit on those
(raw_prob, realized) pairs.

Usage:
    python -m prospects.classifier.train_full_v14c \\
        --panel panel_v1.14c.npz \\
        --out models/event_classifiers_v1.14c.pkl
"""
from __future__ import annotations

import argparse
import csv
import gc
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.classifier.architectures.survival import (
    ELITE_KEY, EVENT_TRIGGER_COL, EXIT_KEY, MAX_OBS_YEAR, STAR_KEY,
    _BetaCalibrator, _PlattCalibrator, _trigger_year,
    exit_labels, labels_and_eligibility,
    predict_cumulative_batch,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator
sys.modules["__main__"]._BetaCalibrator = _BetaCalibrator


# Per-event policy: (right_censor, min_years_to_fire)
# Fast events: exit-based only. Slow events: + 4-yr plausibility.
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

SNAPSHOT_OFFSET = 2  # snap = start_year + 2 for calibration scoring


def _ename(e) -> str:
    if isinstance(e, str):
        return e.lstrip("_")
    return e.name


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
    ap.add_argument("--out", default="models/event_classifiers_v1.14c.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cal-frac", type=float, default=0.10,
                    help="Fraction of players held out for direct calibration")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()

    print("=" * 70)
    print("  v1.14c TRAIN: 90/10 train/cal + per-event right-censoring")
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

    # Stats once for right-censoring
    print(f"\nLoading season_stats for right-censoring")
    db = ProspectDB(args.db)
    with db._connect() as conn:
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    del stats_rows

    # Player-grouped 90/10 split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(unique_players))
    n_cal = int(round(args.cal_frac * len(unique_players)))
    cal_players = {unique_players[i] for i in perm[:n_cal]}
    train_players = {unique_players[i] for i in perm[n_cal:]}
    print(f"  split: train={len(train_players):,}  cal={len(cal_players):,}")

    train_mask = np.array([p in train_players for p in pids_arr], dtype=bool)

    # ---- Phase 1: Train hazards on the 90% train slice ----
    print(f"\n--- Phase 1: Training hazards on 90% train slice ---")
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
                  f"{n_pos:>7d}  skip (pos out of range)")
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
    del X, X_tr_e, y_tr_e, elig_e, y_e_all
    gc.collect()

    # ---- Phase 2: Score the 10% cal slice at snap = start+2 ----
    print(f"\n--- Phase 2: Fitting calibrators on 10% cal slice ---")
    # Pull cal players' full records with outcomes for scoring
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory,
                   o.events_json, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
        """).fetchall()]
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source FROM prospect_rankings"
            ).fetchall()
        except Exception:
            rank_rows = []
    rankings_by_pid: dict[str, list] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for r in rows:
        r["_top100_rankings"] = rankings_by_pid.get(r["player_id"], [])

    cal_rows = [r for r in rows if r["player_id"] in cal_players]
    # Snap = start_year + 2; skip if past observation window or already
    # debuted by snap (no longer a prospect).
    snap_groups: dict[int, list[dict]] = {}
    for r in cal_rows:
        dy = r.get("draft_year")
        if dy is None:
            yrs = [int(s["season_year"]) for s in stats_by_pid.get(r["player_id"], [])
                   if s.get("season_year") is not None
                   and (s.get("level") or "").upper() != "MLB"]
            if not yrs: continue
            start = min(yrs)
        else:
            start = int(dy)
        snap = start + SNAPSHOT_OFFSET
        if snap >= MAX_OBS_YEAR:
            continue
        debut = r.get("mlb_debut_year")
        if debut is not None and debut <= snap:
            continue
        snap_groups.setdefault(snap, []).append(r)
    n_snap_total = sum(len(g) for g in snap_groups.values())
    print(f"  cal snapshots: {n_snap_total:,} across {len(snap_groups)} snap years")

    # Score each snap batch with trained hazards
    score_keys = [e for e in hazards if e != EXIT_KEY]
    per_event_preds: dict = {e: [] for e in score_keys}
    per_event_real: dict = {e: [] for e in score_keys}
    for snap, group in sorted(snap_groups.items()):
        sub_stats = {r["player_id"]: stats_by_pid.get(r["player_id"], [])
                     for r in group}
        out = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=snap, horizon=args.horizon,
        )
        for i, r in enumerate(group):
            for ev in score_keys:
                raw = out.get(("raw", ev), out.get(ev))
                per_event_preds[ev].append(float(raw[i]))
                trig = _trigger_year(r, ev)
                per_event_real[ev].append(
                    int(trig is not None and trig <= MAX_OBS_YEAR)
                )

    print(f"\n{'Event':<22} {'n':>7} {'pos':>5} {'a':>7} {'b':>7} {'c':>7}")
    print("-" * 60)
    for ev in score_keys:
        preds = np.array(per_event_preds[ev], dtype=np.float64)
        labels = np.array(per_event_real[ev], dtype=np.int8)
        pos = int(labels.sum())
        ename = _ename(ev)
        if pos < 5 or pos > len(labels) - 5:
            print(f"{ename:<22} {len(preds):>7,d} {pos:>5d}  "
                  f"skip (positives out of range)")
            continue
        cal = _BetaCalibrator().fit(preds, labels)
        hazards[ev]["calibrator"] = cal
        a, b, c = (float(cal.a), float(cal.b), float(cal.c))
        print(f"{ename:<22} {len(preds):>7,d} {pos:>5d} "
              f"{a:>7.3f} {b:>7.3f} {c:>7.3f}")

    # Save
    print(f"\nSaving {args.out}")
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
