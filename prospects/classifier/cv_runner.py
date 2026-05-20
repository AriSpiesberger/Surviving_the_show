"""5-fold cross-validated OOF prediction generator.

Each player is predicted exactly once by a model that never saw them.
Fold assignment is player-grouped and stratified by (draft_bucket,
debut-outcome) so each fold has a representative slice of every bucket.

Per fold:
  1. Train hazards on the other 4 folds (~80% of cohort).
  2. Within those 4 folds, carve a 10% internal validation slice for
     fitting Beta calibrators (this val slice is NEVER touched by the
     fold's held-out players, so OOF predictions are properly held out
     for both training and calibration).
  3. Refit calibrators on the internal val (using same multi-snapshot
     expansion as refit_calibrators.py with the smaller cal pool).
  4. Score the held-out fold at its snapshot year and store predictions.

Output:
  oof_predictions_v1.13.csv  — one row per player

Usage:
    python -m prospects.classifier.cv_runner \\
        --db prospects_snapshot.db \\
        --out oof_predictions_v1.13.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import pickle
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.classifier.architectures.survival import (
    ELITE_KEY,
    ELITE_COMPONENT_COLS,
    EVENT_TRIGGER_COL,
    EXIT_KEY,
    MAX_OBS_YEAR,
    STAR_KEY,
    STAR_COMPONENT_COLS,
    _BetaCalibrator,
    _PlattCalibrator,
    _trigger_year,
    build_hazard_panel,
    exit_labels,
    labels_and_eligibility,
    predict_cumulative_batch,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator
sys.modules["__main__"]._BetaCalibrator = _BetaCalibrator


N_FOLDS = 5
SNAPSHOT_OFFSETS = (1, 2, 3, 5)

# Per-event (right_censor, min_years_to_fire) policy for v1.14c training.
# Slow events get plausibility censoring; fast events use exit-based only.
V14C_EVENT_POLICY = {
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


def _bucket(p: dict) -> str:
    if int(p.get("is_international") or 0) == 1:
        return "IFA"
    rd = p.get("draft_round")
    if rd is None:
        return "UNK"
    rd = int(rd)
    if rd == 1: return "R1"
    if rd <= 3: return "R2-R3"
    if rd <= 10: return "R4-R10"
    return "R10+"


def _player_buckets_and_debut(joined: list[dict], pids: list[str]
                              ) -> tuple[dict[str, str], dict[str, int]]:
    by_pid: dict[str, dict] = {}
    for p, pid in zip(joined, pids):
        by_pid[pid] = p
    buckets = {pid: _bucket(p) for pid, p in by_pid.items()}
    debut = {pid: int(p.get("mlb_debut_year") is not None)
             for pid, p in by_pid.items()}
    return buckets, debut


def _stratified_folds(unique_players: list[str],
                      buckets: dict[str, str],
                      debut: dict[str, int],
                      n_folds: int = N_FOLDS,
                      seed: int = 42) -> list[set[str]]:
    """Assign each player to one of n_folds, stratified by (bucket, debut)."""
    rng = np.random.default_rng(seed)
    by_strat: dict[tuple, list[str]] = defaultdict(list)
    for pid in unique_players:
        by_strat[(buckets.get(pid, "UNK"), debut.get(pid, 0))].append(pid)
    fold_assign: dict[str, int] = {}
    for strat, pids in by_strat.items():
        perm = rng.permutation(len(pids))
        for i, idx in enumerate(perm):
            fold_assign[pids[idx]] = i % n_folds
    folds: list[set[str]] = [set() for _ in range(n_folds)]
    for pid, f in fold_assign.items():
        folds[f].add(pid)
    return folds


def _train_one_event(X_tr: np.ndarray, y_tr: np.ndarray,
                     seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        min_samples_leaf=30, random_state=seed,
        early_stopping=True, n_iter_no_change=10,
        validation_fraction=0.1,
    ).fit(X_tr, y_tr)


def _fit_fold_hazards(
    X: np.ndarray, pids: list[str], years: list[int],
    joined: list[dict], train_mask: np.ndarray,
    db_for_stats: ProspectDB, seed: int, verbose: bool = True,
    right_censor: bool = False,
    stats_by_pid: dict | None = None,
    censor_events: set[str] | None = None,
) -> dict:
    """Train all hazards on a subset (boolean train_mask over panel rows)."""
    results: dict = {}
    train_events = list(CareerEvent.all_events()) + [ELITE_KEY, STAR_KEY]
    for event in train_events:
        if (event not in (ELITE_KEY, STAR_KEY)
                and event not in EVENT_TRIGGER_COL):
            continue
        ename = event.name if hasattr(event, "name") else str(event).lstrip("_")
        apply_rc = bool(right_censor and (
            censor_events is None or ename in censor_events))
        # v14c policy: also apply per-event min_years_to_fire plausibility
        # censoring. Looked up from V14C_EVENT_POLICY when --policy v14c.
        min_yrs = 0
        if apply_rc:
            policy = V14C_EVENT_POLICY.get(ename, (True, 0))
            min_yrs = policy[1] if right_censor else 0
        eligible, y_all = labels_and_eligibility(
            joined, years, event,
            stats_by_pid=stats_by_pid, right_censor=apply_rc,
            min_years_to_fire=min_yrs, max_obs_year=MAX_OBS_YEAR,
        )
        tr = train_mask & eligible
        X_tr, y_tr = X[tr], y_all[tr]
        n_pos = int(y_tr.sum())
        if n_pos < 10 or n_pos > X_tr.shape[0] - 10:
            if verbose:
                ev_label = event.name if hasattr(event, "name") else str(event)
                print(f"    {ev_label}: skip (pos={n_pos})")
            continue
        clf = _train_one_event(X_tr, y_tr, seed)
        results[event] = {
            "hazard": clf,
            "feature_names": list(FEATURE_NAMES),
        }
        del X_tr, y_tr, eligible, y_all
        gc.collect()
    # Exit hazard
    with db_for_stats._connect() as conn:
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    elig_e, y_e_all = exit_labels(joined, years, stats_by_pid)
    tr_e = train_mask & elig_e
    X_tr_e, y_tr_e = X[tr_e], y_e_all[tr_e]
    if int(y_tr_e.sum()) >= 10:
        clf_e = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=30, random_state=seed,
        ).fit(X_tr_e, y_tr_e)
        results[EXIT_KEY] = {
            "hazard": clf_e,
            "feature_names": list(FEATURE_NAMES),
        }
    return results


def _fit_fold_calibrators(
    hazards: dict, db: ProspectDB, internal_val_players: set[str],
    max_obs_year: int, horizon: int = 15,
) -> None:
    """Fit Beta calibrators on internal val slice. Mutates hazards in-place."""
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= 2018)
               OR COALESCE(p.is_international, 0) = 1
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source FROM prospect_rankings"
            ).fetchall()
        except Exception:
            rank_rows = []
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    rankings_by_pid: dict[str, list] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for r in rows:
        r["_top100_rankings"] = rankings_by_pid.get(r["player_id"], [])

    val_rows = [r for r in rows if r["player_id"] in internal_val_players]
    groups_by_year: dict[int, list[dict]] = defaultdict(list)
    for r in val_rows:
        dy = r.get("draft_year")
        if dy is None:
            stat_yrs = [int(s["season_year"])
                        for s in stats_by_pid.get(r["player_id"], [])
                        if s.get("season_year") is not None]
            if not stat_yrs:
                continue
            start = min(stat_yrs)
        else:
            start = int(dy)
        for off in SNAPSHOT_OFFSETS:
            cur_year = start + off
            if cur_year >= max_obs_year:
                continue
            groups_by_year[cur_year].append(r)

    score_keys = [e for e in hazards if e != EXIT_KEY]
    per_event_preds: dict = {e: [] for e in score_keys}
    per_event_real: dict = {e: [] for e in score_keys}
    for cur_year, group in sorted(groups_by_year.items()):
        sub_stats = {r["player_id"]: stats_by_pid.get(r["player_id"], [])
                     for r in group}
        out = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=cur_year, horizon=horizon,
        )
        for i, r in enumerate(group):
            for ev in score_keys:
                raw = out.get(("raw", ev), out.get(ev))
                per_event_preds[ev].append(float(raw[i]))
                trig = _trigger_year(r, ev)
                per_event_real[ev].append(
                    int(trig is not None and trig <= max_obs_year)
                )

    for ev in score_keys:
        preds = np.array(per_event_preds[ev], dtype=np.float64)
        labels = np.array(per_event_real[ev], dtype=np.int8)
        pos = int(labels.sum())
        if pos < 3 or pos > len(labels) - 3:
            continue
        cal = _BetaCalibrator().fit(preds, labels)
        hazards[ev]["calibrator"] = cal


def _score_holdout_fold(
    hazards: dict, holdout_players: set[str], db: ProspectDB,
    max_obs_year: int, horizon: int = 15,
    offsets: tuple[int, ...] = (2,),
) -> list[dict]:
    """Predict on held-out fold at snap_year. Returns one row per
    (player, offset). When offsets=(2,) this reproduces v1.13/14
    behavior; multi-offset emits leakage-safe per-snap predictions
    for downstream stacking."""
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE ((p.draft_year IS NOT NULL AND p.draft_year <= 2020)
                   OR COALESCE(p.is_international, 0) = 1)
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source FROM prospect_rankings"
            ).fetchall()
        except Exception:
            rank_rows = []
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    rankings_by_pid: dict[str, list] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for r in rows:
        r["_top100_rankings"] = rankings_by_pid.get(r["player_id"], [])

    eval_rows = [r for r in rows if r["player_id"] in holdout_players]
    # (snap_year, offset) -> list of (player_row, offset)
    snap_groups: dict[int, list[tuple[dict, int]]] = {}
    for r in eval_rows:
        dy = r.get("draft_year")
        if dy is None:
            yrs = [s.get("season_year")
                   for s in stats_by_pid.get(r["player_id"], [])
                   if s.get("season_year") is not None
                   and (s.get("level") or "").upper() != "MLB"]
            if not yrs:
                continue
            start = int(min(yrs))
        else:
            start = int(dy)
        for offset in offsets:
            snap = start + offset
            if snap >= max_obs_year:
                continue
            debut = r.get("mlb_debut_year")
            if debut is not None and debut <= snap:
                continue
            snap_groups.setdefault(snap, []).append((r, offset))

    out_rows = []
    score_keys = [e for e in hazards if e != EXIT_KEY]
    for snap, group_pairs in sorted(snap_groups.items()):
        group = [r for r, _ in group_pairs]
        # IMPORTANT: trim stats to <= snap for leakage-safe scoring.
        sub_stats = {r["player_id"]: [s for s in stats_by_pid.get(r["player_id"], [])
                                       if (s.get("season_year") or 0) <= snap]
                     for r, _ in group_pairs}
        cumP = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=snap, horizon=horizon,
        )
        for i, (r, offset) in enumerate(group_pairs):
            row = {
                "player_id": r["player_id"],
                "name": r.get("name"),
                "draft_year": r.get("draft_year"),
                "draft_round": r.get("draft_round"),
                "draft_pick": r.get("draft_pick"),
                "is_international": int(r.get("is_international") or 0),
                "primary_position": r.get("primary_position"),
                "bucket": _bucket(r),
                "birth_date": r.get("birth_date"),
                "snap_year": snap,
                "snap_offset": offset,
                "mlb_debut_year": r.get("mlb_debut_year"),
                "final_mlb_year": r.get("final_mlb_year"),
            }
            for ev in score_keys:
                p_cal = float(cumP[ev][i])
                raw_arr = cumP.get(("raw", ev))
                p_raw = float(raw_arr[i]) if raw_arr is not None else p_cal
                if isinstance(ev, str):
                    ename = ev.lstrip("_")
                else:
                    ename = ev.name
                trig = _trigger_year(r, ev)
                realized = int(trig is not None and trig <= max_obs_year)
                eligible = int(trig is None or trig > snap)
                row[f"p_{ename}"] = round(p_cal, 5)
                row[f"p_{ename}_raw"] = round(p_raw, 5)
                row[f"realized_{ename}"] = realized
                row[f"eligible_at_snap_{ename}"] = eligible
                row[f"trigger_year_{ename}"] = trig
            out_rows.append(row)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--out", default="oof_predictions_v1.13.csv")
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-year", type=int, default=MAX_OBS_YEAR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--internal-val-frac", type=float, default=0.10,
                    help="Fraction of training partition used to fit fold's "
                         "Beta calibrator (default 0.10)")
    ap.add_argument("--fold", type=int, default=None,
                    help="If set, only run this single fold (0..N_FOLDS-1) "
                         "and write a partial output CSV. Use --merge to "
                         "concatenate fold-partial CSVs into the final out.")
    ap.add_argument("--panel", default="panel_v1.13.npz",
                    help="Path to a pre-built panel npz (from build_panel.py). "
                         "If present, skips the panel rebuild per fold.")
    ap.add_argument("--merge", action="store_true",
                    help="Concatenate <out>.fold{0..N-1}.csv into <out>.")
    ap.add_argument("--right-censor", action="store_true",
                    help="Enable right-censoring (drop rows past "
                         "last_active_year for non-firing events).")
    ap.add_argument("--censor-events", default="TOP_100_PROSPECT,MLB_DEBUT",
                    help="Comma-separated list of event names to apply "
                         "right-censoring to (only when --right-censor is "
                         "set). Default is the near-horizon events; "
                         "long-horizon events (ESTABLISHED, STAR, ELITE, "
                         "AS3+) keep v1.13 universal-eligible behavior.")
    ap.add_argument("--use-v14c-policy", action="store_true",
                    help="Use V14C_EVENT_POLICY for all events (right-"
                         "censor + per-event min_years_to_fire). "
                         "Overrides --censor-events when set.")
    ap.add_argument("--offsets", default="2",
                    help="Comma-separated snap_offsets to score each "
                         "held-out player at. Default '2' reproduces v1.13/14 "
                         "behavior. Use '0,1,2,3' for multi-offset OOF "
                         "(needed for stacking/Lasso composite).")
    args = ap.parse_args()

    # Merge mode: combine per-fold CSVs into the final OOF CSV.
    if args.merge:
        import os
        merged: list[dict] = []
        fieldnames = None
        for f in range(args.n_folds):
            path = f"{args.out}.fold{f}.csv"
            if not os.path.exists(path):
                raise SystemExit(f"missing fold partial: {path}")
            with open(path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    merged.append(row)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(merged)
        print(f"Merged {len(merged):,} OOF rows from {args.n_folds} folds "
              f"-> {args.out}")
        return

    print("=" * 70)
    print(f"  CV RUNNER ({args.n_folds}-fold OOF predictions)")
    print("=" * 70)

    db = ProspectDB(args.db)
    import os
    if os.path.exists(args.panel):
        print(f"\nLoading prebuilt panel from {args.panel}")
        with np.load(args.panel, allow_pickle=True) as d:
            X = d["X"].astype(np.float32, copy=False)
            pids = d["pids"].tolist()
            years = d["years"].tolist()
        with open(args.panel.replace(".npz", ".joined.pkl"), "rb") as fh:
            joined = pickle.load(fh)
    else:
        print(f"\nBuilding panel from {args.db}...")
        X, pids, years, joined = build_hazard_panel(
            db, max_draft_year=args.max_draft_year, max_year=args.max_year,
        )
        X = X.astype(np.float32, copy=False)
    pids_arr = np.array(pids)

    unique_players = sorted(set(pids))
    print(f"Panel: {len(pids):,} rows over {len(unique_players):,} players")

    buckets, debut = _player_buckets_and_debut(joined, pids)
    folds = _stratified_folds(unique_players, buckets, debut,
                              n_folds=args.n_folds, seed=args.seed)
    print(f"\nFold sizes: " + ", ".join(f"f{i}={len(s):,}"
                                         for i, s in enumerate(folds)))

    fold_range = ([args.fold] if args.fold is not None
                  else range(args.n_folds))
    all_oof_rows: list[dict] = []
    for fold_idx in fold_range:
        holdout_players = folds[fold_idx]
        train_players = set().union(*[
            folds[i] for i in range(args.n_folds) if i != fold_idx
        ])
        # Internal val: 10% of train, stratified by bucket
        train_list = sorted(train_players)
        rng = np.random.default_rng(args.seed + fold_idx)
        perm = rng.permutation(len(train_list))
        n_iv = int(round(args.internal_val_frac * len(train_list)))
        internal_val_players = set(train_list[i] for i in perm[:n_iv])
        # Train set excludes internal val
        train_only_players = train_players - internal_val_players

        train_mask = np.array(
            [p in train_only_players for p in pids_arr], dtype=bool
        )

        print(f"\n--- Fold {fold_idx+1}/{args.n_folds} ---")
        print(f"  train players: {len(train_only_players):,}  "
              f"internal_val: {len(internal_val_players):,}  "
              f"holdout: {len(holdout_players):,}")
        print(f"  train panel rows: {train_mask.sum():,}")

        print("  Training hazards...")
        # Build stats_by_pid once for right-censoring (no-op if disabled).
        stats_by_pid_train: dict[str, list] | None = None
        if args.right_censor:
            with db._connect() as conn:
                stats_rows = conn.execute(
                    "SELECT * FROM season_stats").fetchall()
            stats_by_pid_train = {}
            for s in stats_rows:
                d = dict(s)
                stats_by_pid_train.setdefault(
                    d["player_id"], []).append(d)
        if args.use_v14c_policy:
            censor_set = None  # all events (V14C_EVENT_POLICY decides min_yrs)
            print(f"  v14c policy: right-censor + per-event min_years_to_fire")
        else:
            censor_set = (set(s.strip() for s in args.censor_events.split(","))
                          if args.right_censor else None)
            if censor_set is not None:
                print(f"  right-censor applied to: {sorted(censor_set)}")
        hazards = _fit_fold_hazards(
            X, pids, years, joined, train_mask, db, seed=args.seed,
            verbose=False,
            right_censor=args.right_censor,
            stats_by_pid=stats_by_pid_train,
            censor_events=censor_set,
        )
        print(f"  Trained {len(hazards)} hazards")

        print("  Fitting Beta calibrators on internal val...")
        _fit_fold_calibrators(
            hazards, db, internal_val_players,
            max_obs_year=args.max_year, horizon=args.horizon,
        )

        print(f"  Scoring holdout fold...")
        offsets_tuple = tuple(int(s.strip()) for s in args.offsets.split(","))
        oof_rows = _score_holdout_fold(
            hazards, holdout_players, db,
            max_obs_year=args.max_year, horizon=args.horizon,
            offsets=offsets_tuple,
        )
        for r in oof_rows:
            r["fold"] = fold_idx
        all_oof_rows.extend(oof_rows)
        print(f"  Scored {len(oof_rows):,} holdout snapshots")

        del hazards
        gc.collect()

    print(f"\nTotal OOF predictions: {len(all_oof_rows):,}")
    if not all_oof_rows:
        print("No predictions written.")
        return

    # Stable column order
    fixed_cols = ["player_id", "name", "fold", "bucket", "draft_year",
                  "draft_round", "draft_pick", "is_international",
                  "primary_position", "snap_year",
                  "mlb_debut_year", "final_mlb_year"]
    extra_cols = sorted(c for c in all_oof_rows[0].keys()
                        if c not in fixed_cols)
    fieldnames = fixed_cols + extra_cols
    # If running a single fold, write to <out>.fold{N}.csv so the next
    # invocation doesn't overwrite it. Use --merge to combine when done.
    out_path = (f"{args.out}.fold{args.fold}.csv"
                if args.fold is not None else args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_oof_rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
