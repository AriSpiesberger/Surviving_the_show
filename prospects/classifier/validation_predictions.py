"""
prospects/classifier/validation_predictions.py
================================================

Score every TEST-split holdout player at their prospect-state snapshot
(draft_year+2 or first-observed-MiLB+2 for IFAs) and join predicted P with
realized outcomes. Lets you audit which players the model got right vs wrong.

Output CSV columns:
    player_id, name, draft_year, is_international, snap_year
    + per event: p_<E>, p_<E>_raw, t_<E>_mean, t_<E>_sd, realized_<E>
    + diff_<E> = predicted - realized (signed; positive = over-predicted)

Usage:
    python -m prospects.classifier.validation_predictions \\
        --db prospects_snapshot.db \\
        --model models/event_classifiers_v1.2_scouting.pkl \\
        --out validation_predictions_v1.2.csv \\
        [--include-val]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, EVENT_TRIGGER_COL, _trigger_year, load_hazards,
    predict_cumulative_batch, MAX_OBS_YEAR,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def _ev_name(e) -> str:
    """Stable column name for either a CareerEvent enum or a string event key."""
    if hasattr(e, "name"):
        return e.name
    if isinstance(e, str):
        return e.lstrip("_")
    return str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects_snapshot.db")
    parser.add_argument("--model",
                        default="models/event_classifiers_v1.2_scouting.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--max-draft-year", type=int, default=2020,
                        help="Match the model's training cohort filter")
    parser.add_argument("--out", default="validation_predictions_v1.2.csv")
    parser.add_argument("--include-val", action="store_true",
                        help="Also include validation-split players (used to "
                             "fit the isotonic calibrator). Useful for sanity "
                             "but biases low.")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading model: {args.model}")
    hazards = load_hazards(args.model)
    # v1.11: emit predictions for every trained hazard, not just the
    # display tier. Lets bucket evaluation cover TOP_100, AS1, and any
    # other event the model knows about.
    display = {CareerEvent.TOP_100_PROSPECT, CareerEvent.MLB_DEBUT,
               CareerEvent.ESTABLISHED_MLB, CareerEvent.ALL_STAR_ONCE}
    event_keys = [k for k in hazards
                  if (k in display) or k == ELITE_KEY or k == STAR_KEY]
    print(f"  events: {[e.name if hasattr(e, 'name') else str(e) for e in event_keys]}")

    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year, o.year_established_mlb, o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= ?)
               OR COALESCE(p.is_international, 0) = 1
        """, (args.max_draft_year,)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    # Reproduce the survival training split (player-grouped)
    unique_players = sorted({r["player_id"] for r in rows})
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(unique_players))
    n_p = len(unique_players)
    n_test = int(round(0.10 * n_p))
    n_val = int(round(0.10 * n_p))
    test_players = set(unique_players[i] for i in perm[:n_test])
    val_players = set(unique_players[i] for i in perm[n_test:n_test + n_val])
    print(f"  player pool: {len(rows):,}")
    print(f"  test players: {len(test_players):,}")
    print(f"  val players:  {len(val_players):,}")

    if args.include_val:
        eval_players = test_players | val_players
        slice_label = "test+val"
    else:
        eval_players = test_players
        slice_label = "test"
    eval_rows = [r for r in rows if r["player_id"] in eval_players]
    print(f"  evaluating {len(eval_rows):,} players ({slice_label})")

    # Snapshot per player: draft_year+2 if drafted, first-observed-MiLB+2 if IFA.
    # Skip players who had already MLB-debuted by snapshot (not prospects then).
    snap_groups: dict[int, list[dict]] = {}
    for r in eval_rows:
        dy = r.get("draft_year")
        if dy is None:
            yrs = [s.get("season_year") for s in stats_by_pid.get(r["player_id"], [])
                   if s.get("season_year") is not None
                   and (s.get("level") or "").upper() != "MLB"]
            if not yrs:
                continue
            start = int(min(yrs))
        else:
            start = int(dy)
        snap = start + 2
        if snap >= MAX_OBS_YEAR:
            continue  # not enough horizon to verify
        debut = r.get("mlb_debut_year")
        if debut is not None and debut <= snap:
            continue  # was already MLB at snapshot
        snap_groups.setdefault(snap, []).append(r)

    total_scored = sum(len(g) for g in snap_groups.values())
    print(f"  usable snapshots: {total_scored:,}")
    print(f"  snapshot year span: {min(snap_groups):,}..{max(snap_groups):,}")

    # Score each snap-year batch
    print("Scoring...")
    out_rows = []
    for snap, group in sorted(snap_groups.items()):
        sub_stats = {r["player_id"]: stats_by_pid.get(r["player_id"], [])
                     for r in group}
        cumP = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=snap, horizon=args.horizon,
        )
        for i, r in enumerate(group):
            row = {
                "player_id": r["player_id"],
                "name": r["name"],
                "draft_year": r.get("draft_year"),
                "draft_round": r.get("draft_round"),
                "draft_pick": r.get("draft_pick"),
                "is_international": int(r.get("is_international") or 0),
                "primary_position": r.get("primary_position"),
                "snap_year": snap,
                "mlb_debut_year": r.get("mlb_debut_year"),
                "final_mlb_year": r.get("final_mlb_year"),
            }
            for e in event_keys:
                p_cal = float(cumP[e][i])
                p_raw_arr = cumP.get(("raw", e))
                mt_arr = cumP.get(("mean_t", e))
                sd_arr = cumP.get(("sd_t", e))
                p_raw = float(p_raw_arr[i]) if p_raw_arr is not None else p_cal
                mt = float(mt_arr[i]) if mt_arr is not None else float("nan")
                sd = float(sd_arr[i]) if sd_arr is not None else float("nan")
                ename = _ev_name(e)
                # realized = event triggered by end of MAX_OBS_YEAR
                trig = _trigger_year(r, e)
                realized = int(trig is not None and trig <= MAX_OBS_YEAR)
                # eligible_at_snap = event had NOT yet triggered as of the
                # snap year. For an honest "did the model predict this
                # future event correctly" eval, we should ignore rows
                # where the event already happened before snap (the
                # prediction is trivially 1.0). For backward compat we
                # still emit `realized_<E>`; downstream evaluators should
                # filter on `eligible_at_snap_<E>` = 1 for fair scoring.
                eligible_at_snap = int(trig is None or trig > snap)
                row[f"p_{ename}"] = round(p_cal, 4)
                row[f"p_{ename}_raw"] = round(p_raw, 4)
                row[f"t_{ename}_mean"] = round(mt, 2) if mt == mt else ""
                row[f"t_{ename}_sd"] = round(sd, 2) if sd == sd else ""
                row[f"realized_{ename}"] = realized
                row[f"eligible_at_snap_{ename}"] = eligible_at_snap
                row[f"diff_{ename}"] = round(p_cal - realized, 4)
                row[f"trigger_year_{ename}"] = trig
            out_rows.append(row)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows):,} rows to {args.out}")

    # Summary: per-event mean |diff| and confusion-ish counts
    print(f"\nPer-event summary on {slice_label} slice:")
    print(f"{'Event':<22} {'pos':>5} {'pred_mean':>10} {'real_rate':>10} "
          f"{'|diff|':>8}")
    print("-" * 60)
    for e in event_keys:
        ename = _ev_name(e)
        p = np.array([r[f"p_{ename}"] for r in out_rows])
        r_a = np.array([r[f"realized_{ename}"] for r in out_rows])
        print(f"{ename:<22} {int(r_a.sum()):>5d} "
              f"{p.mean()*100:>9.2f}% {r_a.mean()*100:>9.2f}% "
              f"{np.abs(p - r_a).mean():>7.3f}")


if __name__ == "__main__":
    main()
