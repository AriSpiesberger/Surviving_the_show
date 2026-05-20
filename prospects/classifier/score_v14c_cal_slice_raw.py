"""Score v1.14c's 10% cal slice at multi-snap with RAW probabilities.

Reproduces the seed=42 perm over panel_v1.14c players, takes the first
10% (v1.14c's calibration slice — the players held out from v1.14c hazard
training). Filters to entry_year <= 2020 so outcomes are mature.
Scores each at snap_offset 0..max_offset with RAW (uncalibrated)
cumulative event probabilities.

Output mirrors val_v14b_long.csv schema so it drops into lasso_composite.py
unchanged. The `p_<event>` columns hold RAW probs (no Beta calibrator).

Usage:
    python -m prospects.classifier.score_v14c_cal_slice_raw \\
        --model models/event_classifiers_v1.14c.pkl \\
        --panel panel_v1.14c.npz \\
        --out v14c_calslice_pre2021_raw_long.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, _trigger_year, load_hazards,
    predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def _entry_year(player: dict, stats_by_pid: dict) -> int | None:
    dy = player.get("draft_year")
    is_intl = int(player.get("is_international") or 0)
    if dy is not None and not is_intl:
        return int(dy)
    yrs = [s.get("season_year") for s in stats_by_pid.get(player["player_id"], [])
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    if yrs:
        return int(min(yrs))
    if dy is not None:
        return int(dy)
    return None


def _bucket_of(player: dict) -> str:
    if int(player.get("is_international") or 0) == 1:
        return "IFA"
    r = player.get("draft_round")
    if r is None:
        return "IFA"
    r = int(r)
    if r == 1: return "R1"
    if r <= 3: return "R2-R3"
    if r <= 10: return "R4-R10"
    return "R10+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--model", default="models/event_classifiers_v1.14c.pkl")
    ap.add_argument("--panel", default="panel_v1.14c.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cal-frac", type=float, default=0.10,
                    help="Match the cal_frac used to train v1.14c")
    ap.add_argument("--players-file", default=None,
                    help="If set, score exactly these player_ids (one per "
                         "line). Overrides the seed-based cal-slice logic.")
    ap.add_argument("--max-entry-year", type=int, default=2020,
                    help="Right-censor: only score players whose entry_year "
                         "<= this. Default 2020 = mature outcomes by 2026.")
    ap.add_argument("--observe-through", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--max-offset", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.players_file:
        with open(args.players_file) as f:
            cal_players = {ln.strip() for ln in f if ln.strip()}
        print(f"Using player set from {args.players_file}: "
              f"{len(cal_players):,} players")
    else:
        print(f"Loading panel {args.panel} to reproduce seed=42 cal slice")
        with np.load(args.panel, allow_pickle=True) as d:
            pids = d["pids"].tolist()
        unique_players = sorted(set(pids))
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(unique_players))
        n_cal = int(round(args.cal_frac * len(unique_players)))
        cal_players = {unique_players[i] for i in perm[:n_cal]}
        print(f"  cal slice: {len(cal_players):,} players "
              f"(panel total {len(unique_players):,})")

    print(f"Loading hazards from {args.model}")
    hazards = load_hazards(args.model)
    event_keys = [k for k in hazards
                  if k in (CareerEvent.TOP_100_PROSPECT,
                           CareerEvent.MLB_DEBUT,
                           CareerEvent.ESTABLISHED_MLB)
                  or k == STAR_KEY or k == ELITE_KEY]

    db = ProspectDB(args.db)
    with db._connect() as conn:
        # Filter to target players upfront to keep memory low (was loading
        # all 47k prospects + 237k stats, which pandas 3.0 / numpy 2.4 can't
        # comfortably hold alongside the prediction working set).
        cp_list = list(cal_players)
        qmark = ",".join("?" * len(cp_list))
        rows = [dict(r) for r in conn.execute(f"""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE p.player_id IN ({qmark})
        """, cp_list).fetchall()]
        stats_rows = conn.execute(
            f"SELECT * FROM season_stats WHERE player_id IN ({qmark})",
            cp_list).fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    # Filter cohort: in cal_players AND entry_year <= max_entry_year
    cohort = []
    for r in rows:
        if r["player_id"] not in cal_players:
            continue
        ent = _entry_year(r, stats_by_pid)
        if ent is None or ent > args.max_entry_year:
            continue
        r["_entry_year"] = ent
        r["_bucket"] = _bucket_of(r)
        cohort.append(r)
    print(f"  filtered to entry<={args.max_entry_year}: "
          f"{len(cohort):,} players")

    # Build (snap_year, list of players) groups; skip already-debuted by snap.
    snap_groups: dict[int, list[tuple[dict, int]]] = {}
    for r in cohort:
        ent = r["_entry_year"]
        debut = r.get("mlb_debut_year")
        for off in range(0, args.max_offset + 1):
            snap = ent + off
            if snap > args.observe_through:
                break
            if debut is not None and debut <= snap:
                continue
            snap_groups.setdefault(snap, []).append((r, off))

    out_rows = []
    print(f"Scoring {sum(len(g) for g in snap_groups.values()):,} "
          f"(player, snap) pairs")
    # Inner batch cap keeps predict_cumulative_batch's intermediate arrays
    # bounded regardless of how many players are in a snap year. Set high
    # enough to amortize call overhead, low enough that peak RSS per call
    # stays well under a GB even on 9-event hazards x 15-yr horizon.
    INNER_BATCH = int(os.environ.get("SCORE_INNER_BATCH", "100"))
    import gc as _gc
    for snap, group_pairs in sorted(snap_groups.items()):
        # Score this snap's players in INNER_BATCH chunks so memory stays flat
        for batch_start in range(0, len(group_pairs), INNER_BATCH):
            batch_pairs = group_pairs[batch_start:batch_start + INNER_BATCH]
            group = [r for r, _ in batch_pairs]
            sub_stats = {r["player_id"]: [s for s in stats_by_pid.get(r["player_id"], [])
                                           if (s.get("season_year") or 0) <= snap]
                         for r, _ in batch_pairs}
            cumP = predict_cumulative_batch(
                hazards, group, sub_stats,
                current_year=snap, horizon=args.horizon,
            )
            for i, (r, offset) in enumerate(batch_pairs):
                row = {
                    "player_id": r["player_id"],
                    "name": r.get("name"),
                    "draft_year": r.get("draft_year"),
                    "draft_round": r.get("draft_round"),
                    "is_international": int(r.get("is_international") or 0),
                    "bucket": r["_bucket"],
                    "entry_year": r["_entry_year"],
                    "snap_year": snap,
                    "snap_offset": offset,
                    "years_fwd": args.observe_through - snap,
                    "mlb_debut_year": r.get("mlb_debut_year"),
                }
                # RAW probs (skip Beta calibrators)
                per_event = {}
                for e in event_keys:
                    ename = e.name if hasattr(e, "name") else str(e).lstrip("_")
                    raw_arr = cumP.get(("raw", e))
                    p_raw = float(raw_arr[i]) if raw_arr is not None else float(cumP[e][i])
                    trig = _trigger_year(r, e)
                    eligible = int(trig is None or trig > snap)
                    realized = int(trig is not None and trig > snap
                                   and trig <= args.observe_through)
                    per_event[ename] = (p_raw, trig, eligible, realized)
                    row[f"p_{ename}"] = p_raw  # RAW stored as p_<event>
                    row[f"eligible_{ename}"] = eligible
                    row[f"realized_{ename}"] = realized
                    row[f"trigger_{ename}"] = trig
                # STAR_PLUS_ELITE union
                if "STAR" in per_event and "ELITE" in per_event:
                    ps, ts, _, _ = per_event["STAR"]
                    pe, te, _, _ = per_event["ELITE"]
                    p_u = 1.0 - (1.0 - ps) * (1.0 - pe)
                    trigs = [t for t in (ts, te) if t is not None]
                    trig_u = min(trigs) if trigs else None
                    elig_u = int(trig_u is None or trig_u > snap)
                    real_u = int(trig_u is not None and trig_u > snap
                                 and trig_u <= args.observe_through)
                    row["p_STAR_PLUS_ELITE"] = p_u
                    row["eligible_STAR_PLUS_ELITE"] = elig_u
                    row["realized_STAR_PLUS_ELITE"] = real_u
                    row["trigger_STAR_PLUS_ELITE"] = trig_u
                out_rows.append(row)
            del cumP, group, sub_stats
            _gc.collect()

    print(f"Writing {len(out_rows):,} rows to {args.out}")
    fieldnames = list(out_rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print("Done.")


if __name__ == "__main__":
    main()
