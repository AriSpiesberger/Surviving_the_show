"""Score a held-out player slice with the calibrated v1.15 hazards.

Emits both calibrated (`p_<event>`) and raw (`p_<event>_raw`) cumulative
probabilities per (player, snap_offset). Output schema matches the existing
*_long.csv consumers (lasso_composite, Model B, validation suite).

Usage:
    python -m prospects.classifier.score_cal_slice \\
        --model models/event_classifiers_v1.15_calibrated.pkl \\
        --panel panel_v1.15.npz \\
        --players-file models/event_classifiers_v1.15_lasso_fit_players.txt \\
        --out v1.15_fit_long.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, _BetaCalibrator, _PlattCalibrator,
    _trigger_year, load_hazards, predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator
sys.modules["__main__"]._BetaCalibrator = _BetaCalibrator


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
    ap.add_argument("--model", required=True,
                    help="Calibrated hazards pickle")
    ap.add_argument("--panel", default="panel_v1.15.npz")
    ap.add_argument("--players-file", required=True,
                    help="Player ids to score (one per line)")
    ap.add_argument("--max-entry-year", type=int, default=2020,
                    help="Right-censor: only score players with entry "
                         "year <= this (mature outcomes)")
    ap.add_argument("--observe-through", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--max-offset", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.players_file) as f:
        target_players = {ln.strip() for ln in f if ln.strip()}
    print(f"Loaded {len(target_players):,} player ids from "
          f"{args.players_file}")

    print(f"Loading hazards from {args.model}")
    hazards = load_hazards(args.model)
    have_cal = any(hazards[e].get("calibrator") is not None for e in hazards)
    print(f"  calibrators present: {have_cal}")

    event_keys = [k for k in hazards
                  if k in (CareerEvent.TOP_100_PROSPECT,
                           CareerEvent.MLB_DEBUT,
                           CareerEvent.ESTABLISHED_MLB)
                  or k == STAR_KEY or k == ELITE_KEY]

    db = ProspectDB(args.db)
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    cohort = []
    for r in rows:
        if r["player_id"] not in target_players:
            continue
        ent = _entry_year(r, stats_by_pid)
        if ent is None or ent > args.max_entry_year:
            continue
        r["_entry_year"] = ent
        r["_bucket"] = _bucket_of(r)
        cohort.append(r)
    print(f"  filtered to entry<={args.max_entry_year}: "
          f"{len(cohort):,} players")

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
    n_pairs = sum(len(g) for g in snap_groups.values())
    print(f"Scoring {n_pairs:,} (player, snap) pairs")
    for snap, group_pairs in sorted(snap_groups.items()):
        group = [r for r, _ in group_pairs]
        sub_stats = {r["player_id"]: [s for s in stats_by_pid.get(r["player_id"], [])
                                       if (s.get("season_year") or 0) <= snap]
                     for r, _ in group_pairs}
        cumP = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=snap, horizon=args.horizon,
        )
        for i, (r, offset) in enumerate(group_pairs):
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
            per_event = {}
            for e in event_keys:
                ename = e.name if hasattr(e, "name") else str(e).lstrip("_")
                cal_arr = cumP[e]
                raw_arr = cumP.get(("raw", e), cal_arr)
                p_cal = float(cal_arr[i])
                p_raw = float(raw_arr[i])
                trig = _trigger_year(r, e)
                eligible = int(trig is None or trig > snap)
                realized = int(trig is not None and trig > snap
                               and trig <= args.observe_through)
                per_event[ename] = (p_cal, p_raw, trig, eligible, realized)
                row[f"p_{ename}"] = p_cal
                row[f"p_{ename}_raw"] = p_raw
                row[f"eligible_{ename}"] = eligible
                row[f"realized_{ename}"] = realized
                row[f"trigger_{ename}"] = trig
            if "STAR" in per_event and "ELITE" in per_event:
                ps_c, ps_r, ts, _, _ = per_event["STAR"]
                pe_c, pe_r, te, _, _ = per_event["ELITE"]
                p_u_c = 1.0 - (1.0 - ps_c) * (1.0 - pe_c)
                p_u_r = 1.0 - (1.0 - ps_r) * (1.0 - pe_r)
                trigs = [t for t in (ts, te) if t is not None]
                trig_u = min(trigs) if trigs else None
                elig_u = int(trig_u is None or trig_u > snap)
                real_u = int(trig_u is not None and trig_u > snap
                             and trig_u <= args.observe_through)
                row["p_STAR_PLUS_ELITE"] = p_u_c
                row["p_STAR_PLUS_ELITE_raw"] = p_u_r
                row["eligible_STAR_PLUS_ELITE"] = elig_u
                row["realized_STAR_PLUS_ELITE"] = real_u
                row["trigger_STAR_PLUS_ELITE"] = trig_u
            out_rows.append(row)

    print(f"Writing {len(out_rows):,} rows to {args.out}")
    fieldnames = list(out_rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print("Done.")


if __name__ == "__main__":
    main()
