"""Fit per-event Beta calibrators on the dedicated hazard-cal slice.

Reads a hazard-only pickle (from train_full_v15), scores the hazard-cal
players with multi-snapshot cumulative probabilities, and fits a Beta
calibrator per event against realized outcomes. Writes the same model
pickle with `hazards[e]["calibrator"]` populated so downstream consumers
(predict_cumulative_batch) automatically apply the calibration.

Usage:
    python -m prospects.classifier.fit_hazard_calibrators \\
        --model models/event_classifiers_v1.15.pkl \\
        --panel panel_v1.15.npz \\
        --players-file models/event_classifiers_v1.15_hazard_cal_players.txt \\
        --out models/event_classifiers_v1.15_calibrated.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY, EXIT_KEY, MAX_OBS_YEAR, STAR_KEY,
    _BetaCalibrator, _PlattCalibrator, _trigger_year,
    load_hazards, predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator
sys.modules["__main__"]._BetaCalibrator = _BetaCalibrator


def _ev_key(e):
    return e if isinstance(e, str) else int(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to hazard-only pickle from train_full_v15")
    ap.add_argument("--panel", required=True,
                    help="Panel npz (only used to verify schema)")
    ap.add_argument("--players-file", required=True,
                    help="Hazard-cal player list, one per line")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--method", default="beta",
                    choices=["beta", "sigmoid", "iso"],
                    help="Calibrator family")
    ap.add_argument("--observe-through", type=int, default=MAX_OBS_YEAR)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--snapshot-offsets", default="1,2,3,5",
                    help="Per-player snapshot offsets (years post-entry)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.players_file) as f:
        cal_players = {ln.strip() for ln in f if ln.strip()}
    print(f"Loaded {len(cal_players):,} hazard-cal players from "
          f"{args.players_file}")

    print(f"Loading hazards from {args.model}")
    hazards = load_hazards(args.model)
    print(f"  events present: "
          f"{[e.name if hasattr(e, 'name') else e for e in hazards]}")

    if any(hazards[e].get("calibrator") is not None for e in hazards):
        print("WARNING: model already has calibrators. Overwriting.")

    # Pull all prospect metadata + season stats
    db = ProspectDB(args.db)
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
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

    cal_rows = [r for r in rows if r["player_id"] in cal_players]
    print(f"  matched {len(cal_rows):,} cal rows in DB")

    snapshot_offsets = tuple(int(s) for s in args.snapshot_offsets.split(","))
    groups_by_year: dict[int, list[dict]] = defaultdict(list)
    for r in cal_rows:
        dy = r.get("draft_year")
        if dy is None:
            yrs = [int(s["season_year"]) for s in stats_by_pid.get(r["player_id"], [])
                   if s.get("season_year") is not None]
            if not yrs: continue
            start = min(yrs)
        else:
            start = int(dy)
        for off in snapshot_offsets:
            cur = start + off
            if cur >= args.observe_through: continue
            groups_by_year[cur].append(r)

    n_snap = sum(len(g) for g in groups_by_year.values())
    print(f"  {len(groups_by_year)} snapshot-year groups, {n_snap:,} "
          f"(player, snap) pairs")

    score_keys = [e for e in hazards if e != EXIT_KEY]
    per_event_preds: dict = {_ev_key(e): [] for e in score_keys}
    per_event_real: dict = {_ev_key(e): [] for e in score_keys}

    for cur_year, group in sorted(groups_by_year.items()):
        sub_stats = {r["player_id"]: stats_by_pid.get(r["player_id"], [])
                     for r in group}
        out = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=cur_year, horizon=args.horizon,
        )
        for i, r in enumerate(group):
            for ev in score_keys:
                k = _ev_key(ev)
                # Use the RAW cumulative — that's what the calibrator must map.
                raw_arr = out.get(("raw", ev))
                p_raw = float(raw_arr[i]) if raw_arr is not None else float(out[ev][i])
                per_event_preds[k].append(p_raw)
                trig = _trigger_year(r, ev)
                per_event_real[k].append(
                    int(trig is not None and trig <= args.observe_through)
                )

    print(f"\nFitting {args.method} calibrators per event")
    print(f"{'event':<22} {'n':>6} {'pos':>5} {'pre_top10':>10} {'post_top10':>11}")
    print("-" * 60)
    for event in score_keys:
        k = _ev_key(event)
        preds = np.asarray(per_event_preds[k], dtype=np.float64)
        real = np.asarray(per_event_real[k], dtype=np.int32)
        n_pos = int(real.sum())
        label = event.name if hasattr(event, "name") else str(event)
        if n_pos < 5 or n_pos > len(real) - 5:
            print(f"  {label:<22} {len(preds):>6,d} {n_pos:>5d}  "
                  f"skip (degenerate)")
            continue

        if args.method == "beta":
            calib = _BetaCalibrator().fit(preds, real)
        elif args.method == "sigmoid":
            calib = _PlattCalibrator().fit(preds, real)
        elif args.method == "iso":
            from sklearn.isotonic import IsotonicRegression
            calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calib.fit(preds, real)
        else:
            raise ValueError(args.method)

        top10_idx = np.argsort(preds)[::-1][:max(1, len(preds) // 10)]
        pre_top10 = float(preds[top10_idx].mean())
        post_top10 = float(np.asarray(calib.predict(preds[top10_idx])).mean())
        hazards[event]["calibrator"] = calib
        hazards[event]["calibrator_method"] = args.method
        print(f"  {label:<22} {len(preds):>6,d} {n_pos:>5d}  "
              f"{pre_top10:>9.3f}  {post_top10:>10.3f}")

    print(f"\nWriting {args.out}")
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    main()
