"""Dump per-year hazard distributions for specific players.

The cumulative-probability calculation is well-calibrated but the timing
predictions are anti-correlated with realized times. The diagnostic
question is: what shape do the per-year hazards have, and do they
respond to the inference-time feature aging?

Usage:
    python -m prospects.classifier.hazard_shape \\
        --model models/event_classifiers_v1.7_bigcal.pkl \\
        --players "Walker Jenkins" "Leo De Vries" "Konnor Griffin"
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from prospects.classifier.architectures.survival import (
    EXIT_KEY, _PlattCalibrator, _trigger_year, build_windowed_features,
    load_hazards,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator


EVENTS_TO_DUMP = [
    CareerEvent.MLB_DEBUT,
    CareerEvent.ESTABLISHED_MLB,
]


def dump_player(hazards: dict, prospect: dict, stats: list, current_year: int,
                horizon: int = 15):
    """Run the inference chain one year at a time and dump hazards."""
    yip_i = FEATURE_NAMES.index("years_in_pro")
    age_i = FEATURE_NAMES.index("age_at_as_of")
    yics_i = FEATURE_NAMES.index("years_in_current_system")

    X0 = build_windowed_features(prospect, stats, current_year,
                                 milb_only=True).reshape(1, -1)
    yip0 = X0[0, yip_i]
    age0 = X0[0, age_i]
    yics0 = X0[0, yics_i]

    has_exit = EXIT_KEY in hazards
    exit_clf = hazards[EXIT_KEY]["hazard"] if has_exit else None

    print(f"\n{'='*92}")
    print(f"Player: {prospect.get('name')}  "
          f"(id={prospect.get('player_id')})")
    print(f"  cur_level snapshot: yip0={yip0}  age0={age0}  yics0={yics0}")
    print(f"  position={prospect.get('primary_position')}  "
          f"is_pitcher={prospect.get('is_pitcher')}  "
          f"is_intl={prospect.get('is_international')}  "
          f"draft_year={prospect.get('draft_year')}")
    # Show the most recent stats row
    if stats:
        latest = max(stats, key=lambda s: (s.get("season_year") or 0,
                                           s.get("level") or ""))
        print(f"  latest stats: year={latest.get('season_year')} "
              f"lvl={latest.get('level')} pa={latest.get('pa')} "
              f"avg={latest.get('avg')} obp={latest.get('obp')} "
              f"ip={latest.get('ip')}")

    # Per-step hazards for each event of interest
    hazard_track: dict = {e: [] for e in EVENTS_TO_DUMP}
    surv_track: dict = {e: [1.0] for e in EVENTS_TO_DUMP}
    step_p_track: dict = {e: [] for e in EVENTS_TO_DUMP}
    in_baseball = 1.0
    exit_track: list[float] = []

    for step in range(horizon):
        X = X0.copy()
        if not np.isnan(yip0):
            X[0, yip_i] = yip0 + (step + 1)
        if not np.isnan(age0):
            X[0, age_i] = age0 + (step + 1)
        if not np.isnan(yics0):
            X[0, yics_i] = yics0 + (step + 1)

        for e in EVENTS_TO_DUMP:
            h = float(hazards[e]["hazard"].predict_proba(X)[0, 1])
            hazard_track[e].append(h)
            surv_prev = surv_track[e][-1]
            step_p = surv_prev * in_baseball * h
            step_p_track[e].append(step_p)
            surv_track[e].append(surv_prev * (1.0 - in_baseball * h))

        if has_exit:
            h_exit = float(exit_clf.predict_proba(X)[0, 1])
            exit_track.append(h_exit)
            in_baseball = in_baseball * (1.0 - h_exit)
        else:
            exit_track.append(0.0)

    # Print table
    print(f"\n{'t':>3} {'h_DEBUT':>9} {'h_EST':>9} "
          f"{'surv_DEBUT':>11} {'surv_EST':>10} "
          f"{'in_bb':>7} {'h_exit':>8} "
          f"{'step_p_D':>10} {'step_p_E':>10}")
    for t in range(horizon):
        print(f"{t+1:>3} "
              f"{hazard_track[CareerEvent.MLB_DEBUT][t]:>9.4f} "
              f"{hazard_track[CareerEvent.ESTABLISHED_MLB][t]:>9.4f} "
              f"{surv_track[CareerEvent.MLB_DEBUT][t]:>11.4f} "
              f"{surv_track[CareerEvent.ESTABLISHED_MLB][t]:>10.4f} "
              f"{(1 - sum(exit_track[:t])):>7.4f} "
              f"{exit_track[t]:>8.4f} "
              f"{step_p_track[CareerEvent.MLB_DEBUT][t]:>10.4f} "
              f"{step_p_track[CareerEvent.ESTABLISHED_MLB][t]:>10.4f}")

    cum_d = 1.0 - surv_track[CareerEvent.MLB_DEBUT][-1]
    cum_e = 1.0 - surv_track[CareerEvent.ESTABLISHED_MLB][-1]
    print(f"  cumulative P(DEBUT by yr {horizon}) = {cum_d:.3f}")
    print(f"  cumulative P(EST   by yr {horizon}) = {cum_e:.3f}")

    sp_d = step_p_track[CareerEvent.MLB_DEBUT]
    sp_e = step_p_track[CareerEvent.ESTABLISHED_MLB]
    total_d = sum(sp_d) or 1e-9
    total_e = sum(sp_e) or 1e-9
    mean_t_d = sum((t + 1) * sp_d[t] for t in range(horizon)) / total_d
    mean_t_e = sum((t + 1) * sp_e[t] for t in range(horizon)) / total_e
    print(f"  E[t_DEBUT | event] (unconditional formula) = {mean_t_d:.2f}")
    print(f"  E[t_EST   | event] (unconditional formula) = {mean_t_e:.2f}")
    # Also show what fraction of step_p mass is in each year
    print(f"  DEBUT step_p mass by year: "
          + ", ".join(f"y{t+1}={sp_d[t]/total_d*100:.1f}%"
                       for t in range(min(horizon, 8))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="models/event_classifiers_v1.7_bigcal.pkl")
    ap.add_argument("--db", default="prospects.db")
    ap.add_argument("--players", nargs="+",
                    default=["Walker Jenkins", "Leo De Vries",
                             "Konnor Griffin", "Sebastian Walcott"])
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()

    print(f"Loading model {args.model}")
    hazards = load_hazards(args.model)

    db = ProspectDB(args.db)
    with db._connect() as conn:
        for name in args.players:
            row = conn.execute(
                "SELECT * FROM prospects WHERE name = ? LIMIT 1", (name,)
            ).fetchone()
            if row is None:
                print(f"\nNot found: {name}")
                continue
            prospect = dict(row)
            stats = [dict(r) for r in conn.execute(
                "SELECT * FROM season_stats WHERE player_id=? "
                "ORDER BY season_year, level",
                (prospect["player_id"],),
            ).fetchall()]
            dump_player(hazards, prospect, stats, current_year=args.season)


if __name__ == "__main__":
    main()
