"""
prospects/classifier/score_recent.py
======================================

Score recent draftees with a saved classifier and dump predictions to CSV.

Defaults score the 2024 and 2025 draft classes with v0.5 (MiLB-only,
selection-bias-corrected). At inference time the as-of year is the latest
year we have stats for (so the most current snapshot), and features include
ONLY MiLB-level rows to match training.

Usage:
    python -m prospects.classifier.score_recent \\
        [--db prospects.db] \\
        [--model models/event_classifiers_v0.5_pre2021_milb.pkl] \\
        [--years 2024 2025] \\
        [--out scored_prospects.csv]
"""

from __future__ import annotations

import argparse
import csv
from datetime import date

import numpy as np

from prospects.classifier.model import load_models
from prospects.features.windowed import (
    FEATURE_NAMES,
    build_windowed_features,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def _latest_milb_year(stats: list[dict], fallback: int,
                      max_year: int | None = None) -> int:
    """Latest MiLB-active year, but never beyond `max_year` (last complete season).
    This prevents using in-progress current-year stats as if they were full-season."""
    yrs = [s["season_year"] for s in stats
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"
           and (max_year is None or s["season_year"] <= max_year)]
    return max(yrs) if yrs else min(fallback, max_year or fallback)


def _latest_season_completeness(stats: list[dict], as_of: int) -> dict:
    """Inspect the as_of year row(s) and return diagnostics for the CSV."""
    rows = [s for s in stats if s.get("season_year") == as_of]
    if not rows:
        return {"as_of_pa": 0, "as_of_ip": 0.0, "as_of_complete": None,
                "as_of_injury_susp": 0, "as_of_games": None}
    return {
        "as_of_pa": int(sum((r.get("pa") or 0) for r in rows)),
        "as_of_ip": round(sum((r.get("ip") or 0.0) for r in rows), 1),
        "as_of_complete": int(all(r.get("season_complete") in (1, None) for r in rows)),
        "as_of_injury_susp": int(any(r.get("injury_suspected") for r in rows)),
        "as_of_games": sum((r.get("games_played") or 0) for r in rows) or None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--model", default="models/event_classifiers_v0.5_pre2021_milb.pkl")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--out", default="scored_prospects.csv")
    parser.add_argument("--current-year", type=int, default=2026)
    parser.add_argument(
        "--max-as-of-year", type=int, default=None,
        help="Cap as-of year to this. Default = current_year - 1 "
             "(use last fully-completed season). Set to current_year to "
             "include partial in-progress stats (risky).",
    )
    args = parser.parse_args()
    max_year = args.max_as_of_year if args.max_as_of_year is not None else args.current_year - 1
    print(f"  capping as_of_year at {max_year} (last complete season)")

    db = ProspectDB(args.db)
    print(f"Loading model: {args.model}")
    models = load_models(args.model)
    print(f"  {len(models)} per-event classifiers")

    print(f"Selecting prospects from draft years {args.years}...")
    with db._connect() as conn:
        placeholders = ",".join("?" * len(args.years))
        prospects = [dict(r) for r in conn.execute(
            f"""SELECT * FROM prospects
                WHERE draft_year IN ({placeholders})
                ORDER BY draft_year, draft_round, draft_pick""",
            args.years,
        ).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    print(f"  {len(prospects)} prospects to score")

    event_names = [e.name for e in CareerEvent.all_events()]
    rows = []
    n_no_stats = 0
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        as_of = _latest_milb_year(
            stats,
            fallback=p.get("draft_year") or max_year,
            max_year=max_year,
        )
        # Match training: MiLB-only features
        x = build_windowed_features(p, stats, as_of, milb_only=True).reshape(1, -1)
        season_meta = _latest_season_completeness(stats, as_of)

        if not any(s for s in stats if (s.get("level") or "").upper() != "MLB"):
            n_no_stats += 1  # still score, but mark

        out = {
            "player_id": p["player_id"],
            "name": p["name"],
            "draft_year": p.get("draft_year"),
            "draft_round": p.get("draft_round"),
            "draft_pick": p.get("draft_pick"),
            "primary_position": p.get("primary_position"),
            "is_pitcher": int(bool(p.get("is_pitcher"))),
            "current_org": p.get("current_org"),
            "origin": p.get("origin"),
            "as_of_year": as_of,
            "has_milb_data": 0 if not any(
                (s.get("level") or "").upper() != "MLB" for s in stats
            ) else 1,
            **season_meta,
        }
        # Bootstrap-ensemble mean per event
        for event in CareerEvent.all_events():
            clf = models[event]
            P = clf.predict_proba(x)
            mean = float(P.mean())
            lo = float(np.percentile(P, 10))
            hi = float(np.percentile(P, 90))
            out[f"p_{event.name}"] = round(mean, 4)
            out[f"p_{event.name}_lo"] = round(lo, 4)
            out[f"p_{event.name}_hi"] = round(hi, 4)
        # A simple composite score: weighted sum of star-tier events
        out["score"] = round(
            out["p_MLB_DEBUT"] * 1
            + out["p_ESTABLISHED_MLB"] * 3
            + out["p_ALL_STAR_ONCE"] * 8
            + out["p_ALL_STAR_THREE_PLUS"] * 12
            + out["p_MAJOR_AWARD"] * 20,
            3,
        )
        rows.append(out)

    rows.sort(key=lambda r: r["score"], reverse=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")
    print(f"  prospects with no MiLB stats yet: {n_no_stats}")
    print(f"\nTop 25 by composite score:")
    print(f"{'Rank':>4} {'Score':>6}  {'P(MLB)':>6} {'P(Est)':>6} {'P(AS1)':>6} {'P(AS3+)':>8}  "
          f"{'Player':<28} {'Yr':>4} Pick")
    print("-" * 100)
    for i, r in enumerate(rows[:25], 1):
        pick_str = f"R{r['draft_round']}.{r['draft_pick']}" if r['draft_round'] else "—"
        print(f"{i:>4} {r['score']:>6.2f}  "
              f"{r['p_MLB_DEBUT']:>6.3f} {r['p_ESTABLISHED_MLB']:>6.3f} "
              f"{r['p_ALL_STAR_ONCE']:>6.3f} {r['p_ALL_STAR_THREE_PLUS']:>8.3f}  "
              f"{r['name'][:28]:<28} {r['draft_year']:>4} {pick_str}")


if __name__ == "__main__":
    main()
