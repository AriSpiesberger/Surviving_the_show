"""
prospects/classifier/grading/grade_all.py
===================================

Grade every prospect who is NOT yet in MLB at the end of `--as-of-year`
(default 2025). Output CSV with per-event probabilities, composite score,
letter grade, and partial-season flags.

Eligibility rule: include a prospect iff
  career_outcomes.mlb_debut_year IS NULL
  OR  mlb_debut_year > as_of_year

This is the actionable set for card-EV: active prospects still in the
minors (or not yet anywhere). Already-debuted players are scored elsewhere
(retired / active MLBers).

Usage:
    python -m prospects.classifier.grade_all \\
        [--db prospects.db] \\
        [--model models/event_classifiers_v0.5_pre2021_milb.pkl] \\
        [--as-of-year 2025] \\
        [--out prospect_grades_2025.csv]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from prospects.classifier.model import load_models
from prospects.classifier.score_recent import (
    _latest_milb_year, _latest_season_completeness,
)
from prospects.features.windowed import build_windowed_features
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


COMPOSITE_WEIGHTS = {
    CareerEvent.MLB_DEBUT: 1.0,
    CareerEvent.ESTABLISHED_MLB: 3.0,
    CareerEvent.ALL_STAR_ONCE: 8.0,
    CareerEvent.ALL_STAR_THREE_PLUS: 12.0,
    CareerEvent.MAJOR_AWARD: 20.0,
    CareerEvent.HOF_TRAJECTORY: 40.0,
}


def _letter_grade(score: float, percentile: float) -> str:
    """Letter grade based on cohort percentile of composite score."""
    if percentile >= 0.99: return "A+"
    if percentile >= 0.95: return "A"
    if percentile >= 0.85: return "A-"
    if percentile >= 0.70: return "B+"
    if percentile >= 0.50: return "B"
    if percentile >= 0.30: return "B-"
    if percentile >= 0.15: return "C+"
    if percentile >= 0.05: return "C"
    return "D"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--model", default="models/event_classifiers_v0.5_pre2021_milb.pkl")
    parser.add_argument("--as-of-year", type=int, default=2025,
                        help="Players not in MLB by end of this year are graded.")
    parser.add_argument("--out", default="prospect_grades_2025.csv")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading model: {args.model}")
    models = load_models(args.model)

    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_all_star_once
            FROM prospects p
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (o.mlb_debut_year IS NULL OR o.mlb_debut_year > ?)
            ORDER BY p.draft_year DESC, p.draft_round, p.draft_pick
        """, (args.as_of_year,)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()

    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    print(f"Grading {len(prospects):,} prospects not in MLB by end of {args.as_of_year}")
    print(f"  feature window cap: as_of <= {args.as_of_year}")

    rows = []
    n_with_stats = 0
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        as_of = _latest_milb_year(
            stats,
            fallback=p.get("draft_year") or args.as_of_year,
            max_year=args.as_of_year,
        )
        x = build_windowed_features(p, stats, as_of, milb_only=True).reshape(1, -1)
        meta = _latest_season_completeness(stats, as_of)
        has_stats = any((s.get("level") or "").upper() != "MLB" for s in stats)
        if has_stats:
            n_with_stats += 1

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
            "birth_date": p.get("birth_date"),
            "as_of_year": as_of,
            "has_milb_data": int(has_stats),
            **meta,
        }
        for event in CareerEvent.all_events():
            clf = models[event]
            P = clf.predict_proba(x)
            out[f"p_{event.name}"] = round(float(P.mean()), 4)
            out[f"p_{event.name}_lo"] = round(float(np.percentile(P, 10)), 4)
            out[f"p_{event.name}_hi"] = round(float(np.percentile(P, 90)), 4)

        out["composite_score"] = round(sum(
            out[f"p_{e.name}"] * w for e, w in COMPOSITE_WEIGHTS.items()
        ), 3)
        rows.append(out)

    # Letter grades by composite-score percentile within this cohort
    scores = np.array([r["composite_score"] for r in rows])
    rank = scores.argsort()
    pct = np.empty_like(scores, dtype=np.float64)
    pct[rank] = np.linspace(0.0, 1.0, len(scores))
    for r, pc, s in zip(rows, pct, scores):
        r["percentile"] = round(float(pc), 4)
        r["grade"] = _letter_grade(s, pc)

    rows.sort(key=lambda r: r["composite_score"], reverse=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    grade_counts: dict[str, int] = {}
    for r in rows:
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1

    print(f"\nWrote {len(rows):,} rows to {args.out}")
    print(f"  with any MiLB data: {n_with_stats:,}  ({n_with_stats/len(rows):.1%})")
    print()
    print("Grade distribution:")
    for g in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D"]:
        n = grade_counts.get(g, 0)
        bar = "#" * int(60 * n / max(grade_counts.values()))
        print(f"  {g:<3} {n:>5,d}  {bar}")
    print()
    print("Top 20 graded prospects:")
    print(f"{'Rank':>4} {'Grade':<5} {'Score':>6}  {'P(MLB)':>6} {'P(Est)':>6} "
          f"{'P(AS1)':>6}  {'Player':<28} {'Yr':>4}  Pick")
    print("-" * 110)
    for i, r in enumerate(rows[:20], 1):
        pick_str = f"R{r['draft_round']}.{r['draft_pick']}" if r['draft_round'] else "—"
        print(f"{i:>4} {r['grade']:<5} {r['composite_score']:>6.2f}  "
              f"{r['p_MLB_DEBUT']:>6.3f} {r['p_ESTABLISHED_MLB']:>6.3f} "
              f"{r['p_ALL_STAR_ONCE']:>6.3f}  "
              f"{r['name'][:28]:<28} {r['draft_year']:>4}  {pick_str}")


if __name__ == "__main__":
    main()
