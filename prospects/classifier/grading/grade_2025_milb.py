"""
prospects/classifier/grading/grade_2025_milb.py
=========================================

Grade every player who appeared in MiLB during the 2025 season.

Prediction set:
    All prospects with at least one season_stats row where
    season_year = 2025 AND level != 'MLB', and who have NOT yet debuted
    in MLB by end of 2025.

Training set used (already enforced upstream by v0.6 train script):
    Players drafted 2005-2020 with MiLB-only features and future-only loss.

as_of_year for inference = 2025 (their most recent MiLB snapshot).

Usage:
    python -m prospects.classifier.grade_2025_milb \\
        [--model models/event_classifiers_v0.6_extended.pkl] \\
        [--out grades_milb_2025.csv]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from prospects.classifier.model import load_models
from prospects.classifier.score_recent import _latest_season_completeness
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


def _letter_grade(pct: float) -> str:
    if pct >= 0.99: return "A+"
    if pct >= 0.95: return "A"
    if pct >= 0.85: return "A-"
    if pct >= 0.70: return "B+"
    if pct >= 0.50: return "B"
    if pct >= 0.30: return "B-"
    if pct >= 0.15: return "C+"
    if pct >= 0.05: return "C"
    return "D"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--model", default="models/event_classifiers_v0.6_extended.pkl")
    parser.add_argument("--season", type=int, default=2025,
                        help="Filter prospects to those who played MiLB in this season")
    parser.add_argument("--out", default="grades_milb_2025.csv")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading model: {args.model}")
    models = load_models(args.model)

    with db._connect() as conn:
        rows = conn.execute("""
            SELECT DISTINCT p.player_id
            FROM prospects p
            JOIN season_stats s ON s.player_id = p.player_id
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE s.season_year = ? AND s.level != 'MLB'
              AND (o.mlb_debut_year IS NULL OR o.mlb_debut_year > ?)
        """, (args.season, args.season)).fetchall()
        eligible_ids = [r["player_id"] for r in rows]
        print(f"Eligible players (MiLB in {args.season}, not yet MLB): "
              f"{len(eligible_ids):,}")

        # Load prospect rows
        placeholders = ",".join("?" * len(eligible_ids))
        prospects = [dict(r) for r in conn.execute(
            f"""SELECT p.*, o.mlb_debut_year FROM prospects p
                LEFT JOIN career_outcomes o ON o.player_id = p.player_id
                WHERE p.player_id IN ({placeholders})
                ORDER BY p.draft_year, p.draft_round, p.draft_pick""",
            eligible_ids,
        ).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()

    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    print(f"Scoring {len(prospects):,} prospects with as_of_year = {args.season}")
    rows_out = []
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        as_of = args.season
        x = build_windowed_features(p, stats, as_of, milb_only=True).reshape(1, -1)
        meta = _latest_season_completeness(stats, as_of)

        out = {
            "player_id": p["player_id"],
            "mlbam_id": p.get("mlbam_id"),
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
        rows_out.append(out)

    scores = np.array([r["composite_score"] for r in rows_out])
    rank = scores.argsort()
    pct = np.empty_like(scores, dtype=np.float64)
    pct[rank] = np.linspace(0.0, 1.0, len(scores))
    for r, pc in zip(rows_out, pct):
        r["percentile"] = round(float(pc), 4)
        r["grade"] = _letter_grade(pc)

    rows_out.sort(key=lambda r: r["composite_score"], reverse=True)

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nWrote {len(rows_out):,} rows to {args.out}")

    grade_counts: dict[str, int] = {}
    for r in rows_out:
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1

    print()
    print("Grade distribution:")
    for g in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D"]:
        n = grade_counts.get(g, 0)
        bar = "#" * int(60 * n / max(grade_counts.values()))
        print(f"  {g:<3} {n:>5,d}  {bar}")
    print()
    print("Top 25:")
    print(f"{'Rank':>4} {'Grade':<5} {'Score':>6}  {'P(MLB)':>6} {'P(Est)':>6} "
          f"{'P(AS1)':>6}  {'Player':<28} {'Yr':>4}  Pick")
    print("-" * 110)
    for i, r in enumerate(rows_out[:25], 1):
        pick = f"R{r['draft_round']}.{r['draft_pick']}" if r['draft_round'] else "—"
        print(f"{i:>4} {r['grade']:<5} {r['composite_score']:>6.2f}  "
              f"{r['p_MLB_DEBUT']:>6.3f} {r['p_ESTABLISHED_MLB']:>6.3f} "
              f"{r['p_ALL_STAR_ONCE']:>6.3f}  "
              f"{r['name'][:28]:<28} {r['draft_year']:>4}  {pick}")


if __name__ == "__main__":
    main()
