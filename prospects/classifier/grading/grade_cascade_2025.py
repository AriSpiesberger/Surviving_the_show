"""
prospects/classifier/grading/grade_cascade_2025.py
============================================

Score every 2025-MiLB-active, not-yet-MLB prospect with the v0.9.2 cascaded
stack and write a CSV.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from prospects.classifier.architectures.cascaded import load_cascade, predict_cascade
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
    parser.add_argument("--db", default="prospects_snapshot.db")
    parser.add_argument("--model",
                        default="models/event_classifiers_v0.9.2_cascaded.pkl")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--out", default="grades_cascade_2025.csv")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading cascade: {args.model}")
    cascade = load_cascade(args.model)

    with db._connect() as conn:
        eligible = [r["player_id"] for r in conn.execute("""
            SELECT DISTINCT p.player_id FROM prospects p
            JOIN season_stats s ON s.player_id = p.player_id
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE s.season_year = ? AND s.level != 'MLB'
              AND (o.mlb_debut_year IS NULL OR o.mlb_debut_year > ?)
        """, (args.season, args.season)).fetchall()]
        print(f"Eligible: {len(eligible):,} prospects in {args.season} MiLB, not yet MLB")
        ph = ",".join("?" * len(eligible))
        prospects = [dict(r) for r in conn.execute(
            f"SELECT p.* FROM prospects p WHERE p.player_id IN ({ph}) "
            f"ORDER BY p.draft_year, p.draft_round, p.draft_pick",
            eligible,
        ).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    print(f"Scoring with as_of_year = {args.season}")
    X_list = []
    meta_list = []
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        x = build_windowed_features(p, stats, args.season, milb_only=True)
        X_list.append(x)
        meta_list.append((p, _latest_season_completeness(stats, args.season)))
    X = np.vstack(X_list)

    all_preds = predict_cascade(cascade, X)

    out_rows = []
    for i, (p, meta) in enumerate(meta_list):
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
            "as_of_year": args.season,
            **meta,
        }
        for event in CareerEvent.all_events():
            pred_arr = all_preds.get(event)
            out[f"p_{event.name}"] = (
                round(float(pred_arr[i]), 4) if pred_arr is not None else 0.0
            )
        out["composite_score"] = round(sum(
            out[f"p_{e.name}"] * w for e, w in COMPOSITE_WEIGHTS.items()
        ), 3)
        out_rows.append(out)

    scores = np.array([r["composite_score"] for r in out_rows])
    rank = scores.argsort()
    pct = np.empty_like(scores, dtype=np.float64)
    pct[rank] = np.linspace(0.0, 1.0, len(scores))
    for r, pc in zip(out_rows, pct):
        r["percentile"] = round(float(pc), 4)
        r["grade"] = _letter_grade(pc)
    out_rows.sort(key=lambda r: r["composite_score"], reverse=True)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows):,} rows to {args.out}")

    print("\nTop 25:")
    print(f"{'Rnk':>3} {'Gr':<3} {'Score':>6}  {'P(MLB)':>6} {'P(Est)':>6} "
          f"{'P(AS1)':>6} {'P(AS3+)':>8}  {'Player':<28} {'Yr':>4} Pick")
    print("-" * 110)
    for i, r in enumerate(out_rows[:25], 1):
        pick = f"R{r['draft_round']}.{r['draft_pick']}" if r['draft_round'] else "—"
        print(f"{i:>3} {r['grade']:<3} {r['composite_score']:>6.2f}  "
              f"{r['p_MLB_DEBUT']:>6.3f} {r['p_ESTABLISHED_MLB']:>6.3f} "
              f"{r['p_ALL_STAR_ONCE']:>6.3f} {r['p_ALL_STAR_THREE_PLUS']:>8.3f}  "
              f"{r['name'][:28]:<28} {r['draft_year']:>4} {pick}")


if __name__ == "__main__":
    main()
