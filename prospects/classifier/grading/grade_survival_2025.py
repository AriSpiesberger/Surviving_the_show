"""
Score every 2025-MiLB-active, not-yet-MLB prospect with the v1.0 survival
hazard model. Outputs cumulative P(event by 15-year horizon).
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from prospects.classifier.score_recent import _latest_season_completeness
from prospects.classifier.architectures.survival import load_hazards, predict_cumulative_batch
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
                        default="models/event_classifiers_v1.0_survival.pkl")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--out", default="grades_survival_2025.csv")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading hazards: {args.model}")
    hazards = load_hazards(args.model)
    print(f"  events: {[e.name if hasattr(e, 'name') else str(e) for e in hazards]}")

    with db._connect() as conn:
        eligible = [r["player_id"] for r in conn.execute("""
            SELECT DISTINCT p.player_id FROM prospects p
            JOIN season_stats s ON s.player_id = p.player_id
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE s.season_year = ? AND s.level != 'MLB'
              AND (o.mlb_debut_year IS NULL OR o.mlb_debut_year > ?)
        """, (args.season, args.season)).fetchall()]
        print(f"Eligible: {len(eligible):,} prospects active in {args.season} MiLB")
        ph = ",".join("?" * len(eligible))
        prospects = [dict(r) for r in conn.execute(
            f"""SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                       o.year_all_star_once, o.year_all_star_three,
                       o.year_major_award, o.year_hof_trajectory
                FROM prospects p
                LEFT JOIN career_outcomes o ON o.player_id = p.player_id
                WHERE p.player_id IN ({ph})
                ORDER BY p.draft_year, p.draft_round, p.draft_pick""",
            eligible,
        ).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    print(f"Scoring with current_year = {args.season}, horizon = {args.horizon} yrs")
    cumP = predict_cumulative_batch(
        hazards, prospects, stats_by_pid,
        current_year=args.season, horizon=args.horizon,
    )

    out_rows = []
    for i, p in enumerate(prospects):
        stats = stats_by_pid.get(p["player_id"], [])
        meta = _latest_season_completeness(stats, args.season)
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
            "is_international": int(p.get("is_international") or 0),
            "birth_date": p.get("birth_date"),
            "as_of_year": args.season,
            "horizon": args.horizon,
            **meta,
        }
        for event in CareerEvent.all_events():
            if event in cumP:
                out[f"p_{event.name}"] = round(float(cumP[event][i]), 4)
                raw_key = ("raw", event)
                if raw_key in cumP:
                    out[f"p_{event.name}_raw"] = round(float(cumP[raw_key][i]), 4)
                else:
                    out[f"p_{event.name}_raw"] = out[f"p_{event.name}"]
                # E[T] and SD[T] in years from as_of_year (conditional on event)
                mt_key = ("mean_t", event)
                sd_key = ("sd_t", event)
                if mt_key in cumP:
                    mt = float(cumP[mt_key][i])
                    sd = float(cumP[sd_key][i])
                    out[f"t_{event.name}_mean"] = round(mt, 2) if mt == mt else ""  # NaN -> ""
                    out[f"t_{event.name}_sd"] = round(sd, 2) if sd == sd else ""
                else:
                    out[f"t_{event.name}_mean"] = ""
                    out[f"t_{event.name}_sd"] = ""
            else:
                out[f"p_{event.name}"] = 0.0
                out[f"p_{event.name}_raw"] = 0.0
                out[f"t_{event.name}_mean"] = ""
                out[f"t_{event.name}_sd"] = ""
        # Enforce monotonicity on BOTH calibrated and raw.
        chain = [
            (CareerEvent.MLB_DEBUT, CareerEvent.ESTABLISHED_MLB),
            (CareerEvent.MLB_DEBUT, CareerEvent.ALL_STAR_ONCE),
            (CareerEvent.ESTABLISHED_MLB, CareerEvent.ALL_STAR_ONCE),
            (CareerEvent.ALL_STAR_ONCE, CareerEvent.ALL_STAR_THREE_PLUS),
            (CareerEvent.ALL_STAR_ONCE, CareerEvent.MAJOR_AWARD),
        ]
        for broader, narrower in chain:
            for suffix in ("", "_raw"):
                kn = f"p_{narrower.name}{suffix}"
                kb = f"p_{broader.name}{suffix}"
                out[kn] = round(min(out.get(kn, 0.0), out.get(kb, 1.0)), 4)
        # composite_score: calibrated probabilities (honest EV).
        # composite_score_raw: rank-preserving raw probabilities (use for sorting
        # within cohort; isotonic flattens fine distinctions at the top tier).
        out["composite_score"] = round(sum(
            out[f"p_{e.name}"] * w for e, w in COMPOSITE_WEIGHTS.items()
        ), 3)
        out["composite_score_raw"] = round(sum(
            out[f"p_{e.name}_raw"] * w for e, w in COMPOSITE_WEIGHTS.items()
        ), 3)
        out_rows.append(out)

    # Rank by RAW (preserves elite-tier ordering); percentile/grade follow RAW.
    scores = np.array([r["composite_score_raw"] for r in out_rows])
    rank = scores.argsort()
    pct = np.empty_like(scores, dtype=np.float64)
    pct[rank] = np.linspace(0.0, 1.0, len(scores))
    for r, pc in zip(out_rows, pct):
        r["percentile"] = round(float(pc), 4)
        r["grade"] = _letter_grade(pc)
    out_rows.sort(key=lambda r: r["composite_score_raw"], reverse=True)

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
        yr = r['draft_year'] if r['draft_year'] is not None else "IFA"
        print(f"{i:>3} {r['grade']:<3} {r['composite_score']:>6.2f}  "
              f"{r['p_MLB_DEBUT']:>6.3f} {r['p_ESTABLISHED_MLB']:>6.3f} "
              f"{r['p_ALL_STAR_ONCE']:>6.3f} {r['p_ALL_STAR_THREE_PLUS']:>8.3f}  "
              f"{r['name'][:28]:<28} {str(yr):>4} {pick}")


if __name__ == "__main__":
    main()
