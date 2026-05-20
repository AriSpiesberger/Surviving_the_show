"""
prospects/classifier/grade_2026_blended.py
==========================================

Mid-season grading with empirical-Bayes shrinkage of in-progress 2026 stats.

The hazard model was trained on COMPLETE seasons; raw mid-season totals
trick it into reading partial samples as low playing-time signals. This
script pre-blends 2026 partial stats with each player's most-recent prior
year via shrinkage (see prospects.features.partial_season) so the model
gets a stable virtual-full-season view.

Two CSV outputs identical in schema to grade_2025_sheets.py:
    probs CSV  — calibrated + raw probabilities, ELITE, composite, grade
    timing CSV — E[T] / SD[T] per event

Usage:
    python -m prospects.classifier.grade_2026_blended \\
        [--season 2026] [--season-progress AUTO] \\
        [--model models/event_classifiers_v1.4_platt.pkl]
"""
from __future__ import annotations

import argparse
import csv
from datetime import date

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, load_hazards, predict_cumulative_batch,
)
from prospects.features.partial_season import apply_blender_to_stats
from prospects.features.scouting import build_scouting_features  # noqa (kept for parity)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


LEVEL_RANK = {"DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
              "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}

# v1.11: TOP_100_PROSPECT added as a hazard. It fires when a player first
# appears on the BBC top-100. From a card-market view it's a smaller
# multiplier than MLB_DEBUT but it triggers earlier and more often, and
# it's a strong leading indicator for downstream events.
COMPOSITE_WEIGHTS = {
    CareerEvent.TOP_100_PROSPECT: 0.5,
    CareerEvent.MLB_DEBUT: 1.0,
    CareerEvent.ESTABLISHED_MLB: 3.0,
}
STAR_WEIGHT = 10.0

DISPLAY_EVENTS = [
    CareerEvent.TOP_100_PROSPECT,
    CareerEvent.MLB_DEBUT,
    CareerEvent.ESTABLISHED_MLB,
]


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


def _estimate_season_progress(season: int, today: date | None = None) -> float:
    """MiLB season approximation: April 1 -> end of September (~26 weeks).
    Returns fraction elapsed for the given season as of `today`. Clamps [0,1]."""
    today = today or date.today()
    season_start = date(season, 4, 1)
    season_end = date(season, 9, 30)
    if today < season_start:
        return 0.0
    if today > season_end:
        return 1.0
    total_days = (season_end - season_start).days
    elapsed = (today - season_start).days
    return max(0.0, min(1.0, elapsed / max(total_days, 1)))


def _level_summary_for_year(stats: list[dict], season: int,
                            blended: bool = False) -> dict:
    """Top-level snapshot for the chosen season (post-blend if blended=True)."""
    yr_rows = [s for s in stats if s.get("season_year") == season
               and (s.get("level") or "").upper() != "MLB"]
    if not yr_rows:
        return {
            "cur_level": "", "cur_pa": 0, "cur_ip": 0.0,
            "cur_avg": "", "cur_obp": "", "cur_slg": "", "cur_iso": "",
            "cur_k_pct": "", "cur_bb_pct": "", "cur_woba": "",
            "cur_hr": "", "cur_sb": "",
            "cur_era": "", "cur_k9": "", "cur_bb9": "", "cur_whip": "",
            "cur_fip": "", "cur_hr9": "",
            "blended": int(blended),
        }
    best = max(yr_rows,
               key=lambda s: LEVEL_RANK.get((s.get("level") or "").upper(), 0))
    lvl = (best.get("level") or "").upper()
    rows_at_lvl = [s for s in yr_rows if (s.get("level") or "").upper() == lvl]

    def _wavg(key, weight_key):
        vals = [(s.get(key), s.get(weight_key) or 0)
                for s in rows_at_lvl
                if s.get(key) is not None and (s.get(weight_key) or 0) > 0]
        denom = sum(w for _, w in vals)
        if denom <= 0:
            return None
        return sum(v * w for v, w in vals) / denom

    def _fmt(v, p=3):
        return round(float(v), p) if v is not None else ""

    pa = sum((s.get("pa") or 0) for s in rows_at_lvl)
    ip = sum((s.get("ip") or 0.0) for s in rows_at_lvl)
    hr = sum((s.get("home_runs") or 0) for s in rows_at_lvl) if pa else None
    sb = sum((s.get("stolen_bases") or 0) for s in rows_at_lvl) if pa else None

    return {
        "cur_level": lvl,
        "cur_pa": int(pa) if isinstance(pa, (int, float)) else 0,
        "cur_ip": round(float(ip), 1),
        "cur_avg":   _fmt(_wavg("avg", "pa")),
        "cur_obp":   _fmt(_wavg("obp", "pa")),
        "cur_slg":   _fmt(_wavg("slg", "pa")),
        "cur_iso":   _fmt(_wavg("iso", "pa")),
        "cur_k_pct": _fmt(_wavg("k_pct", "pa")),
        "cur_bb_pct":_fmt(_wavg("bb_pct", "pa")),
        "cur_woba":  _fmt(_wavg("woba", "pa")),
        "cur_hr":    int(hr) if hr is not None else "",
        "cur_sb":    int(sb) if sb is not None else "",
        "cur_era":   _fmt(_wavg("era", "ip"), 2),
        "cur_k9":    _fmt(_wavg("k9", "ip"), 2),
        "cur_bb9":   _fmt(_wavg("bb9", "ip"), 2),
        "cur_whip":  _fmt(_wavg("whip", "ip"), 2),
        "cur_fip":   _fmt(_wavg("fip", "ip"), 2),
        "cur_hr9":   _fmt(_wavg("hr9", "ip"), 2),
        "blended":   int(blended),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects_snapshot.db")
    parser.add_argument("--model",
                        default="models/event_classifiers_v1.4_platt.pkl")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--season-progress", type=str, default="AUTO",
                        help="Fraction of season elapsed (0-1) or 'AUTO' to "
                             "derive from today's date")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--probs-out", default="grades_probs_2026.csv")
    parser.add_argument("--timing-out", default="grades_timing_2026.csv")
    args = parser.parse_args()

    sp = (_estimate_season_progress(args.season)
          if args.season_progress.upper() == "AUTO"
          else float(args.season_progress))
    print(f"Season {args.season}, progress estimated at {sp:.1%}")

    db = ProspectDB(args.db)
    print(f"Loading hazards: {args.model}")
    hazards = load_hazards(args.model)
    event_keys = [k for k in hazards if not isinstance(k, str)]
    print(f"  events: {[e.name for e in event_keys] + ['_ELITE']}")

    # Eligible = active in `--season` MiLB OR (for early-season runs with
    # no 2026 data yet) active in `season-1` and not yet MLB-debuted.
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT DISTINCT p.player_id FROM prospects p
            JOIN season_stats s ON s.player_id = p.player_id
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (s.season_year = ? OR s.season_year = ?)
              AND s.level != 'MLB'
              AND (o.mlb_debut_year IS NULL OR o.mlb_debut_year > ?)
        """, (args.season, args.season - 1, args.season)).fetchall()
        eligible = [r["player_id"] for r in rows]
        print(f"Eligible: {len(eligible):,} prospects "
              f"(2025 or {args.season} MiLB activity, not yet MLB)")
        ph = ",".join("?" * len(eligible))
        prospects = [dict(r) for r in conn.execute(
            f"SELECT p.*, o.mlb_debut_year, o.year_established_mlb, "
            f"o.year_top_100, o.year_top_25, "
            f"o.year_all_star_once, o.year_all_star_three, "
            f"o.year_major_award, o.year_hof_trajectory, o.final_mlb_year "
            f"FROM prospects p "
            f"LEFT JOIN career_outcomes o ON o.player_id = p.player_id "
            f"WHERE p.player_id IN ({ph}) "
            f"ORDER BY COALESCE(p.draft_year, 9999), p.draft_round, p.draft_pick",
            eligible,
        ).fetchall()]
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
    for p in prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    # Pre-blend partial-season rows for each player
    blended_stats: dict[str, list] = {}
    n_blended = 0
    for pid, stats in stats_by_pid.items():
        out = apply_blender_to_stats(stats, current_year=args.season,
                                     season_progress=sp)
        if any(s.get("_blended_partial") for s in out):
            n_blended += 1
        blended_stats[pid] = out
    print(f"Blended partial-season rows for {n_blended:,} players")

    # Score with as_of = season (now-blended)
    print(f"Scoring with current_year = {args.season} (blended)")
    cumP: dict = predict_cumulative_batch(
        hazards, prospects,
        {p["player_id"]: blended_stats.get(p["player_id"], []) for p in prospects},
        current_year=args.season, horizon=args.horizon,
    )

    # v1.5: STAR replaces the AS1+ELITE pair. Pull STAR cal+raw; fall back
    # to ELITE for older model files that don't have STAR trained yet.
    n = len(prospects)
    rare_key = STAR_KEY if STAR_KEY in cumP else ELITE_KEY
    if rare_key in cumP:
        star_cal = np.asarray(cumP[rare_key], dtype=np.float64)
        star_raw = np.asarray(cumP.get(("raw", rare_key), star_cal),
                              dtype=np.float64)
    else:
        star_cal = np.zeros(n); star_raw = np.zeros(n)
    star_mean_t = np.asarray(cumP.get(("mean_t", rare_key),
                                      np.full(n, np.nan)), dtype=np.float64)
    star_sd = np.asarray(cumP.get(("sd_t", rare_key),
                                  np.full(n, np.nan)), dtype=np.float64)

    # ---- Monotonicity in RAW space, then re-calibrate ----
    # Single chain: MLB_DEBUT >= ESTABLISHED >= STAR. The survival framework
    # requires P(narrower) <= P(broader) at the raw level. Clipping in raw
    # space and re-calibrating preserves discrimination at the top tier
    # (cal-space clipping flattens it).
    _raw_chain = [
        (CareerEvent.MLB_DEBUT, CareerEvent.ESTABLISHED_MLB),
    ]
    for broader, narrower in _raw_chain:
        if narrower not in cumP or broader not in cumP:
            continue
        raw_b = np.asarray(cumP.get(("raw", broader), cumP[broader]),
                           dtype=np.float64)
        raw_n = np.asarray(cumP.get(("raw", narrower), cumP[narrower]),
                           dtype=np.float64)
        raw_n_clipped = np.minimum(raw_n, raw_b)
        cumP[("raw", narrower)] = raw_n_clipped
        cal = hazards.get(narrower, {}).get("calibrator")
        if cal is not None:
            cumP[narrower] = np.asarray(cal.predict(raw_n_clipped),
                                        dtype=np.float64)
        else:
            cumP[narrower] = raw_n_clipped

    # STAR: clip raw against ESTABLISHED_MLB, then re-calibrate. STAR is
    # by definition a subset of ESTABLISHED (you can't be a star without
    # establishing) so the constraint is structural.
    if CareerEvent.ESTABLISHED_MLB in cumP:
        raw_est = np.asarray(cumP.get(("raw", CareerEvent.ESTABLISHED_MLB),
                                      cumP[CareerEvent.ESTABLISHED_MLB]),
                             dtype=np.float64)
        star_raw = np.minimum(star_raw, raw_est)
    star_cal_obj = hazards.get(rare_key, {}).get("calibrator")
    if star_cal_obj is not None:
        star_cal = np.asarray(star_cal_obj.predict(star_raw),
                              dtype=np.float64)
    else:
        star_cal = star_raw
    cumP[rare_key] = star_cal
    cumP[("raw", rare_key)] = star_raw

    rows_probs = []
    rows_timing = []
    for i, p in enumerate(prospects):
        pid = p["player_id"]
        stats = blended_stats.get(pid, [])
        # Use blended virtual-full-season for the level summary so the
        # output reflects what the model actually scored on.
        lvl_summary = _level_summary_for_year(stats, args.season,
                                              blended=True)
        # If no current-season data even after blending, fall back to prior
        if not lvl_summary["cur_level"]:
            lvl_summary = _level_summary_for_year(
                stats_by_pid.get(pid, []), args.season - 1, blended=False
            )

        # IFA fallback start_year
        draft_year = p.get("draft_year")
        if draft_year is None:
            milb_years = [s.get("season_year") for s in stats
                          if s.get("season_year") is not None
                          and (s.get("level") or "").upper() != "MLB"]
            start_year = min(milb_years) if milb_years else ""
        else:
            start_year = draft_year

        # As-of-aware BBC top-100 summary for this player.
        rankings = p.get("_top100_rankings") or []
        past_ranks = [(y, r) for (y, r, *_rest) in rankings
                      if y is not None and r is not None
                      and int(y) <= args.season]
        if past_ranks:
            yrs = [int(y) for y, _ in past_ranks]
            rs = [int(r) for _, r in past_ranks]
            best_rank = min(rs)
            latest_year = max(yrs)
            recent_rank = next(int(r) for y, r in past_ranks if y == latest_year)
            times = len(past_ranks)
            first_year = min(yrs)
        else:
            best_rank = None
            recent_rank = None
            times = 0
            first_year = None

        ident = {
            "player_id": pid,
            "mlbam_id": p.get("mlbam_id"),
            "name": p["name"],
            "draft_year": draft_year,
            "start_year": start_year,
            "draft_round": p.get("draft_round"),
            "draft_pick": p.get("draft_pick"),
            "is_international": int(p.get("is_international") or 0),
            "primary_position": p.get("primary_position"),
            "is_pitcher": int(bool(p.get("is_pitcher"))),
            "current_org": p.get("current_org"),
            "birth_date": p.get("birth_date"),
            "height_inches": p.get("height_inches"),
            "weight_lbs": p.get("weight_lbs"),
            "bats": p.get("bats"),
            "throws": p.get("throws"),
            "signing_bonus_usd": p.get("signing_bonus_usd"),
            "pick_value_usd": p.get("pick_value_usd"),
            "season": args.season,
            "season_progress": round(sp, 3),
            "best_top100_rank": best_rank,
            "recent_top100_rank": recent_rank,
            "times_top100": times,
            "first_top100_year": first_year,
            **lvl_summary,
        }

        # ---- Probabilities sheet ----
        prow = dict(ident)
        for e in DISPLAY_EVENTS:
            if e in cumP:
                prow[f"p_{e.name}"] = round(float(cumP[e][i]), 4)
                raw = cumP.get(("raw", e))
                prow[f"p_{e.name}_raw"] = (
                    round(float(raw[i]), 4) if raw is not None
                    else prow[f"p_{e.name}"]
                )
            else:
                prow[f"p_{e.name}"] = 0.0
                prow[f"p_{e.name}_raw"] = 0.0
        # Monotonicity enforced in raw space upstream; no post-cal clip.
        prow["p_STAR"] = round(float(star_cal[i]), 4)
        prow["p_STAR_raw"] = round(float(star_raw[i]), 4)
        prow["composite_score"] = round(
            prow.get("p_TOP_100_PROSPECT", 0.0) * COMPOSITE_WEIGHTS[CareerEvent.TOP_100_PROSPECT]
            + prow["p_MLB_DEBUT"] * COMPOSITE_WEIGHTS[CareerEvent.MLB_DEBUT]
            + prow["p_ESTABLISHED_MLB"] * COMPOSITE_WEIGHTS[CareerEvent.ESTABLISHED_MLB]
            + prow["p_STAR"] * STAR_WEIGHT,
            3,
        )
        prow["composite_score_raw"] = round(
            prow.get("p_TOP_100_PROSPECT_raw", 0.0) * COMPOSITE_WEIGHTS[CareerEvent.TOP_100_PROSPECT]
            + prow["p_MLB_DEBUT_raw"] * COMPOSITE_WEIGHTS[CareerEvent.MLB_DEBUT]
            + prow["p_ESTABLISHED_MLB_raw"] * COMPOSITE_WEIGHTS[CareerEvent.ESTABLISHED_MLB]
            + prow["p_STAR_raw"] * STAR_WEIGHT,
            3,
        )
        rows_probs.append(prow)

        # ---- Timing sheet ----
        trow = dict(ident)
        for e in DISPLAY_EVENTS:
            mt = cumP.get(("mean_t", e))
            sd = cumP.get(("sd_t", e))
            if mt is None:
                trow[f"t_{e.name}_mean"] = ""
                trow[f"t_{e.name}_sd"] = ""
            else:
                v = float(mt[i]); s = float(sd[i])
                trow[f"t_{e.name}_mean"] = round(v, 2) if v == v else ""
                trow[f"t_{e.name}_sd"] = round(s, 2) if s == s else ""
        em = float(star_mean_t[i]); es = float(star_sd[i])
        trow["t_STAR_mean"] = round(em, 2) if em == em else ""
        trow["t_STAR_sd"] = round(es, 2) if es == es else ""
        rows_timing.append(trow)

    # Rank by raw composite (preserves Platt's continuous separation)
    scores = np.array([r["composite_score_raw"] for r in rows_probs])
    order = scores.argsort()[::-1]
    pct = np.empty_like(scores, dtype=np.float64)
    for rank_pos, orig_i in enumerate(order):
        pct[orig_i] = 1.0 - rank_pos / max(len(order) - 1, 1)
    for i in range(len(rows_probs)):
        rows_probs[i]["percentile"] = round(float(pct[i]), 4)
        rows_probs[i]["grade"] = _letter_grade(pct[i])
        rows_timing[i]["percentile"] = rows_probs[i]["percentile"]
        rows_timing[i]["grade"] = rows_probs[i]["grade"]

    rows_probs.sort(key=lambda r: r["composite_score_raw"], reverse=True)
    rows_timing.sort(key=lambda r: r["percentile"], reverse=True)

    def _write(path, rows):
        if not rows: return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows):,} rows to {path}")
    _write(args.probs_out, rows_probs)
    _write(args.timing_out, rows_timing)

    print(f"\nTop 20 (by composite_score):")
    print(f"{'Rnk':>3} {'Gr':<3} {'Lvl':<4} {'Bld':>3} {'Player':<28} "
          f"{'P(MLB)':>7} {'P(Est)':>7} {'P(STAR)':>8} {'comp':>6}")
    print("-" * 100)
    for i, r in enumerate(rows_probs[:20], 1):
        print(f"{i:>3} {r['grade']:<3} {r['cur_level']:<4} "
              f"{r.get('blended', 0):>3} "
              f"{r['name'][:28]:<28} {r['p_MLB_DEBUT']:>7.3f} "
              f"{r['p_ESTABLISHED_MLB']:>7.3f} "
              f"{r['p_STAR']:>8.3f} {r['composite_score']:>6.2f}")


if __name__ == "__main__":
    main()
