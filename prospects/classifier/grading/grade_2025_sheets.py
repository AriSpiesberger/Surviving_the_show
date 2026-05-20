"""
prospects/classifier/grading/grade_2025_sheets.py
==========================================

Grade all 2025-active non-MLB prospects (drafted + IFA) and emit TWO CSVs:

    grades_probs.csv   — identity, level, 2025 stat line, calibrated +
                         raw probabilities per event, ELITE pooled metric,
                         composite score
    grades_timing.csv  — identity, level, 2025 stat line, E[T] and SD[T]
                         per event (years-from-now)

ELITE := 1 - (1 - P_AS3+)(1 - P_MAJOR_AWARD)(1 - P_HOF)
Captures "ever becomes elite" with a single, less-noisy number.
"""
from __future__ import annotations

import argparse
import csv
from typing import Optional

import numpy as np

from prospects.classifier.score_recent import _latest_season_completeness
from prospects.classifier.architectures.survival import (
    ELITE_KEY, load_hazards, predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


LEVEL_RANK = {"DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
              "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}


def _level_summary_2025(stats_2025: list[dict]) -> dict:
    """Return the player's top-level 2025 line (highest level by rank, with
    aggregated rate stats if multiple levels)."""
    if not stats_2025:
        return {
            "cur_level": "", "cur_pa": 0, "cur_ip": 0.0,
            "cur_avg": "", "cur_obp": "", "cur_slg": "", "cur_iso": "",
            "cur_k_pct": "", "cur_bb_pct": "", "cur_woba": "",
            "cur_hr": "", "cur_sb": "",
            "cur_era": "", "cur_k9": "", "cur_bb9": "", "cur_whip": "",
            "cur_fip": "", "cur_hr9": "",
        }
    # Best level by rank
    best = max(stats_2025,
               key=lambda s: LEVEL_RANK.get((s.get("level") or "").upper(), 0))
    lvl = (best.get("level") or "").upper()
    rows_at_lvl = [s for s in stats_2025 if (s.get("level") or "").upper() == lvl]

    def _wavg(key, weight_key):
        vals = [(s.get(key), s.get(weight_key) or 0)
                for s in rows_at_lvl
                if s.get(key) is not None and (s.get(weight_key) or 0) > 0]
        denom = sum(w for _, w in vals)
        if denom <= 0:
            return None
        return sum(v * w for v, w in vals) / denom

    pa = sum((s.get("pa") or 0) for s in rows_at_lvl)
    ip = sum((s.get("ip") or 0.0) for s in rows_at_lvl)
    hr = sum((s.get("home_runs") or 0) for s in rows_at_lvl) if pa > 0 else None
    sb = sum((s.get("stolen_bases") or 0) for s in rows_at_lvl) if pa > 0 else None

    def _fmt(v, p=3):
        return round(float(v), p) if v is not None else ""

    return {
        "cur_level": lvl,
        "cur_pa": pa,
        "cur_ip": round(ip, 1),
        "cur_avg":   _fmt(_wavg("avg", "pa")),
        "cur_obp":   _fmt(_wavg("obp", "pa")),
        "cur_slg":   _fmt(_wavg("slg", "pa")),
        "cur_iso":   _fmt(_wavg("iso", "pa")),
        "cur_k_pct": _fmt(_wavg("k_pct", "pa")),
        "cur_bb_pct":_fmt(_wavg("bb_pct", "pa")),
        "cur_woba":  _fmt(_wavg("woba", "pa")),
        "cur_hr":    hr if hr is not None else "",
        "cur_sb":    sb if sb is not None else "",
        "cur_era":   _fmt(_wavg("era", "ip"), 2),
        "cur_k9":    _fmt(_wavg("k9", "ip"), 2),
        "cur_bb9":   _fmt(_wavg("bb9", "ip"), 2),
        "cur_whip":  _fmt(_wavg("whip", "ip"), 2),
        "cur_fip":   _fmt(_wavg("fip", "ip"), 2),
        "cur_hr9":   _fmt(_wavg("hr9", "ip"), 2),
    }


COMPOSITE_WEIGHTS = {
    CareerEvent.MLB_DEBUT: 1.0,
    CareerEvent.ESTABLISHED_MLB: 3.0,
    CareerEvent.ALL_STAR_ONCE: 8.0,
}
ELITE_WEIGHT = 20.0
# Events we surface in the output. Skip AS3+/MAJOR_AWARD/HOF as individuals —
# they're folded into ELITE which is far less noisy.
DISPLAY_EVENTS = [
    CareerEvent.MLB_DEBUT,
    CareerEvent.ESTABLISHED_MLB,
    CareerEvent.ALL_STAR_ONCE,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects_snapshot.db")
    parser.add_argument("--model",
                        default="models/event_classifiers_v1.2_scouting.pkl")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--probs-out", default="grades_probs.csv")
    parser.add_argument("--timing-out", default="grades_timing.csv")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading hazards: {args.model}")
    hazards = load_hazards(args.model)
    event_keys = [k for k in hazards if not isinstance(k, str)]

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
            f"ORDER BY COALESCE(p.draft_year, 9999), p.draft_round, p.draft_pick",
            eligible,
        ).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    print(f"Scoring with current_year = {args.season}")
    X_list = []
    meta_list = []
    for p in prospects:
        all_stats = stats_by_pid.get(p["player_id"], [])
        meta_list.append((p, all_stats))
    cumP = predict_cumulative_batch(
        hazards, prospects,
        {p["player_id"]: stats_by_pid.get(p["player_id"], []) for p in prospects},
        current_year=args.season, horizon=args.horizon,
    )

    # ELITE comes directly from the trained pooled hazard (v1.3+).
    n = len(prospects)
    if ELITE_KEY in cumP:
        elite_cal = np.asarray(cumP[ELITE_KEY], dtype=np.float64)
        elite_raw = np.asarray(cumP.get(("raw", ELITE_KEY), elite_cal), dtype=np.float64)
        elite_mean_t = np.asarray(cumP.get(("mean_t", ELITE_KEY),
                                          np.full(n, np.nan)), dtype=np.float64)
        elite_sd = np.asarray(cumP.get(("sd_t", ELITE_KEY),
                                       np.full(n, np.nan)), dtype=np.float64)
    else:
        # Fallback: derive from individual events (v1.2 and earlier).
        elite_cal = np.ones(n); elite_raw = np.ones(n)
        for e in (CareerEvent.ALL_STAR_THREE_PLUS, CareerEvent.MAJOR_AWARD,
                  CareerEvent.HOF_TRAJECTORY):
            if e in cumP:
                elite_cal *= (1.0 - cumP[e])
                elite_raw *= (1.0 - cumP.get(("raw", e), cumP[e]))
        elite_cal = 1.0 - elite_cal
        elite_raw = 1.0 - elite_raw
        elite_mean_t = np.full(n, np.nan)
        elite_sd = np.full(n, np.nan)

    # Build per-prospect rows
    rows_probs = []
    rows_timing = []
    for i, (p, all_stats) in enumerate(meta_list):
        stats_2025 = [s for s in all_stats
                      if s.get("season_year") == args.season
                      and (s.get("level") or "").upper() != "MLB"]
        lvl_summary = _level_summary_2025(stats_2025)

        # IFAs don't have a draft_year; use their first observed MiLB year
        # as the start-of-pro reference. Surface both raw draft_year and a
        # derived "start_year" so downstream consumers can distinguish.
        draft_year = p.get("draft_year")
        if draft_year is None:
            milb_years = [s.get("season_year") for s in all_stats
                          if s.get("season_year") is not None
                          and (s.get("level") or "").upper() != "MLB"]
            start_year = min(milb_years) if milb_years else ""
        else:
            start_year = draft_year
        ident = {
            "player_id": p["player_id"],
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
            **lvl_summary,
        }

        # ---- Probabilities sheet ----
        prow = dict(ident)
        # Per-event P (calibrated and raw) — DISPLAY_EVENTS only
        for e in DISPLAY_EVENTS:
            if e in cumP:
                prow[f"p_{e.name}"] = round(float(cumP[e][i]), 4)
                raw = cumP.get(("raw", e))
                prow[f"p_{e.name}_raw"] = round(float(raw[i]), 4) if raw is not None else prow[f"p_{e.name}"]
            else:
                prow[f"p_{e.name}"] = 0.0
                prow[f"p_{e.name}_raw"] = 0.0
        # Monotonicity on both cal & raw
        chain = [
            (CareerEvent.MLB_DEBUT, CareerEvent.ESTABLISHED_MLB),
            (CareerEvent.MLB_DEBUT, CareerEvent.ALL_STAR_ONCE),
            (CareerEvent.ESTABLISHED_MLB, CareerEvent.ALL_STAR_ONCE),
        ]
        for broader, narrower in chain:
            for suffix in ("", "_raw"):
                kn = f"p_{narrower.name}{suffix}"
                kb = f"p_{broader.name}{suffix}"
                prow[kn] = round(min(prow.get(kn, 0.0), prow.get(kb, 1.0)), 4)
        prow["p_ELITE"] = round(float(elite_cal[i]), 4)
        prow["p_ELITE_raw"] = round(float(elite_raw[i]), 4)
        # ELITE requires both MLB debut AND establishment. Independent hazards
        # can fire inconsistently; cap ELITE by ESTABLISHED_MLB which already
        # is capped by MLB_DEBUT.
        for suffix in ("", "_raw"):
            ke = f"p_ELITE{suffix}"
            ks = f"p_ESTABLISHED_MLB{suffix}"
            prow[ke] = round(min(prow.get(ke, 0.0), prow.get(ks, 1.0)), 4)
        prow["composite_score"] = round(
            prow["p_MLB_DEBUT"] * COMPOSITE_WEIGHTS[CareerEvent.MLB_DEBUT]
            + prow["p_ESTABLISHED_MLB"] * COMPOSITE_WEIGHTS[CareerEvent.ESTABLISHED_MLB]
            + prow["p_ALL_STAR_ONCE"] * COMPOSITE_WEIGHTS[CareerEvent.ALL_STAR_ONCE]
            + prow["p_ELITE"] * ELITE_WEIGHT,
            3,
        )
        prow["composite_score_raw"] = round(
            prow["p_MLB_DEBUT_raw"] * COMPOSITE_WEIGHTS[CareerEvent.MLB_DEBUT]
            + prow["p_ESTABLISHED_MLB_raw"] * COMPOSITE_WEIGHTS[CareerEvent.ESTABLISHED_MLB]
            + prow["p_ALL_STAR_ONCE_raw"] * COMPOSITE_WEIGHTS[CareerEvent.ALL_STAR_ONCE]
            + prow["p_ELITE_raw"] * ELITE_WEIGHT,
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
        em = elite_mean_t[i]; es = elite_sd[i]
        trow["t_ELITE_mean"] = round(float(em), 2) if em == em else ""
        trow["t_ELITE_sd"] = round(float(es), 2) if es == es else ""
        rows_timing.append(trow)

    # Rank both sheets by composite_score_raw (preserves elite-tier ordering)
    order = sorted(range(len(rows_probs)),
                   key=lambda i: rows_probs[i]["composite_score_raw"],
                   reverse=True)
    n = len(rows_probs)
    pct_arr = np.empty(n, dtype=np.float64)
    for rank_pos, orig_i in enumerate(order):
        pct_arr[orig_i] = 1.0 - rank_pos / max(n - 1, 1)
    for i in range(n):
        rows_probs[i]["percentile"] = round(float(pct_arr[i]), 4)
        rows_probs[i]["grade"] = _letter_grade(pct_arr[i])
        rows_timing[i]["percentile"] = rows_probs[i]["percentile"]
        rows_timing[i]["grade"] = rows_probs[i]["grade"]

    rows_probs.sort(key=lambda r: r["composite_score_raw"], reverse=True)
    rows_timing.sort(key=lambda r: r["percentile"], reverse=True)

    def _write(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows):,} rows to {path}")
    _write(args.probs_out, rows_probs)
    _write(args.timing_out, rows_timing)

    # Print top 20 from probs
    print(f"\nTop 20 (by composite_score_raw):")
    print(f"{'Rnk':>3} {'Gr':<3} {'Lvl':<4} {'Player':<28} {'P(MLB)':>7} "
          f"{'P(Est)':>7} {'P(AS1)':>7} {'P(ELITE)':>8}")
    print("-" * 90)
    for i, r in enumerate(rows_probs[:20], 1):
        print(f"{i:>3} {r['grade']:<3} {r['cur_level']:<4} "
              f"{r['name'][:28]:<28} {r['p_MLB_DEBUT_raw']:>7.3f} "
              f"{r['p_ESTABLISHED_MLB_raw']:>7.3f} "
              f"{r['p_ALL_STAR_ONCE_raw']:>7.3f} {r['p_ELITE_raw']:>8.3f}")


if __name__ == "__main__":
    main()
