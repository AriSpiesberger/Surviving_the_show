"""Generate snap=2026 buy list with v1.14d hazards (raw) + Lasso composite.

Hazards trained on 80% panel (held out: lasso fit 10% + lasso val 10%).
No Beta calibration. Lasso is fit on time-decay target (TOP_100 H=3,
MLB_DEBUT H=4) on raw hazard outputs.

Strategy filters available:
  --drop-r1                drop R1 picks (already priced in)
  --drop-already-top100    drop players already on a Top-100 list

Bins by absolute score (from val-slice calibration):
  >=1.5  near-lock          (val: 74-89% hit rate)
  1.0-1.5 high-conviction   (val: 52% hit rate)
  0.7-1.0 medium            (val: 42% hit rate)
  0.5-0.7 basket-play       (val: 30%)
  0.3-0.5 long-tail         (val: 23%)

Usage:
    python -m prospects.classifier.score_buy_list_v14d \\
        --model models/event_classifiers_v1.14d.pkl \\
        --lasso lasso_v14d_td.pkl \\
        --out buy_list_v14d_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sqlite3
from datetime import date

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, load_hazards, predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def _entry_year(p, stats_by_pid):
    dy = p.get("draft_year"); is_intl = int(p.get("is_international") or 0)
    if dy is not None and not is_intl: return int(dy)
    yrs = [s.get("season_year") for s in stats_by_pid.get(p["player_id"], [])
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    if yrs: return int(min(yrs))
    if dy is not None: return int(dy)
    return None


def _bucket_of(p):
    if int(p.get("is_international") or 0) == 1: return "IFA"
    r = p.get("draft_round")
    if r is None: return "IFA"
    r = int(r)
    if r == 1: return "R1"
    if r <= 3: return "R2-R3"
    if r <= 10: return "R4-R10"
    return "R10+"


def _age_at(birth, year):
    try:
        y, m, d = (int(x) for x in str(birth)[:10].split("-"))
        return (date(year, 6, 30) - date(y, m, d)).days / 365.25
    except Exception:
        return None


def _score_tier(s):
    if s >= 1.5: return "near-lock"
    if s >= 1.0: return "high-conviction"
    if s >= 0.7: return "medium"
    if s >= 0.5: return "basket-play"
    if s >= 0.3: return "long-tail"
    return "below-edge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--model", default="models/event_classifiers_v1.14d.pkl")
    ap.add_argument("--lasso", default="lasso_v14d_td.pkl")
    ap.add_argument("--snap-year", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--min-entry-year", type=int, default=2021)
    ap.add_argument("--max-entry-year", type=int, default=2026)
    ap.add_argument("--drop-r1", action="store_true")
    ap.add_argument("--drop-already-top100", action="store_true")
    ap.add_argument("--out", default="buy_list_v14d_2026.csv")
    args = ap.parse_args()

    print(f"Loading hazards: {args.model}")
    hazards = load_hazards(args.model)
    event_keys = [k for k in hazards
                  if k in (CareerEvent.TOP_100_PROSPECT, CareerEvent.MLB_DEBUT,
                           CareerEvent.ESTABLISHED_MLB)
                  or k == STAR_KEY or k == ELITE_KEY]

    print(f"Loading Lasso: {args.lasso}")
    with open(args.lasso, "rb") as f:
        comp = pickle.load(f)
    scaler = comp["scaler"]; lasso = comp["lasso"]
    age_center = comp["age_center"]
    yip_center = comp.get("years_in_pro_center", 3)

    db = ProspectDB(args.db)
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_top_100, o.year_established_mlb,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s); stats_by_pid.setdefault(d["player_id"], []).append(d)

    # Already-Top-100 lookup
    top100_first = {}
    with sqlite3.connect(args.db) as cx:
        for r in cx.execute("SELECT player_id, MIN(year) FROM prospect_rankings WHERE rank IS NOT NULL GROUP BY player_id"):
            top100_first[r[0]] = int(r[1])

    cohort = []
    n_drop_entry = n_drop_debut = n_drop_age = n_drop_r1 = n_drop_top100 = 0
    for r in rows:
        ent = _entry_year(r, stats_by_pid)
        if ent is None or not (args.min_entry_year <= ent <= args.max_entry_year):
            n_drop_entry += 1; continue
        debut = r.get("mlb_debut_year")
        if debut is not None and debut <= args.snap_year:
            n_drop_debut += 1; continue
        age = _age_at(r.get("birth_date"), args.snap_year)
        if age is None:
            n_drop_age += 1; continue
        bkt = _bucket_of(r)
        if args.drop_r1 and bkt == "R1":
            n_drop_r1 += 1; continue
        if args.drop_already_top100:
            t100 = top100_first.get(r["player_id"])
            if t100 is not None and t100 <= args.snap_year:
                n_drop_top100 += 1; continue
        r["_entry"] = ent; r["_age"] = age; r["_bucket"] = bkt
        cohort.append(r)
    print(f"Cohort: {len(cohort):,}  (skipped: entry={n_drop_entry}, "
          f"debuted={n_drop_debut}, no_age={n_drop_age}, R1={n_drop_r1}, "
          f"already_top100={n_drop_top100})")

    # Trim stats to <= snap (leakage-safe even though it's the current year)
    sub_stats = {r["player_id"]: [s for s in stats_by_pid.get(r["player_id"], [])
                                   if (s.get("season_year") or 0) <= args.snap_year]
                 for r in cohort}
    print(f"Predicting raw hazards at snap={args.snap_year}")
    cumP = predict_cumulative_batch(hazards, cohort, sub_stats,
                                    current_year=args.snap_year, horizon=args.horizon)
    n = len(cohort)
    # RAW probs (no Beta cal -- v1.14d has none anyway)
    def _raw(e, i):
        arr = cumP.get(("raw", e))
        return float(arr[i] if arr is not None else cumP[e][i])
    p_top = np.array([_raw(CareerEvent.TOP_100_PROSPECT, i) for i in range(n)])
    p_mlb = np.array([_raw(CareerEvent.MLB_DEBUT, i) for i in range(n)])
    p_est = np.array([_raw(CareerEvent.ESTABLISHED_MLB, i) for i in range(n)])
    p_star = np.array([_raw(STAR_KEY, i) for i in range(n)])
    p_elite = np.array([_raw(ELITE_KEY, i) for i in range(n)])
    p_spe = 1.0 - (1.0 - p_star) * (1.0 - p_elite)

    ages = np.array([r["_age"] for r in cohort])
    yips = np.array([args.snap_year - r["_entry"] for r in cohort])
    yip_c = yips - yip_center
    X = np.column_stack([
        p_top, p_mlb, p_est, p_spe, ages - age_center, yips,
        p_top * yip_c, p_mlb * yip_c, p_est * yip_c, p_spe * yip_c,
    ])
    X_scaled = scaler.transform(X)
    scores = lasso.predict(X_scaled)

    order = np.argsort(-scores)
    LEVEL_RANK = {"DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
                  "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}

    def _agg_year(rows, year):
        """Aggregate this season's stats across all (level, org) rows.
        Returns a dict with the primary level's stats."""
        yr = [s for s in rows if (s.get("season_year") or 0) == year]
        if not yr: return None
        # Primary level: highest level played that year
        best = max(yr, key=lambda s: LEVEL_RANK.get((s.get("level") or "").upper(), 0))
        # Aggregate at that primary level (sum across orgs at same level)
        lvl = (best.get("level") or "").upper()
        rows_at_lvl = [s for s in yr if (s.get("level") or "").upper() == lvl]
        # PA-weighted hitting / IP-weighted pitching
        def _wsum(key, w):
            tot_w = sum((r.get(w) or 0) for r in rows_at_lvl)
            if tot_w <= 0: return None
            tot_v = sum(((r.get(key) or 0) * (r.get(w) or 0)) for r in rows_at_lvl)
            return tot_v / tot_w
        pa = sum((r.get("pa") or 0) for r in rows_at_lvl)
        ip = sum((r.get("ip") or 0) for r in rows_at_lvl)
        return {
            "level": lvl,
            "pa": int(pa) if pa else 0,
            "ip": round(float(ip), 1) if ip else 0.0,
            "avg": round(_wsum("avg", "pa"), 3) if _wsum("avg", "pa") else None,
            "obp": round(_wsum("obp", "pa"), 3) if _wsum("obp", "pa") else None,
            "slg": round(_wsum("slg", "pa"), 3) if _wsum("slg", "pa") else None,
            "iso": round(_wsum("iso", "pa"), 3) if _wsum("iso", "pa") else None,
            "k_pct": round(_wsum("k_pct", "pa"), 3) if _wsum("k_pct", "pa") else None,
            "bb_pct": round(_wsum("bb_pct", "pa"), 3) if _wsum("bb_pct", "pa") else None,
            "hr": int(sum((r.get("home_runs") or 0) for r in rows_at_lvl)),
            "sb": int(sum((r.get("stolen_bases") or 0) for r in rows_at_lvl)),
            "era": round(_wsum("era", "ip"), 2) if _wsum("era", "ip") else None,
            "k9": round(_wsum("k9", "ip"), 2) if _wsum("k9", "ip") else None,
            "bb9": round(_wsum("bb9", "ip"), 2) if _wsum("bb9", "ip") else None,
            "whip": round(_wsum("whip", "ip"), 2) if _wsum("whip", "ip") else None,
            "fip": round(_wsum("fip", "ip"), 2) if _wsum("fip", "ip") else None,
        }

    out_rows = []
    for rank, i in enumerate(order, 1):
        r = cohort[i]; pid = r["player_id"]
        srows = sub_stats.get(pid, [])
        # This season's stats (could be partial mid-season)
        ys = _agg_year(srows, args.snap_year)
        # Last full year as fallback
        last_full = _agg_year(srows, args.snap_year - 1)
        cur_level = (ys["level"] if ys else (last_full["level"] if last_full else ""))
        bt = top100_first.get(pid)
        bt_str = bt if (bt is not None and bt <= args.snap_year) else ""
        row = {
            "buy_rank": rank,
            "buy_score": round(float(scores[i]), 4),
            "tier": _score_tier(float(scores[i])),
            "player_id": pid,
            "name": r.get("name"),
            "bucket": r["_bucket"],
            "entry_year": r["_entry"],
            "years_in_pro": int(yips[i]),
            "age_at_snap": round(r["_age"], 1),
            "primary_position": r.get("primary_position"),
            "current_org": r.get("current_org"),
            "cur_level": cur_level,
            "first_top100_yr": bt_str,
            "p_TOP_100_PROSPECT_raw": round(float(p_top[i]), 4),
            "p_MLB_DEBUT_raw": round(float(p_mlb[i]), 4),
            "p_ESTABLISHED_MLB_raw": round(float(p_est[i]), 4),
            "p_STAR_PLUS_ELITE_raw": round(float(p_spe[i]), 4),
        }
        # This season (snap_year) stats
        ys = ys or {}
        for k in ("pa","ip","avg","obp","slg","iso","k_pct","bb_pct","hr","sb",
                  "era","k9","bb9","whip","fip"):
            row[f"y{args.snap_year}_{k}"] = ys.get(k, "")
        out_rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"Wrote {args.out} ({len(out_rows):,} rows)\n")

    # Tier counts
    from collections import Counter
    tier_c = Counter(r["tier"] for r in out_rows)
    print(f"Tier distribution:")
    for t in ("near-lock", "high-conviction", "medium", "basket-play",
              "long-tail", "below-edge"):
        if tier_c.get(t, 0) > 0:
            print(f"  {t:<18} {tier_c[t]:>5}")
    print()
    # Show top of each tier
    for tier_filter in ("near-lock", "high-conviction", "medium"):
        chunk = [r for r in out_rows if r["tier"] == tier_filter]
        if not chunk: continue
        print(f"\n=== {tier_filter.upper()} ({len(chunk)} players) — top 20 ===")
        print(f"  {'rk':>4} {'score':>6} {'name':<28} {'bkt':<7} {'yip':>3} "
              f"{'lvl':>4} {'org':<22}")
        for r in chunk[:20]:
            print(f"  {r['buy_rank']:>4} {r['buy_score']:>6.2f} "
                  f"{(r['name'] or '')[:28]:<28} {r['bucket']:<7} "
                  f"{r['years_in_pro']:>3} {r['cur_level']:>4} "
                  f"{(r.get('current_org','') or '')[:22]:<22}")


if __name__ == "__main__":
    main()
