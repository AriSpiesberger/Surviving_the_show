"""Apply v1.14b hazards + Lasso composite to current MiLB prospects.

Cohort: players with entry_year in [min_entry_year, max_entry_year] who
have not yet MLB-debuted as of snap_year. Default snap_year = 2026.

For each player:
  1. Score with v1.14b at snap = snap_year (stats trimmed leakage-safe).
  2. Synthesize p_STAR_PLUS_ELITE = 1 - (1 - p_STAR)(1 - p_ELITE).
  3. Compute age_at_snap from birth_date.
  4. Compute years_in_pro = snap_year - entry_year.
  5. Apply Lasso composite (lasso_composite_v14b.pkl) -> buy_score.
  6. Rank descending; emit CSV.

Usage:
    python -m prospects.classifier.score_buy_list_v14b \\
        --model models/event_classifiers_v1.14b.pkl \\
        --lasso lasso_composite_v14b.pkl \\
        --snap-year 2026 \\
        --out buy_list_v14b_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
from datetime import date

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, load_hazards, predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def _entry_year(player: dict, stats_by_pid: dict) -> int | None:
    dy = player.get("draft_year")
    is_intl = int(player.get("is_international") or 0)
    if dy is not None and not is_intl:
        return int(dy)
    yrs = [s.get("season_year")
           for s in stats_by_pid.get(player["player_id"], [])
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    if yrs:
        return int(min(yrs))
    if dy is not None:
        return int(dy)
    return None


def _bucket_of(player: dict) -> str:
    if int(player.get("is_international") or 0) == 1:
        return "IFA"
    r = player.get("draft_round")
    if r is None:
        return "IFA"
    r = int(r)
    if r == 1: return "R1"
    if r <= 3: return "R2-R3"
    if r <= 10: return "R4-R10"
    return "R10+"


def _age_at(birth_iso: str | None, year: int) -> float | None:
    if not birth_iso:
        return None
    try:
        y, m, d = (int(x) for x in str(birth_iso)[:10].split("-"))
        return (date(year, 6, 30) - date(y, m, d)).days / 365.25
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--model", default="models/event_classifiers_v1.14b.pkl")
    ap.add_argument("--lasso", default="lasso_composite_v14b.pkl")
    ap.add_argument("--snap-year", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--min-entry-year", type=int, default=2021)
    ap.add_argument("--max-entry-year", type=int, default=2026)
    ap.add_argument("--out", default="buy_list_v14b_2026.csv")
    args = ap.parse_args()

    print(f"Loading hazards from {args.model}")
    hazards = load_hazards(args.model)
    event_keys = [k for k in hazards
                  if k in (CareerEvent.TOP_100_PROSPECT,
                           CareerEvent.MLB_DEBUT,
                           CareerEvent.ESTABLISHED_MLB)
                  or k == STAR_KEY or k == ELITE_KEY]

    print(f"Loading Lasso composite from {args.lasso}")
    with open(args.lasso, "rb") as f:
        comp = pickle.load(f)
    scaler = comp["scaler"]
    lasso = comp["lasso"]
    feature_names = comp["feature_names"]
    age_center = comp["age_center"]
    print(f"  features: {feature_names}")

    db = ProspectDB(args.db)
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_all_star_once,
                   o.year_all_star_three, o.year_major_award,
                   o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
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
    print(f"Loaded {len(rows):,} prospects")

    cohort = []
    skipped_debut = skipped_entry = skipped_age = 0
    for r in rows:
        ent = _entry_year(r, stats_by_pid)
        if ent is None or not (args.min_entry_year <= ent <= args.max_entry_year):
            skipped_entry += 1; continue
        debut = r.get("mlb_debut_year")
        if debut is not None and debut <= args.snap_year:
            skipped_debut += 1; continue
        age = _age_at(r.get("birth_date"), args.snap_year)
        if age is None:
            skipped_age += 1; continue
        r["_entry_year"] = ent
        r["_age"] = age
        cohort.append(r)
    print(f"Cohort at snap={args.snap_year}: {len(cohort):,} players "
          f"(entry {args.min_entry_year}-{args.max_entry_year}, "
          f"not-debuted)")
    print(f"  skipped: debut_by_snap={skipped_debut}, "
          f"entry_outside_range={skipped_entry}, no_birth_date={skipped_age}")

    # Trim stats to <= snap for leakage-safe scoring (already current
    # year but defensive).
    sub_stats = {}
    for r in cohort:
        pid = r["player_id"]
        sub_stats[pid] = [s for s in stats_by_pid.get(pid, [])
                          if (s.get("season_year") or 0) <= args.snap_year]
    print(f"Predicting hazards at snap={args.snap_year} (horizon={args.horizon})")
    cumP = predict_cumulative_batch(
        hazards, cohort, sub_stats,
        current_year=args.snap_year, horizon=args.horizon,
    )

    # Build feature matrix for Lasso (must match training order)
    n = len(cohort)
    p_top = np.array([float(cumP[CareerEvent.TOP_100_PROSPECT][i]) for i in range(n)])
    p_mlb = np.array([float(cumP[CareerEvent.MLB_DEBUT][i]) for i in range(n)])
    p_est = np.array([float(cumP[CareerEvent.ESTABLISHED_MLB][i]) for i in range(n)])
    p_star = np.array([float(cumP[STAR_KEY][i]) for i in range(n)])
    p_elite = np.array([float(cumP[ELITE_KEY][i]) for i in range(n)])
    p_starplus = 1.0 - (1.0 - p_star) * (1.0 - p_elite)

    yip_center = comp.get("years_in_pro_center", 3)
    ages = np.array([r["_age"] for r in cohort])
    yips = np.array([args.snap_year - r["_entry_year"] for r in cohort])
    yip_c = yips - yip_center

    # Feature column order (must match build_feature_matrix):
    #   p_TOP_100, p_MLB_DEBUT, p_ESTABLISHED, p_STAR_PLUS_ELITE,
    #   age_at_snap_centered, years_in_pro,
    #   p_*_x_yip_centered (4 interactions)
    X = np.column_stack([
        p_top, p_mlb, p_est, p_starplus,
        ages - age_center,
        yips,
        p_top * yip_c, p_mlb * yip_c, p_est * yip_c, p_starplus * yip_c,
    ])
    X_scaled = scaler.transform(X)
    scores = lasso.predict(X_scaled)

    # Rank descending
    order = np.argsort(-scores)
    n_total = len(cohort)

    out_rows = []
    for rank, i in enumerate(order, 1):
        r = cohort[i]
        pid = r["player_id"]
        # Best top-100 rank ever (informational)
        rk_hist = [int(rk) for (yr, rk, *_)
                   in rankings_by_pid.get(pid, [])
                   if rk is not None]
        best_top100 = min(rk_hist) if rk_hist else ""
        out_rows.append({
            "buy_rank": rank,
            "buy_score": round(float(scores[i]), 4),
            "score_pctile": round(100 * (1 - (rank - 1) / n_total), 2),
            "player_id": pid,
            "name": r.get("name"),
            "bucket": _bucket_of(r),
            "entry_year": r["_entry_year"],
            "years_in_pro": int(yips[i]),
            "age_at_snap": round(r["_age"], 1),
            "primary_position": r.get("primary_position"),
            "current_org": r.get("current_org"),
            "cur_level": "",  # populated below if available
            "best_top100_rank": best_top100,
            "p_TOP_100_PROSPECT": round(float(p_top[i]), 4),
            "p_MLB_DEBUT": round(float(p_mlb[i]), 4),
            "p_ESTABLISHED_MLB": round(float(p_est[i]), 4),
            "p_STAR_PLUS_ELITE": round(float(p_starplus[i]), 4),
        })

    # Populate cur_level (highest level seen)
    LEVEL_RANK = {"DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
                  "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}
    by_pid = {r["player_id"]: r for r in out_rows}
    for pid, srows in stats_by_pid.items():
        if pid not in by_pid: continue
        recent = [s for s in srows if (s.get("season_year") or 0) == args.snap_year]
        if not recent:
            recent = [s for s in srows if (s.get("season_year") or 0) == args.snap_year - 1]
        if not recent: continue
        best = max(recent,
                   key=lambda s: LEVEL_RANK.get((s.get("level") or "").upper(), 0))
        by_pid[pid]["cur_level"] = (best.get("level") or "").upper()

    fieldnames = list(out_rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {args.out} ({len(out_rows):,} rows)")

    # Quick summary: top 1%, 2%, 5%, 10% by bucket
    print(f"\nTop-K% bucket distribution:")
    for pct in (1, 2, 5, 10):
        k = max(1, int(round(n_total * pct / 100)))
        top = out_rows[:k]
        from collections import Counter
        c = Counter(r["bucket"] for r in top)
        print(f"  top {pct}% ({k:>4d} players): "
              + " ".join(f"{b}={c.get(b,0)}"
                         for b in ("R1","R2-R3","R4-R10","R10+","IFA")))

    print(f"\nTop 25 by buy_score:")
    print(f"  {'rk':>3} {'score':>6} {'name':<28} {'bkt':<7} "
          f"{'yip':>3} {'age':>4} {'lvl':>4} {'pTOP':>5} {'pDEB':>5} "
          f"{'pEST':>5} {'pSTAR':>5}")
    for r in out_rows[:25]:
        print(f"  {r['buy_rank']:>3d} {r['buy_score']:>6.2f} "
              f"{(r['name'] or '')[:28]:<28} {r['bucket']:<7} "
              f"{r['years_in_pro']:>3d} {r['age_at_snap']:>4.1f} "
              f"{r['cur_level']:>4} "
              f"{100*r['p_TOP_100_PROSPECT']:>4.1f} "
              f"{100*r['p_MLB_DEBUT']:>4.1f} "
              f"{100*r['p_ESTABLISHED_MLB']:>4.1f} "
              f"{100*r['p_STAR_PLUS_ELITE']:>4.1f}")


if __name__ == "__main__":
    main()
