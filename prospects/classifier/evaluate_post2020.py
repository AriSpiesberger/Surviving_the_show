"""Evaluate v1.13 on the held-out 2021-2026 entry cohort.

For each snap year (entry_year + 1), produce a sheet that mirrors the
live grades sheet (bio, level snapshot as-of snap, Top-100 history
as-of snap, predicted probabilities, composite scores) AND appends the
realized future outcomes through --observe-through:

    realized_<E>          1 if event triggered in (snap, observe_through]
    trigger_year_<E>      first year the event fired (any time)
    eligible_at_snap_<E>  1 if event had NOT yet fired at snap_year

Strictly forward-looking: per-event metrics consider only rows where
eligible_at_snap_<E> = 1.

Output: one CSV per snap year, plus a combined _ALL.csv. Columns match
grades_probs_2026_v13.csv plus the realized_* / trigger_year_* /
eligible_* fields.

Usage:
    python -m prospects.classifier.evaluate_post2020 \\
        --model models/event_classifiers_v1.13.pkl \\
        --observe-through 2026 \\
        --out-prefix eval_post2020_v13
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, _trigger_year, load_hazards,
    predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


DISPLAY_EVENTS = (
    CareerEvent.TOP_100_PROSPECT,
    CareerEvent.MLB_DEBUT,
    CareerEvent.ESTABLISHED_MLB,
)
COMPOSITE_WEIGHTS = {
    CareerEvent.TOP_100_PROSPECT: 0.5,
    CareerEvent.MLB_DEBUT: 1.0,
    CareerEvent.ESTABLISHED_MLB: 3.0,
}
STAR_WEIGHT = 10.0
LEVEL_RANK = {"DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
              "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}


def _ev_name(e) -> str:
    if hasattr(e, "name"):
        return e.name
    return str(e).lstrip("_")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _age_at(birth_date_iso: str | None, year: int) -> float | None:
    if not birth_date_iso:
        return None
    try:
        y, m, d = (int(x) for x in str(birth_date_iso)[:10].split("-"))
        return (date(year, 6, 30) - date(y, m, d)).days / 365.25
    except (TypeError, ValueError):
        return None


def _level_summary_for_year(stats: list[dict], season: int) -> dict:
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
        }
    best = max(yr_rows,
               key=lambda s: LEVEL_RANK.get((s.get("level") or "").upper(), 0))
    lvl = (best.get("level") or "").upper()
    rows_at_lvl = [s for s in yr_rows
                   if (s.get("level") or "").upper() == lvl]

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
        "cur_bb_pct": _fmt(_wavg("bb_pct", "pa")),
        "cur_woba":  _fmt(_wavg("woba", "pa")),
        "cur_hr":    int(hr) if hr is not None else "",
        "cur_sb":    int(sb) if sb is not None else "",
        "cur_era":   _fmt(_wavg("era", "ip"), 2),
        "cur_k9":    _fmt(_wavg("k9", "ip"), 2),
        "cur_bb9":   _fmt(_wavg("bb9", "ip"), 2),
        "cur_whip":  _fmt(_wavg("whip", "ip"), 2),
        "cur_fip":   _fmt(_wavg("fip", "ip"), 2),
        "cur_hr9":   _fmt(_wavg("hr9", "ip"), 2),
    }


def _top100_summary_as_of(rankings: list, snap: int) -> dict:
    """rankings is list of (year, rank, source). Summarize what we knew
    about the player's Top-100 history AS OF end of `snap`."""
    past = [(int(y), int(r)) for (y, r, *_rest) in rankings
            if y is not None and r is not None and int(y) <= snap]
    if not past:
        return {
            "best_top100_rank": "",
            "recent_top100_rank": "",
            "times_top100": 0,
            "first_top100_year": "",
        }
    yrs = [y for y, _ in past]; rs = [r for _, r in past]
    latest = max(yrs)
    return {
        "best_top100_rank": min(rs),
        "recent_top100_rank": next(r for y, r in past if y == latest),
        "times_top100": len(past),
        "first_top100_year": min(yrs),
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--model", default="models/event_classifiers_v1.13.pkl")
    ap.add_argument("--min-entry-year", type=int, default=2021)
    ap.add_argument("--max-entry-year", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--observe-through", type=int, default=2026)
    ap.add_argument("--out-prefix", default="eval_post2020_v13")
    args = ap.parse_args()
    MAX_OBS = args.observe_through

    db = ProspectDB(args.db)
    print(f"Loading model: {args.model}")
    hazards = load_hazards(args.model)
    event_keys = [k for k in hazards
                  if (k in DISPLAY_EVENTS) or k == STAR_KEY or k == ELITE_KEY]
    print(f"  events: {[_ev_name(e) for e in event_keys]}")

    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
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
    print(f"  loaded {len(rows):,} prospects, "
          f"{sum(len(v) for v in stats_by_pid.values()):,} stat rows, "
          f"{len(rank_rows):,} rank rows")

    cohort: list[dict] = []
    for r in rows:
        ent = _entry_year(r, stats_by_pid)
        if ent is None:
            continue
        if not (args.min_entry_year <= ent <= args.max_entry_year):
            continue
        r["_entry_year"] = ent
        cohort.append(r)
    n_drafted = sum(1 for r in cohort
                    if not int(r.get("is_international") or 0))
    n_ifa = len(cohort) - n_drafted
    print(f"Cohort: {len(cohort):,} ({n_drafted:,} drafted, "
          f"{n_ifa:,} IFA) entry_year in "
          f"[{args.min_entry_year}, {args.max_entry_year}]")

    snap_groups: dict[int, list[dict]] = {}
    skipped_debuted = 0
    skipped_horizon = 0
    for r in cohort:
        snap = int(r["_entry_year"]) + 1
        if snap > MAX_OBS:
            skipped_horizon += 1
            continue
        debut = r.get("mlb_debut_year")
        if debut is not None and debut <= snap:
            skipped_debuted += 1
            continue
        snap_groups.setdefault(snap, []).append(r)
    total = sum(len(g) for g in snap_groups.values())
    print(f"  scoring {total:,} snapshots across snap years "
          f"{min(snap_groups)}-{max(snap_groups)}")
    print(f"  skipped: debuted-by-snap={skipped_debuted}, "
          f"snap>{MAX_OBS}={skipped_horizon}")

    per_snap_rows: dict[int, list[dict]] = {}
    for snap, group in sorted(snap_groups.items()):
        # IMPORTANT: predict_cumulative_batch must NOT see stats from
        # snap+1 onward (would be leakage). Trim stats to season<=snap.
        sub_stats = {}
        for r in group:
            pid = r["player_id"]
            sub_stats[pid] = [s for s in stats_by_pid.get(pid, [])
                              if (s.get("season_year") or 0) <= snap]
        cumP = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=snap, horizon=args.horizon,
        )
        rows_out = []
        for i, r in enumerate(group):
            pid = r["player_id"]
            stats_to_snap = sub_stats[pid]
            lvl = _level_summary_for_year(stats_to_snap, snap)
            # Fallback to prior year if no snap-year stats yet
            if not lvl["cur_level"] and snap - 1 >= 2021:
                lvl = _level_summary_for_year(stats_to_snap, snap - 1)
            rankings_to_snap = [(y, rk, src)
                                for (y, rk, src) in rankings_by_pid.get(pid, [])
                                if y is not None and y <= snap]
            top100 = _top100_summary_as_of(rankings_to_snap, snap)
            age = _age_at(r.get("birth_date"), snap)

            row = {
                "player_id": pid,
                "name": r.get("name"),
                "snap_year": snap,
                "entry_year": r["_entry_year"],
                "is_international": int(r.get("is_international") or 0),
                "draft_round": r.get("draft_round"),
                "primary_position": r.get("primary_position"),
                "current_org": r.get("current_org"),
                "age_at_snap": (round(age, 2)
                                if age is not None else ""),
                "cur_level_at_snap": lvl.get("cur_level", ""),
                "years_observed_after_snap": MAX_OBS - snap,
            }
            # Predictions
            comp = 0.0; comp_raw = 0.0
            for e in event_keys:
                ename = _ev_name(e)
                p_cal = float(cumP[e][i])
                p_raw_arr = cumP.get(("raw", e))
                p_raw = (float(p_raw_arr[i])
                         if p_raw_arr is not None else p_cal)
                row[f"p_{ename}"] = round(p_cal, 4)
                row[f"p_{ename}_raw"] = round(p_raw, 4)
                # Composite (matches grader's weighting)
                if e in COMPOSITE_WEIGHTS:
                    comp += p_cal * COMPOSITE_WEIGHTS[e]
                    comp_raw += p_raw * COMPOSITE_WEIGHTS[e]
                elif e == STAR_KEY:
                    comp += p_cal * STAR_WEIGHT
                    comp_raw += p_raw * STAR_WEIGHT
            row["composite_score"] = round(comp, 3)
            row["composite_score_raw"] = round(comp_raw, 3)

            # Realized + eligibility (forward-looking)
            for e in event_keys:
                ename = _ev_name(e)
                trig = _trigger_year(r, e)
                eligible = int(trig is None or trig > snap)
                realized = int(trig is not None
                               and trig > snap
                               and trig <= MAX_OBS)
                row[f"eligible_at_snap_{ename}"] = eligible
                row[f"realized_after_snap_{ename}"] = realized
                row[f"trigger_year_{ename}"] = trig
                # Years from snap to event (if realized)
                if realized:
                    row[f"years_to_realize_{ename}"] = trig - snap
                else:
                    row[f"years_to_realize_{ename}"] = ""
            rows_out.append(row)
        # Rank within snap by composite_score
        rows_out.sort(key=lambda x: -x["composite_score"])
        for rank_idx, x in enumerate(rows_out, 1):
            x["rank_at_snap"] = rank_idx
        per_snap_rows[snap] = rows_out

    # ---- Write per-snap CSVs ----
    out_dir = os.path.dirname(args.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    first_rows = next(iter(per_snap_rows.values()))
    fieldnames = ["rank_at_snap"] + [k for k in first_rows[0].keys()
                                     if k != "rank_at_snap"]
    all_rows: list[dict] = []
    written = []
    for snap, rows_out in sorted(per_snap_rows.items()):
        path = f"{args.out_prefix}_snap{snap}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        written.append((path, len(rows_out)))
        all_rows.extend(rows_out)
    all_path = f"{args.out_prefix}_ALL.csv"
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    written.append((all_path, len(all_rows)))
    for p, n in written:
        print(f"  wrote {p}  ({n:,} rows)")

    # ---- Summary ----
    def _metrics(rows, ename):
        if not rows:
            return (0, 0, float("nan"), float("nan"),
                    float("nan"), float("nan"))
        p = np.array([r[f"p_{ename}"] for r in rows])
        y = np.array([r[f"realized_after_snap_{ename}"] for r in rows])
        try:
            auc = (roc_auc_score(y, p)
                   if 0 < y.sum() < len(y) else float("nan"))
            brier = brier_score_loss(y, p)
        except Exception:
            auc = float("nan"); brier = float("nan")
        return (len(rows), int(y.sum()), p.mean(), y.mean(), auc, brier)

    print(f"\n{'='*78}")
    print(f"  FORWARD-LOOKING EVAL  (predict @ snap, realized in "
          f"(snap, {MAX_OBS}])")
    print(f"{'='*78}\n")
    print(f"OVERALL (entry_year {args.min_entry_year}-"
          f"{args.max_entry_year}, eligible only):")
    print(f"  {'Event':<22} {'n':>6} {'pos':>5} {'pred%':>7} "
          f"{'real%':>7} {'AUC':>6} {'Brier':>7}")
    for e in event_keys:
        ename = _ev_name(e)
        elig = [r for r in all_rows if r[f"eligible_at_snap_{ename}"] == 1]
        n, pos, pmean, rmean, auc, brier = _metrics(elig, ename)
        print(f"  {ename:<22} {n:>6d} {pos:>5d} "
              f"{100*pmean:>6.2f}% {100*rmean:>6.2f}% "
              f"{auc:>6.3f} {brier:>7.4f}")

    print(f"\nBY SNAP_YEAR (eligible only):")
    for e in event_keys:
        ename = _ev_name(e)
        print(f"\n  Event: {ename}")
        print(f"  {'snap':>5} {'obs_yrs':>7} {'n':>6} {'pos':>5} "
              f"{'pred%':>7} {'real%':>7} {'AUC':>6}")
        for snap in sorted(per_snap_rows):
            elig = [r for r in per_snap_rows[snap]
                    if r[f"eligible_at_snap_{ename}"] == 1]
            n, pos, pmean, rmean, auc, brier = _metrics(elig, ename)
            print(f"  {snap:>5d} {MAX_OBS - snap:>7d} {n:>6d} "
                  f"{pos:>5d} {100*pmean:>6.2f}% {100*rmean:>6.2f}% "
                  f"{auc:>6.3f}")


if __name__ == "__main__":
    main()
