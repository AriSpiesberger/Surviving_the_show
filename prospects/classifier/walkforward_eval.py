"""Walk-forward evaluation: score each player at MULTIPLE snap years.

Answers "when are my predictions safe?" by re-scoring the same player at
snap = entry_year + 0, +1, +2, ... and tracking how AUC / Brier / mean(p)
evolves as the model accumulates more MiLB seasons of observation.

Two cohorts are produced (selectable via --mode):

  testsplit  10% held-out test players (drafted <= --max-draft-year, plus
             all IFAs), reproducing the validation_predictions.py split
             (seed=42, player-grouped).

  cohort2021 All 2021 draftees (is_international=0, draft_year=2021).

For each (player, snap) pair:
  - Stats are trimmed to season_year <= snap before prediction (leakage-safe).
  - Skip if player MLB-debuted by snap (no longer a prospect).
  - Skip if snap > --observe-through.
  - realized_after_snap_<E> = 1 iff trigger_year in (snap, observe_through].

Output:
  <prefix>_long.csv      one row per (cohort, player_id, snap_year)
  <prefix>_summary.txt   per-(cohort, snap_offset, event) AUC/Brier/means

Usage:
  python -m prospects.classifier.walkforward_eval \\
      --model models/event_classifiers_v1.13.pkl \\
      --observe-through 2025 \\
      --out-prefix walkforward_v13
"""
from __future__ import annotations

import argparse
import csv
import os

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


def _ev_name(e) -> str:
    if hasattr(e, "name"):
        return e.name
    return str(e).lstrip("_")


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


def _testsplit_players(rows, seed: int, max_draft_year: int) -> set[str]:
    """Reproduce the player-grouped 80/10/10 split from training. Returns
    the first 10% slice — the truly held-out set (never seen by training
    OR by calibration; calibration uses the next 10%, the 'val' slice)."""
    pool = [r for r in rows
            if (r.get("draft_year") is not None
                and r["draft_year"] <= max_draft_year)
            or int(r.get("is_international") or 0) == 1]
    unique = sorted({r["player_id"] for r in pool})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    n_test = int(round(0.10 * len(unique)))
    return {unique[i] for i in perm[:n_test]}


def _filter_by_entry_year(players: set[str], rows: list[dict],
                          stats_by_pid: dict, max_entry_year: int) -> set[str]:
    """Keep only players whose entry_year (draft_year for drafted, first
    non-MLB season for IFA) is <= max_entry_year. Ensures forward
    realization window is long enough for slow events."""
    out = set()
    for r in rows:
        if r["player_id"] not in players:
            continue
        ent = _entry_year(r, stats_by_pid)
        if ent is not None and ent <= max_entry_year:
            out.add(r["player_id"])
    return out


def _cohort2021_players(rows) -> set[str]:
    return {r["player_id"] for r in rows
            if r.get("draft_year") == 2021
            and int(r.get("is_international") or 0) == 0}


def _score_cohort(
    *,
    cohort_tag: str,
    cohort_rows: list[dict],
    stats_by_pid: dict,
    hazards,
    event_keys,
    observe_through: int,
    horizon: int,
    min_snap_offset: int,
) -> list[dict]:
    """Walk forward over snaps for each player in `cohort_rows`. Returns
    a long-form list of dicts."""
    # Build (snap_year -> [player_rows]) groups so we can batch by snap.
    snap_groups: dict[int, list[dict]] = {}
    skipped_debuted = 0
    skipped_horizon = 0
    skipped_no_entry = 0
    for r in cohort_rows:
        ent = _entry_year(r, stats_by_pid)
        if ent is None:
            skipped_no_entry += 1
            continue
        debut = r.get("mlb_debut_year")
        for offset in range(min_snap_offset, (observe_through - ent) + 1):
            snap = ent + offset
            if snap > observe_through:
                skipped_horizon += 1
                continue
            if debut is not None and debut <= snap:
                skipped_debuted += 1
                continue
            r_copy = dict(r)
            r_copy["_entry_year"] = ent
            r_copy["_snap"] = snap
            r_copy["_snap_offset"] = offset
            snap_groups.setdefault(snap, []).append(r_copy)

    total = sum(len(g) for g in snap_groups.values())
    print(f"  [{cohort_tag}] {total:,} (player, snap) rows across snap "
          f"years {min(snap_groups)}..{max(snap_groups)}  "
          f"(skipped: debuted={skipped_debuted}, "
          f"beyond_obs={skipped_horizon}, no_entry={skipped_no_entry})")

    out_rows: list[dict] = []
    for snap, group in sorted(snap_groups.items()):
        # Trim stats to <= snap for leakage-safe prediction.
        sub_stats = {}
        for r in group:
            pid = r["player_id"]
            sub_stats[pid] = [s for s in stats_by_pid.get(pid, [])
                              if (s.get("season_year") or 0) <= snap]
        # NOTE: horizon is kept at the full 15 years (matching the model's
        # training/inference convention). The realization window we score
        # against is only (snap, observe_through], which is shorter — so
        # mean(p) will exceed mean(realized) by construction. AUC ordering
        # is unaffected; treat the calibration numbers as upper bounds on
        # the next observe_through-snap years, not as point estimates.
        # A horizon-matched variant segfaults inside predict_cumulative_batch
        # at small horizons; needs separate investigation.
        cumP = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=snap, horizon=horizon,
        )
        for i, r in enumerate(group):
            row = {
                "cohort": cohort_tag,
                "player_id": r["player_id"],
                "name": r.get("name"),
                "draft_year": r.get("draft_year"),
                "draft_round": r.get("draft_round"),
                "draft_pick": r.get("draft_pick"),
                "is_international": int(r.get("is_international") or 0),
                "primary_position": r.get("primary_position"),
                "current_org": r.get("current_org"),
                "entry_year": r["_entry_year"],
                "snap_year": snap,
                "snap_offset": r["_snap_offset"],
                "years_observed_after_snap": observe_through - snap,
                "mlb_debut_year": r.get("mlb_debut_year"),
            }
            per_event = {}
            for e in event_keys:
                ename = _ev_name(e)
                p_cal = float(cumP[e][i])
                p_raw_arr = cumP.get(("raw", e))
                p_raw = (float(p_raw_arr[i])
                         if p_raw_arr is not None else p_cal)
                trig = _trigger_year(r, e)
                eligible = int(trig is None or trig > snap)
                realized = int(trig is not None
                               and trig > snap
                               and trig <= observe_through)
                per_event[ename] = (p_cal, p_raw, trig, eligible, realized)
                row[f"p_{ename}"] = round(p_cal, 4)
                row[f"p_{ename}_raw"] = round(p_raw, 4)
                row[f"eligible_at_snap_{ename}"] = eligible
                row[f"realized_after_snap_{ename}"] = realized
                row[f"trigger_year_{ename}"] = trig
                row[f"years_to_realize_{ename}"] = (
                    trig - snap if realized else "")
            # Synthetic STAR_PLUS_ELITE: ELITE is a higher tier — fold it
            # into STAR so we evaluate the "star-or-better" question rather
            # than two thin, rare tiers separately.
            if "STAR" in per_event and "ELITE" in per_event:
                ps, ps_raw, ts, es, rs = per_event["STAR"]
                pe, pe_raw, te, ee, re_ = per_event["ELITE"]
                # P(STAR or ELITE) under independence: 1 - (1-ps)(1-pe).
                # Same for raw.
                p_union = 1.0 - (1.0 - ps) * (1.0 - pe)
                p_union_raw = 1.0 - (1.0 - ps_raw) * (1.0 - pe_raw)
                # Earliest trigger; eligible if neither fired by snap;
                # realized if either fired in (snap, observe_through].
                trigs = [t for t in (ts, te) if t is not None]
                trig_u = min(trigs) if trigs else None
                eligible_u = int(trig_u is None or trig_u > snap)
                realized_u = int(trig_u is not None
                                 and trig_u > snap
                                 and trig_u <= observe_through)
                row["p_STAR_PLUS_ELITE"] = round(p_union, 4)
                row["p_STAR_PLUS_ELITE_raw"] = round(p_union_raw, 4)
                row["eligible_at_snap_STAR_PLUS_ELITE"] = eligible_u
                row["realized_after_snap_STAR_PLUS_ELITE"] = realized_u
                row["trigger_year_STAR_PLUS_ELITE"] = trig_u
                row["years_to_realize_STAR_PLUS_ELITE"] = (
                    trig_u - snap if realized_u else "")
            out_rows.append(row)
    return out_rows


def _metrics(rows, ename):
    """AUC/Brier/means on the eligible subset of `rows` for one event."""
    elig = [r for r in rows if r[f"eligible_at_snap_{ename}"] == 1]
    if not elig:
        return (0, 0, float("nan"), float("nan"),
                float("nan"), float("nan"))
    p = np.array([r[f"p_{ename}"] for r in elig], dtype=float)
    y = np.array([r[f"realized_after_snap_{ename}"]
                  for r in elig], dtype=float)
    try:
        auc = (roc_auc_score(y, p)
               if 0 < y.sum() < len(y) else float("nan"))
        brier = brier_score_loss(y, p)
    except Exception:
        auc = float("nan"); brier = float("nan")
    return (len(elig), int(y.sum()), float(p.mean()),
            float(y.mean()), auc, brier)


_REPORT_EVENT_NAMES = [
    "TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE",
]


def _write_summary(rows, event_keys, path, observe_through):
    """Group by snap_offset (years since entry) — the walk-forward axis.
    Each row answers: 'with K years of pro data, how good are predictions?'
    Forward observation window varies across players in the same offset
    bucket (depending on entry_year); mean_fwd_yr is reported per row for
    context. Players whose event already fired at snap are excluded from
    that event's metrics via the eligible_at_snap flag.
    """
    lines = []
    lines.append(f"WALK-FORWARD EVAL SUMMARY")
    lines.append(f"  realization window per player: (snap, {observe_through}]")
    lines.append(f"  axis: snap_offset = years since entry_year")
    lines.append(f"  eligibility: player excluded from event E's row "
                 f"if E already fired by snap")
    lines.append(f"  mean_fwd = avg forward observation years "
                 f"(higher = more reliable real%)")
    lines.append("")
    cohorts = sorted({r["cohort"] for r in rows})
    for cohort in cohorts:
        crows = [r for r in rows if r["cohort"] == cohort]
        lines.append(f"{'='*78}")
        lines.append(f"COHORT: {cohort}   "
                     f"(n_players={len({r['player_id'] for r in crows}):,}, "
                     f"n_rows={len(crows):,})")
        lines.append(f"{'='*78}")
        report_names = [n for n in _REPORT_EVENT_NAMES
                        if any(f"p_{n}" in r for r in crows[:1])]
        for ename in report_names:
            lines.append(f"\n  Event: {ename}")
            lines.append(f"  {'offset':>6} {'n_elig':>6} "
                         f"{'mean_fwd':>8} {'pos':>4} "
                         f"{'pred%':>7} {'real%':>7} "
                         f"{'AUC':>6} {'Brier':>7}")
            for k in sorted({r["snap_offset"] for r in crows}):
                sub = [r for r in crows if r["snap_offset"] == k]
                elig_sub = [r for r in sub
                            if r.get(f"eligible_at_snap_{ename}") == 1]
                if not elig_sub:
                    continue
                mean_fwd = sum(observe_through - r["snap_year"]
                               for r in elig_sub) / len(elig_sub)
                n, pos, pm, rm, auc, br = _metrics(sub, ename)
                lines.append(f"  {k:>6d} {n:>6d} "
                             f"{mean_fwd:>8.1f} {pos:>4d} "
                             f"{100*pm:>6.2f}% {100*rm:>6.2f}% "
                             f"{auc:>6.3f} {br:>7.4f}")
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--model", default="models/event_classifiers_v1.13.pkl")
    ap.add_argument("--mode", default="validation",
                    choices=["validation", "testsplit", "cohort2021"])
    ap.add_argument("--seed", type=int, default=42,
                    help="Match the seed used in validation_predictions.py")
    ap.add_argument("--max-draft-year", type=int, default=2020,
                    help="Filter for the split pool (matches training)")
    ap.add_argument("--max-eval-entry-year", type=int, default=2015,
                    help="Restrict the walk-forward cohort to players "
                         "whose entry_year <= this. Default 2015 ensures "
                         "10+ yr forward observation for slow events.")
    ap.add_argument("--observe-through", type=int, default=2025)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--min-snap-offset", type=int, default=0,
                    help="Start snap at entry_year + this offset")
    ap.add_argument("--out-prefix", default="walkforward_v13")
    args = ap.parse_args()

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
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    print(f"  loaded {len(rows):,} prospects, "
          f"{sum(len(v) for v in stats_by_pid.values()):,} stat rows")

    all_out: list[dict] = []

    if args.mode == "validation":
        held_out = _testsplit_players(rows, args.seed, args.max_draft_year)
        held_out = _filter_by_entry_year(
            held_out, rows, stats_by_pid, args.max_eval_entry_year)
        cohort_rows = [r for r in rows if r["player_id"] in held_out]
        print(f"\nScoring VALIDATION (held-out, entry_year<="
              f"{args.max_eval_entry_year}): {len(cohort_rows):,} players")
        all_out.extend(_score_cohort(
            cohort_tag="validation",
            cohort_rows=cohort_rows,
            stats_by_pid=stats_by_pid,
            hazards=hazards,
            event_keys=event_keys,
            observe_through=args.observe_through,
            horizon=args.horizon,
            min_snap_offset=args.min_snap_offset,
        ))

    if args.mode == "testsplit":
        test_ids = _testsplit_players(rows, args.seed, args.max_draft_year)
        cohort_rows = [r for r in rows if r["player_id"] in test_ids]
        print(f"\nScoring TESTSPLIT (full, no entry-year filter): "
              f"{len(cohort_rows):,} players")
        all_out.extend(_score_cohort(
            cohort_tag="testsplit",
            cohort_rows=cohort_rows,
            stats_by_pid=stats_by_pid,
            hazards=hazards,
            event_keys=event_keys,
            observe_through=args.observe_through,
            horizon=args.horizon,
            min_snap_offset=args.min_snap_offset,
        ))

    if args.mode == "cohort2021":
        c21_ids = _cohort2021_players(rows)
        cohort_rows = [r for r in rows if r["player_id"] in c21_ids]
        print(f"\nScoring COHORT2021: {len(cohort_rows):,} players")
        all_out.extend(_score_cohort(
            cohort_tag="cohort2021",
            cohort_rows=cohort_rows,
            stats_by_pid=stats_by_pid,
            hazards=hazards,
            event_keys=event_keys,
            observe_through=args.observe_through,
            horizon=args.horizon,
            min_snap_offset=args.min_snap_offset,
        ))

    out_dir = os.path.dirname(args.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    long_path = f"{args.out_prefix}_long.csv"
    fieldnames = list(all_out[0].keys())
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_out)
    print(f"\nWrote {len(all_out):,} rows to {long_path}")

    summary_path = f"{args.out_prefix}_summary.txt"
    _write_summary(all_out, event_keys, summary_path, args.observe_through)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
