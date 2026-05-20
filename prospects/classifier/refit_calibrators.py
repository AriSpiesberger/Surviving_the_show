"""Refit Platt calibrators per event with an expanded calibration set.

Two changes from the default training pipeline:
  (a) Snapshot offsets expanded from (1,2,3,5) to (1,2,3,4,5,6,7,8).
      Each cal player contributes ~2x as many (snapshot, outcome) samples.
  (b) Calibration pool widened from val to val UNION test.
      Doubles the player pool used for the Platt fit.

Cost of (b): there is no longer a clean held-out slice for reporting.
A new holdout would need to be carved from the training pool for any
post-refit validation summary.

Hazard models are not retrained; only the per-event calibrator is
replaced. Saves a new pickle with the updated calibrators.

Usage:
    python -m prospects.classifier.refit_calibrators \\
        --in  models/event_classifiers_v1.5_star.pkl \\
        --out models/event_classifiers_v1.6_bigcal.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict

import numpy as np

from prospects.classifier.architectures.survival import (
    ELITE_KEY,
    EXIT_KEY,
    MAX_OBS_YEAR,
    STAR_KEY,
    _BetaCalibrator,
    _PlattCalibrator,
    _trigger_year,
    predict_cumulative_batch,
)
from prospects.storage import ProspectDB


def _player_pool_for_split(db: ProspectDB, max_draft_year: int,
                           min_year: int = 2005,
                           max_year: int = MAX_OBS_YEAR) -> list[str]:
    """Reproduce build_hazard_panel's player set without building the panel.

    A player is included iff:
      - drafted with draft_year <= max_draft_year, OR is_international=1, AND
      - has at least one (year) row in the panel range, i.e.
        start_year+1 <= max_year, where start_year is draft_year for drafted
        players or min(season_year) for IFAs.
    Returns the sorted unique player list, which matches the order used by
    fit_hazards for the seed-based split.
    """
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute(
            """SELECT p.player_id, p.draft_year, p.is_international
               FROM prospects p
               JOIN career_outcomes o ON o.player_id = p.player_id
               WHERE (p.draft_year IS NOT NULL AND p.draft_year <= ?)
                  OR COALESCE(p.is_international, 0) = 1""",
            (max_draft_year,),
        ).fetchall()]
        first_stat_year = {
            r["player_id"]: r["yr"] for r in conn.execute(
                """SELECT player_id, MIN(season_year) AS yr
                   FROM season_stats GROUP BY player_id"""
            ).fetchall() if r["yr"] is not None
        }
    pool: set[str] = set()
    for r in rows:
        pid = r["player_id"]
        dy = r["draft_year"]
        if dy is not None:
            start = int(dy)
        else:
            if pid not in first_stat_year:
                continue
            start = int(first_stat_year[pid])
        if max(start + 1, min_year) <= max_year:
            pool.add(pid)
    return sorted(pool)

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator


EXPANDED_OFFSETS = (1, 2, 3, 4, 5, 6, 7, 8)


def _ev_key(e):
    return e if isinstance(e, str) else int(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--in", dest="in_path",
                    default="models/event_classifiers_v1.5_star.pkl")
    ap.add_argument("--out",
                    default="models/event_classifiers_v1.6_bigcal.pkl")
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-year", type=int, default=MAX_OBS_YEAR)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", choices=["beta", "sigmoid"], default="beta",
                    help="Calibration family (default: beta, the v1.12+ "
                         "three-parameter generalization of Platt)")
    args = ap.parse_args()

    with open(args.in_path, "rb") as f:
        hazards = pickle.load(f)
    print(f"Loaded hazards: {list(hazards.keys())}")

    db = ProspectDB(args.db)
    unique_players = _player_pool_for_split(
        db, max_draft_year=args.max_draft_year, max_year=args.max_year,
    )
    print(f"Player pool (matches panel split): {len(unique_players):,}")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(unique_players))
    n_p = len(unique_players)
    n_test = int(round(0.10 * n_p))
    n_val = int(round(0.10 * n_p))
    test_players = set(unique_players[i] for i in perm[:n_test])
    val_players = set(unique_players[i] for i in perm[n_test:n_test + n_val])
    cal_players = val_players | test_players
    print(f"Calibration pool: val {len(val_players):,} + test {len(test_players):,} "
          f"= {len(cal_players):,} players (vs val-only {len(val_players):,})")

    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year, o.year_established_mlb, o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= 2018)
               OR COALESCE(p.is_international, 0) = 1
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    cal_rows = [r for r in rows if r["player_id"] in cal_players]
    print(f"Cal rows from DB (drafted<=2018 OR IFA): {len(cal_rows):,}")

    # Multi-snapshot expansion with the wider offset set.
    groups_by_year: dict[int, list[dict]] = defaultdict(list)
    for r in cal_rows:
        dy = r.get("draft_year")
        if dy is None:
            stat_yrs = [int(s["season_year"])
                        for s in stats_by_pid.get(r["player_id"], [])
                        if s.get("season_year") is not None]
            if not stat_yrs:
                continue
            start = min(stat_yrs)
        else:
            start = int(dy)
        for off in EXPANDED_OFFSETS:
            cur_year = start + off
            if cur_year >= args.max_year:
                continue
            groups_by_year[cur_year].append(r)
    total_samples = sum(len(g) for g in groups_by_year.values())
    print(f"Cal samples after expansion to offsets {EXPANDED_OFFSETS}: "
          f"{total_samples:,}")

    # Score every snapshot batch with the existing (untouched) hazards.
    # Chunk each year-group into batches to bound peak memory; larger groups
    # (with the expanded cal pool) were segfaulting predict_cumulative_batch.
    import gc
    CHUNK = 400
    score_keys = [e for e in hazards if e != EXIT_KEY]
    # Preallocate numpy buffers; Python list-of-floats was bloating heap.
    per_event_preds: dict = {
        _ev_key(e): np.empty(total_samples, dtype=np.float32)
        for e in score_keys
    }
    per_event_real: dict = {
        _ev_key(e): np.empty(total_samples, dtype=np.int8)
        for e in score_keys
    }
    write_idx = 0
    total_done = 0
    for cur_year, group in sorted(groups_by_year.items()):
        for start in range(0, len(group), CHUNK):
            chunk = group[start:start + CHUNK]
            sub_stats = {r["player_id"]: stats_by_pid.get(r["player_id"], [])
                         for r in chunk}
            out = predict_cumulative_batch(
                hazards, chunk, sub_stats,
                current_year=cur_year, horizon=args.horizon,
            )
            m = len(chunk)
            sl = slice(write_idx, write_idx + m)
            for ev in score_keys:
                k = _ev_key(ev)
                raw_arr = out.get(("raw", ev))
                if raw_arr is None:
                    raw_arr = out.get(ev)
                per_event_preds[k][sl] = np.asarray(raw_arr, dtype=np.float32)
                labels = np.fromiter(
                    (int((t := _trigger_year(r, ev)) is not None
                         and t <= args.max_year) for r in chunk),
                    dtype=np.int8, count=m,
                )
                per_event_real[k][sl] = labels
            write_idx += m
            total_done += m
            del out, sub_stats
            gc.collect()
        print(f"  scored {total_done:,}/{total_samples:,}", flush=True)

    print(f"\nRefitting Platt calibrators per event:")
    print(f"{'event':<22} {'n_samples':>10} {'pos':>5} {'a':>9} {'b':>9} "
          f"{'top10_obs':>10}")
    print("-" * 70)
    from prospects.schema import CareerEvent
    for ev in score_keys:
        k = _ev_key(ev)
        preds_a = per_event_preds[k].astype(np.float64)
        labels_a = per_event_real[k]
        pos = int(labels_a.sum())
        if isinstance(ev, str):
            ename = ev
        elif hasattr(ev, "name"):
            ename = ev.name
        else:
            try:
                ename = CareerEvent(ev).name
            except Exception:
                ename = str(ev)
        if pos < 3 or pos > len(labels_a) - 3:
            print(f"{str(ename):<22} {len(preds_a):>10,d} {pos:>5d}  "
                  f"skip (too few/many positives)")
            continue
        if args.method == "sigmoid":
            cal = _PlattCalibrator().fit(preds_a, labels_a)
        else:
            cal = _BetaCalibrator().fit(preds_a, labels_a)
        hazards[ev]["calibrator"] = cal
        order = np.argsort(preds_a)[::-1]
        top10 = order[: max(1, len(preds_a) // 10)]
        obs10 = float(labels_a[top10].mean())
        # Show params: a,b for Platt; a,b,c for Beta
        if hasattr(cal, 'c'):
            print(f"{str(ename):<22} {len(preds_a):>10,d} {pos:>5d} "
                  f"a={cal.a:>7.3f} b={cal.b:>7.3f} c={cal.c:>7.3f} "
                  f"top10_obs={obs10:>6.3f}")
        else:
            print(f"{str(ename):<22} {len(preds_a):>10,d} {pos:>5d} "
                  f"a={cal.a:>7.3f} b={cal.b:>7.3f} "
                  f"top10_obs={obs10:>6.3f}")

    with open(args.out, "wb") as f:
        pickle.dump(hazards, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
