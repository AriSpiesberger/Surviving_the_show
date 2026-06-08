"""Score the FULL current prospect universe at snap=2026.

The OOF panel cache is capped at draft_year <= 2020 because TRAINING
needs realized outcomes (a 6-year forward window from 2020 -> 2026).
But INFERENCE should score every active prospect — including the 2021-
2025 draft classes that drive most of the buy-list signal.

This script:
  1. Loads every prospect from the DB (no draft-year cap)
  2. Loads all season_stats
  3. Loads the test hazards (default HP, 90% trained on 2005-2020 cohort)
  4. Scores snap=2026 across the entire current universe
  5. Writes a snap long CSV for the buy-list builder
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.storage import ProspectDB
from scripts_v17.train.train_v2_0b_prod import score_snap_with_landmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--hazards",
                    default=str(REPO_ROOT / "scratch" / "v20b_oof"
                                 / "hazards_full.pkl"))
    ap.add_argument("--snap-year", type=int, default=2026)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "scored"
                                          / "snap2026_v2.0b_full_long.csv"))
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading hazards {args.hazards}...")
    with open(args.hazards, "rb") as fh:
        hazards = pickle.load(fh)

    print(f"Loading EVERY prospect from {args.db} (no draft-year cap)...")
    db = ProspectDB(args.db)
    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory,
                   o.events_json, o.final_mlb_year
            FROM prospects p
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source "
                "FROM prospect_rankings").fetchall()
        except Exception:
            rank_rows = []

    stats_by_pid: dict = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    rankings_by_pid: dict = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for p in prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    n_draft = sum(1 for p in prospects if p.get("draft_year") is not None)
    n_ifa = sum(1 for p in prospects
                 if int(p.get("is_international") or 0) == 1)
    print(f"  {len(prospects):,} prospects total "
          f"(drafted {n_draft:,} + IFA {n_ifa:,})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = score_snap_with_landmark(
        hazards, prospects, stats_by_pid,
        snap_year=args.snap_year, out_csv=out,
        horizon=args.horizon, verbose=True,
    )
    print(f"\nWrote {out}: {n:,} rows in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
