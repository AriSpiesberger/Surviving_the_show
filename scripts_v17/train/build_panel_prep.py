"""Phase 1 of chunked panel build: load DB, build plan, save metadata.

Saves:
  scratch/v20b_oof/panel_plan.npz  -- pids, S_yrs, joined_idx (487k each)
  scratch/v20b_oof/panel_meta.pkl  -- prospects list + stats_by_pid

Phase 2 (build_panel_chunk.py) reads these to compute X_lm slices.
Phase 3 (merge in build_landmark_panel_cache.py) concats slices.

Runs in ~30s typically — fast enough that OS process kills don't get
their teeth into it.
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures.landmark_survival import _start_year
from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from prospects.storage import ProspectDB

CACHE_DIR = REPO_ROOT / "scratch" / "v20b_oof"
PLAN_NPZ = CACHE_DIR / "panel_plan.npz"
META_PKL = CACHE_DIR / "panel_meta.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--min-landmark-year", type=int, default=2007)
    ap.add_argument("--max-landmark-year", type=int,
                    default=MAX_OBS_YEAR - 1)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    db = ProspectDB(args.db)
    print(f"Loading prospects + stats from DB...", flush=True)
    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory,
                   o.events_json, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= ?)
               OR COALESCE(p.is_international, 0) = 1
        """, (args.max_draft_year,)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source "
                "FROM prospect_rankings").fetchall()
        except Exception:
            rank_rows = []

    stats_by_pid: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    rankings_by_pid: dict[str, list[tuple[int, int, str]]] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for p in prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    n_draft = sum(1 for p in prospects if p.get("draft_year") is not None)
    n_ifa = len(prospects) - n_draft
    print(f"{len(prospects):,} prospects (drafted {n_draft:,} + "
          f"IFA {n_ifa:,})  landmark range "
          f"{args.min_landmark_year}..{args.max_landmark_year}", flush=True)

    print("Building plan...", flush=True)
    pids_list: list[str] = []
    S_list: list[int] = []
    prospect_idx_list: list[int] = []
    n_skipped = 0
    for idx, p in enumerate(prospects):
        stats = stats_by_pid.get(p["player_id"], [])
        sy = _start_year(p, stats_by_pid)
        if sy is None:
            n_skipped += 1
            continue
        lo = max(sy + 1, args.min_landmark_year)
        hi = args.max_landmark_year
        if lo > hi:
            continue
        for S in range(lo, hi + 1):
            pids_list.append(p["player_id"])
            S_list.append(S)
            prospect_idx_list.append(idx)
    n_rows = len(pids_list)
    print(f"{n_rows:,} landmark rows planned "
          f"({n_skipped} prospects skipped, no start_year)", flush=True)

    tmp_npz = PLAN_NPZ.with_suffix(".npz.tmp")
    np.savez(tmp_npz,
             pids=np.array(pids_list, dtype=object),
             S_yrs=np.array(S_list, dtype=np.int32),
             prospect_idx=np.array(prospect_idx_list, dtype=np.int32),
             n_rows=np.int64(n_rows))
    tmp_npz.replace(PLAN_NPZ)
    print(f"Wrote {PLAN_NPZ.name}", flush=True)

    tmp_pkl = META_PKL.with_suffix(".pkl.tmp")
    with tmp_pkl.open("wb") as fh:
        pickle.dump({
            "prospects": prospects,
            "stats_by_pid": stats_by_pid,
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_pkl.replace(META_PKL)
    print(f"Wrote {META_PKL.name} ({META_PKL.stat().st_size/1e6:.0f} MB)",
          flush=True)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
