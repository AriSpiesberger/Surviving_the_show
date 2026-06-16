"""Build the landmark panel cache with tiny, resumable chunks.

Each chunk processes a small window of landmark rows and IMMEDIATELY
saves to disk, then the loop continues. If the process dies (or the
whole system crashes with a hypervisor error), every chunk written
before the crash is preserved on disk — just re-run this script and
it'll resume from where it left off.

Usage:
    python -m scripts_v17.train.build_landmark_panel_cache

    # smaller chunks if 5000 still triggers the crash:
    python -m scripts_v17.train.build_landmark_panel_cache --chunk-rows 2000

    # force rebuild from scratch:
    python -m scripts_v17.train.build_landmark_panel_cache --force

Pipeline:
  Step 1 — load DB, build plan (pids, S_yrs, prospect_idx), save:
             scratch/v20b_oof/panel_plan.npz
             scratch/v20b_oof/panel_meta.pkl
  Step 2 — for each chunk c of CHUNK_ROWS rows:
             if scratch/v20b_oof/chunks/panel_chunk_NNN.npz exists -> skip
             else build X_chunk, save it, free memory, tqdm bar updates
  Step 3 — merge all chunks into scratch/v20b_oof/panel_cache.npz

Working set per chunk is ~5 MB at chunk_rows=5000. Even on a system that
crashes at higher memory pressure, this should fit comfortably.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "LOKY_MAX_CPU_COUNT"):
    os.environ[_k] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures.landmark_survival import (
    N_FEATURES, _start_year,
)
from prospects.classifier.architectures.survival import (
    MAX_OBS_YEAR, build_windowed_features,
)
from prospects.storage import ProspectDB

CACHE_DIR = REPO_ROOT / "scratch" / "v20b_oof"
CHUNKS_DIR = CACHE_DIR / "chunks"
PLAN_NPZ = CACHE_DIR / "panel_plan.npz"
META_PKL = CACHE_DIR / "panel_meta.pkl"
PANEL_NPZ = CACHE_DIR / "panel_cache.npz"


def step1_prep(db_path: str, max_draft_year: int) -> int:
    """Return n_rows in the plan."""
    if PLAN_NPZ.exists() and META_PKL.exists():
        plan = np.load(PLAN_NPZ, allow_pickle=True)
        n_rows = int(plan["n_rows"])
        print(f"[Step 1] reusing plan ({n_rows:,} rows) + meta")
        return n_rows

    t = time.time()
    print(f"[Step 1] loading DB + building plan...")
    db = ProspectDB(db_path)
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
               OR (p.draft_year IS NULL
                    AND COALESCE(p.is_international, 0) = 0
                    AND p.player_id IN (SELECT DISTINCT player_id
                                        FROM season_stats))
        """, (max_draft_year,)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source "
                "FROM prospect_rankings").fetchall()
        except Exception:
            rank_rows = []
        try:
            # v2.0c: TBC org (team-level) rankings. as_of is 'YYYY-01-01'.
            org_rank_rows = conn.execute(
                "SELECT player_id, as_of, org_rank FROM rankings_history "
                "WHERE org_rank IS NOT NULL").fetchall()
        except Exception:
            org_rank_rows = []

    stats_by_pid: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    rankings_by_pid: dict[str, list[tuple[int, int, str]]] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for p in prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    # v2.0c: attach TBC org rankings as (year, org_rank) tuples.
    org_rankings_by_pid: dict[str, list[tuple[int, int]]] = {}
    for r in org_rank_rows:
        try:
            yr = int(str(r[1])[:4])
        except (ValueError, TypeError):
            continue
        org_rankings_by_pid.setdefault(r[0], []).append((yr, int(r[2])))
    for p in prospects:
        p["_org_rankings"] = org_rankings_by_pid.get(p["player_id"], [])
    n_with_org = sum(1 for p in prospects if p["_org_rankings"])
    print(f"         {n_with_org:,} prospects have >=1 org ranking")

    n_draft = sum(1 for p in prospects if p.get("draft_year") is not None)
    print(f"         {len(prospects):,} prospects "
          f"(drafted {n_draft:,} + IFA {len(prospects)-n_draft:,})")

    pids_list: list[str] = []
    S_list: list[int] = []
    prospect_idx_list: list[int] = []
    n_skipped = 0
    for idx, p in enumerate(prospects):
        sy = _start_year(p, stats_by_pid)
        if sy is None:
            n_skipped += 1
            continue
        lo = max(sy + 1, 2007)
        hi = MAX_OBS_YEAR - 1
        if lo > hi:
            continue
        for S in range(lo, hi + 1):
            pids_list.append(p["player_id"])
            S_list.append(S)
            prospect_idx_list.append(idx)
    n_rows = len(pids_list)
    print(f"         {n_rows:,} landmark rows planned "
          f"({n_skipped} skipped)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(str(PLAN_NPZ),
             pids=np.array(pids_list, dtype=object),
             S_yrs=np.array(S_list, dtype=np.int32),
             prospect_idx=np.array(prospect_idx_list, dtype=np.int32),
             n_rows=np.int64(n_rows))

    with META_PKL.open("wb") as fh:
        pickle.dump({"prospects": prospects,
                     "stats_by_pid": stats_by_pid},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"         wrote {PLAN_NPZ.name} + {META_PKL.name} "
          f"in {time.time()-t:.0f}s")
    return n_rows


def step2_chunks(n_rows: int, chunk_rows: int, start_chunk: int):
    """Build per-chunk feature matrices. Resumable; each chunk written
    atomically before moving on."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    width = max(3, len(str(n_chunks)))

    print(f"[Step 2] {n_chunks} chunks of <= {chunk_rows:,} rows  "
          f"(starting at chunk {start_chunk})")
    plan = np.load(PLAN_NPZ, allow_pickle=True)
    S_yrs = plan["S_yrs"]
    prospect_idx = plan["prospect_idx"]
    with META_PKL.open("rb") as fh:
        meta = pickle.load(fh)
    prospects = meta["prospects"]
    stats_by_pid = meta["stats_by_pid"]

    done = 0
    skipped = 0
    rebuilt = 0
    pbar = tqdm(range(start_chunk, n_chunks), desc="chunks",
                unit="chunk", mininterval=0.5)
    for c in pbar:
        out = CHUNKS_DIR / f"panel_chunk_{c:0{width}d}.npz"
        if out.exists():
            try:
                with np.load(out) as z:
                    _ = z["X_chunk"].shape  # force read
                skipped += 1
                pbar.set_postfix(done=done, skipped=skipped,
                                 rebuilt=rebuilt)
                continue
            except Exception:
                out.unlink(missing_ok=True)
                rebuilt += 1

        lo = c * chunk_rows
        hi = min(lo + chunk_rows, n_rows)
        size = hi - lo
        X_chunk = np.empty((size, N_FEATURES), dtype=np.float32)
        for j in range(size):
            i = lo + j
            p = prospects[int(prospect_idx[i])]
            stats = stats_by_pid.get(p["player_id"], [])
            S = int(S_yrs[i])
            X_chunk[j, :] = build_windowed_features(
                p, stats, S, milb_only=True)

        np.savez(str(out), X_chunk=X_chunk,
                 lo=np.int64(lo), hi=np.int64(hi))
        del X_chunk
        gc.collect()
        done += 1
        pbar.set_postfix(done=done, skipped=skipped, rebuilt=rebuilt)

    print(f"[Step 2] done — {done} chunks written, {skipped} skipped")
    return n_chunks, width


def step3_merge(n_rows: int, n_chunks: int, width: int):
    if PANEL_NPZ.exists():
        print(f"[Step 3] {PANEL_NPZ.name} already exists, skipping merge")
        return

    print(f"[Step 3] merging {n_chunks} chunks -> {PANEL_NPZ.name}")
    X_lm = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    filled = 0
    for c in tqdm(range(n_chunks), desc="merge", unit="chunk"):
        out = CHUNKS_DIR / f"panel_chunk_{c:0{width}d}.npz"
        chunk = np.load(out)
        lo = int(chunk["lo"]); hi = int(chunk["hi"])
        X_lm[lo:hi, :] = chunk["X_chunk"]
        filled += (hi - lo)
    assert filled == n_rows, f"coverage gap: {filled} != {n_rows}"

    plan = np.load(PLAN_NPZ, allow_pickle=True)
    np.savez_compressed(str(PANEL_NPZ), X_lm=X_lm,
                        pids=plan["pids"],
                        S_yrs=plan["S_yrs"],
                        joined_idx=plan["prospect_idx"])
    print(f"         {PANEL_NPZ}: "
          f"{PANEL_NPZ.stat().st_size/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--chunk-rows", type=int, default=5000,
                    help="Rows per chunk. Smaller = lower peak memory.")
    ap.add_argument("--start-chunk", type=int, default=0,
                    help="Resume from chunk index (on-disk chunks are "
                         "auto-skipped regardless).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if PANEL_NPZ.exists() and META_PKL.exists() and not args.force:
        print(f"Panel cache already exists ({PANEL_NPZ.name}) — done.")
        return 0

    if args.force:
        for f in [PLAN_NPZ, META_PKL, PANEL_NPZ]:
            if f.exists():
                f.unlink()
        if CHUNKS_DIR.exists():
            for f in CHUNKS_DIR.glob("panel_chunk_*.npz"):
                f.unlink()

    t0 = time.time()
    n_rows = step1_prep(args.db, args.max_draft_year)
    n_chunks, width = step2_chunks(n_rows, args.chunk_rows, args.start_chunk)
    step3_merge(n_rows, n_chunks, width)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"  cache: {PANEL_NPZ}")
    print(f"  meta:  {META_PKL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
