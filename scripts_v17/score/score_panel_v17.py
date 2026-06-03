"""Score all current prospects at snap=<year> with v1.17 production hazards.

This is the upstream of build_v17_buylist.py:
   1. Query prospects DB for everyone eligible at snap_year
   2. Chunk into 100-player batches (segfault-resilient)
   3. Score each chunk with prospects.classifier.score_v14c_cal_slice_raw
      (uses event_classifiers_v1.17_prod.pkl + panel_v1.17.npz)
   4. Merge all chunk outputs into snap{year}_v17_all_long.csv

Usage:
    python -m scripts_v17.score.score_panel_v17 --snap-year 2026

Output:
    snap{snap-year}_v17_all_long.csv  (input to build_v17_buylist.py)

Knobs:
    --chunk-size       players per worker process (default 100)
    --max-tries        retries per chunk on segfault (default 3)
    --output-dir       where to put the intermediate chunks (default scratch)
    --hazards          override hazard pkl path
    --panel            override panel npz path
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd


def get_eligible_pids(snap_year: int, db: str) -> list[str]:
    """All prospects who could be at snap=snap_year and haven't already debuted by it."""
    c = sqlite3.connect(db)
    rows = c.execute("""
        SELECT DISTINCT p.player_id
        FROM prospects p
        LEFT JOIN career_outcomes o ON o.player_id = p.player_id
        WHERE (
            (p.draft_year IS NOT NULL AND p.draft_year BETWEEN 2010 AND ?)
            OR (COALESCE(p.is_international, 0) = 1)
        )
        AND (o.mlb_debut_year IS NULL OR o.mlb_debut_year > ?)
    """, (snap_year - 1, snap_year - 1)).fetchall()
    c.close()
    return [r[0] for r in rows]


def score_chunk(pid_file: Path, out_file: Path, hazards: str, panel: str,
                snap_year: int, max_tries: int) -> bool:
    """Score one chunk. Returns True on success."""
    for attempt in range(1, max_tries + 1):
        result = subprocess.run([
            sys.executable, "-m", "prospects.classifier.score_v14c_cal_slice_raw",
            "--model", hazards,
            "--panel", panel,
            "--players-file", str(pid_file),
            "--max-entry-year", str(snap_year - 1),
            "--observe-through", str(snap_year),
            "--max-offset", "16",
            "--out", str(out_file),
        ], env={**os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=300)
        if out_file.exists():
            return True
        if attempt < max_tries:
            print(f"    chunk {pid_file.name} try {attempt} failed (exit {result.returncode}), retrying...")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-year", type=int, default=2026)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--max-tries", type=int, default=3)
    ap.add_argument("--output-dir", default="scratch/v17_score_panel")
    ap.add_argument("--hazards", default="models/event_classifiers_v1.17_prod.pkl")
    ap.add_argument("--panel", default="panels/panel_v1.17.npz")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-chunks", action="store_true",
                    help="Don't delete intermediate chunk files after merge")
    args = ap.parse_args()

    snap = args.snap_year
    out_path = args.out or f"snap{snap}_v17_all_long.csv"
    work = Path(args.output_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "in").mkdir(exist_ok=True)
    (work / "out").mkdir(exist_ok=True)

    print(f"=== Score panel @ snap={snap} ===")
    print(f"  hazards: {args.hazards}")
    print(f"  panel:   {args.panel}")
    print(f"  db:      {args.db}")

    # Confirm artifacts exist
    for f in (args.hazards, args.panel, args.db):
        if not Path(f).exists():
            sys.exit(f"FATAL: missing {f}")

    pids = get_eligible_pids(snap, args.db)
    print(f"  candidates: {len(pids):,}")

    # Write chunks
    for i in range(0, len(pids), args.chunk_size):
        chunk_pids = pids[i:i + args.chunk_size]
        cf = work / "in" / f"chunk_{i//args.chunk_size:04d}.txt"
        if not cf.exists():
            cf.write_text("\n".join(chunk_pids) + "\n")
    chunk_files = sorted((work / "in").glob("chunk_*.txt"))
    print(f"  chunks:     {len(chunk_files)} (size={args.chunk_size})")

    # Score each chunk
    n_done = 0; n_skip = 0; n_fail = 0
    for cf in chunk_files:
        out_chunk = work / "out" / f"{cf.stem}.csv"
        if out_chunk.exists():
            n_skip += 1
            continue
        ok = score_chunk(cf, out_chunk, args.hazards, args.panel, snap, args.max_tries)
        if ok:
            n_done += 1
        else:
            print(f"    FAILED after {args.max_tries} tries: {cf.name}")
            n_fail += 1
        if (n_done + n_skip) % 25 == 0:
            print(f"    progress: {n_done + n_skip}/{len(chunk_files)} "
                  f"(skipped {n_skip}, failed {n_fail})")
    print(f"  done: {n_done} scored, {n_skip} cached, {n_fail} failed")

    if n_fail > 0:
        sys.exit(f"FATAL: {n_fail} chunks failed; aborting merge to avoid silent gaps")

    # Merge
    csvs = sorted((work / "out").glob("*.csv"))
    dfs = []
    for f in csvs:
        if f.stat().st_size < 200:  # header-only / empty
            continue
        try:
            d = pd.read_csv(f, encoding="utf-8")
            if len(d) > 0:
                dfs.append(d)
        except Exception as e:
            print(f"    warn: skipped {f}: {e}")
    if not dfs:
        sys.exit("FATAL: no chunk produced data")
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged[merged.snap_year == snap].drop_duplicates("player_id")
    merged.to_csv(out_path, index=False)
    print(f"  wrote {out_path}: {len(merged):,} unique players")

    if not args.keep_chunks:
        shutil.rmtree(work / "in", ignore_errors=True)
        shutil.rmtree(work / "out", ignore_errors=True)
        print(f"  cleaned intermediates under {work}")


if __name__ == "__main__":
    main()
