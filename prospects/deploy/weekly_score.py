"""Weekly v1.17 scoring + buy list build.

Pipeline:
  1. scripts_v17/score/score_panel_v17.py  → snap{season}_v17_all_long.csv
  2. scripts_v17/buylist/build_v17_buylist.py  → buy_list_v1.17_FINAL.csv

Idempotent — score_panel_v17 caches per-chunk outputs in scratch/v17_score_panel
so reruns only fill gaps. build_v17_buylist always rewrites from the merged
long file.

Usage:
    python -m prospects.deploy.weekly_score --season 2026

    # Force chunk-cache wipe before rescoring (when hazards change):
    python -m prospects.deploy.weekly_score --season 2026 --fresh

    # Run scoring only (skip buy-list rebuild):
    python -m prospects.deploy.weekly_score --season 2026 --score-only

    # Run buy-list rebuild only (assume snap CSV already exists):
    python -m prospects.deploy.weekly_score --season 2026 --buylist-only

Exit codes:
    0 = success
    1 = scoring failed
    2 = buylist build failed
    3 = required artifacts missing
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Required artifacts. If any missing, abort early (3) — better than a half-run
# that emails a stale buy list.
REQUIRED = [
    "models/event_classifiers_v1.17_prod.pkl",
    "models/debut_lasso_universe_v1.17h.pkl",
    "models/top100_lasso_v1.17h.pkl",
    "models/model_b_outcomes_v1.17h.pkl",
    "models/player_position_from_stats.csv",
    "panels/panel_v1.17.npz",
    "prospects_snapshot.db",
    "scripts_v17/score/score_panel_v17.py",
    "scripts_v17/buylist/build_v17_buylist.py",
]


def check_artifacts() -> list[str]:
    missing = []
    for rel in REQUIRED:
        if not (REPO_ROOT / rel).exists():
            missing.append(rel)
    return missing


def run_step(label: str, cmd: list[str], cwd: Path) -> int:
    """Run a subprocess, stream its output. Return exit code."""
    print(f"\n{'='*70}\n[{label}] {' '.join(cmd)}\n{'='*70}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env={
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    print(f"\n[{label}] exit={proc.returncode}", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True,
                    help="snap year (e.g. 2026)")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe scratch/v17_score_panel before scoring (forces "
                         "full re-score; use after hazard model updates)")
    ap.add_argument("--score-only", action="store_true",
                    help="run only the scoring step")
    ap.add_argument("--buylist-only", action="store_true",
                    help="run only the buylist build step")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="don't clean intermediate per-chunk CSVs after merge")
    args = ap.parse_args()

    if args.score_only and args.buylist_only:
        sys.exit("--score-only and --buylist-only are mutually exclusive")

    print(f"=== weekly_score for season={args.season} ===")

    # Refresh prospects_snapshot.db from the live prospects.db so the buy-list
    # build sees today's MiLB stats. daily_data.py writes to prospects.db;
    # build_v17_buylist.py reads from prospects_snapshot.db. This bridge keeps
    # the weekly buy list current. Idempotent.
    live_db = REPO_ROOT / "prospects.db"
    snap_db = REPO_ROOT / "prospects_snapshot.db"
    if live_db.exists():
        print(f"[snapshot] copying {live_db.name} -> {snap_db.name}")
        shutil.copy2(live_db, snap_db)
    else:
        print(f"[snapshot] WARN: {live_db.name} not found; "
              f"using existing {snap_db.name} if present")

    missing = check_artifacts()
    if missing:
        print(f"\nFATAL: required artifacts missing:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(3)

    snap_long = REPO_ROOT / f"snap{args.season}_v17_all_long.csv"
    scratch_dir = REPO_ROOT / "scratch" / "v17_score_panel"

    # Step 1: score panel
    if not args.buylist_only:
        if args.fresh and scratch_dir.exists():
            print(f"[score] wiping {scratch_dir}")
            shutil.rmtree(scratch_dir)
        score_cmd = [
            sys.executable, "-m", "scripts_v17.score.score_panel_v17",
            "--snap-year", str(args.season),
            "--output-dir", str(scratch_dir),
            "--out", str(snap_long),
            "--panel", str(REPO_ROOT / "panels" / "panel_v1.17.npz"),
        ]
        if args.keep_chunks:
            score_cmd.append("--keep-chunks")
        rc = run_step("score", score_cmd, REPO_ROOT)
        if rc != 0:
            sys.exit(1)
        if not snap_long.exists():
            print(f"FATAL: score step exited 0 but {snap_long.name} not written")
            sys.exit(1)

    # Step 2: build buy list
    if not args.score_only:
        if not snap_long.exists():
            print(f"FATAL: need {snap_long.name} but it doesn't exist; "
                  f"run without --buylist-only first")
            sys.exit(3)
        # build_v17_buylist.py reads snap2026_v17_all_long.csv from cwd
        # (its current default). Symlink-or-copy if season differs:
        canonical = REPO_ROOT / f"snap{args.season}_v17_all_long.csv"
        expected = REPO_ROOT / "snap2026_v17_all_long.csv"
        if args.season != 2026 and canonical != expected:
            shutil.copy2(canonical, expected)
            print(f"[buylist] copied {canonical.name} -> {expected.name}")
        rc = run_step("buylist", [
            sys.executable, str(REPO_ROOT / "scripts_v17" / "buylist" / "build_v17_buylist.py"),
        ], REPO_ROOT)
        if rc != 0:
            sys.exit(2)

    print(f"\n=== weekly_score season={args.season} OK ===")
    print(f"  snap long file: {snap_long}")
    print(f"  buy list:       {REPO_ROOT / 'buy_list_v1.17_FINAL.csv'}")
    print(f"  all scored:     {REPO_ROOT / 'buy_list_v1.17_ALL_SCORED.csv'}")


if __name__ == "__main__":
    main()
