"""v2.0b OOF — K-fold out-of-fold stacked XGB.

SERIAL subprocess-per-fold orchestration. Each fold spawns a fresh Python
process that builds the panel, trains hazards on the train_pids, scores
the heldout pids, writes the CSV, and exits. Memory is fully released
between folds — critical because the panel + hazards + stats_by_pid each
hold ~500MB-1GB of working set and you can't hold K of them at once.

Pipeline (K = 6 by default):

  Stage A — partition the 90% universe (entry <= 2020, excluding val pids)
            into K folds (~15% per fold). Write each fold's pid list to disk.

  Stage B — for each fold k (one subprocess):
            train_v2_0b_oof_fold.py:
              build panel -> train hazards on K-1 folds (~75%)
              -> score heldout fold k -> write CSV -> exit
            Memory fully released before fold k+1 starts.

  Stage C — stack the K fold CSVs.
            -> results/training/v2.0b_oof_stacked_long.csv

  Stage D — score the val pids using fold-K-1's hazards (val was excluded
            from every fold's training).
            -> results/training/v2.0b_oof_val_long.csv

  Stage E — fit_joint_xgb_v2 on the stacked OOF (fit) + val OOF (val).
            -> models/joint_xgb_v2.0b_oof.pkl

Each subprocess takes the time of one panel build (~5 min) + one hazards
fit (~5-10 min) + scoring its fold (~3 min) = ~15-20 min per fold.
K=6 -> total ~90-120 min, plus ~10 min for stack + val + XGB.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from prospects.storage import ProspectDB

VAL_PIDS_PATH = REPO_ROOT / "results" / "training" / "v17_prod_val_pids.txt"
OOF_SCRATCH = REPO_ROOT / "scratch" / "v20b_oof"
TRAIN_DIR = REPO_ROOT / "results" / "training"
OOF_STACKED_LONG = TRAIN_DIR / "v2.0b_oof_stacked_long.csv"
OOF_VAL_LONG = TRAIN_DIR / "v2.0b_oof_val_long.csv"
XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"


def _entry_year(player: dict, stats_by_pid: dict) -> int | None:
    dy = player.get("draft_year")
    if dy is not None and int(player.get("is_international") or 0) == 0:
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


def _build_universe(db_path: str, max_entry_year: int, max_draft_year: int,
                    val_pids: set[str]) -> list[str]:
    """Return sorted list of pids in the OOF universe:
    entry <= max_entry_year, not in val, drafted or IFA."""
    db = ProspectDB(db_path)
    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.* FROM prospects p
            WHERE (p.draft_year IS NOT NULL
                    AND p.draft_year BETWEEN 2007 AND ?)
               OR COALESCE(p.is_international, 0) = 1
        """, (max_draft_year,)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s); stats_by_pid.setdefault(d["player_id"], []).append(d)

    universe = []
    for p in prospects:
        if p["player_id"] in val_pids:
            continue
        ey = _entry_year(p, stats_by_pid)
        if ey is None or ey > max_entry_year:
            continue
        universe.append(p["player_id"])
    return sorted(set(universe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--skip-folds", action="store_true",
                    help="Reuse existing fold CSVs in scratch/v20b_oof/")
    ap.add_argument("--skip-val-score", action="store_true")
    ap.add_argument("--skip-xgb", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    OOF_SCRATCH.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"v2.0b OOF (SERIAL subprocess-per-fold)  K={args.k}  "
          f"seed={args.seed}")
    print("=" * 78)

    val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    print(f"[OOF] Val pids held out: {len(val_pid_set):,}", flush=True)

    # --- Stage 0: build panel cache ONCE (with retries) ---
    panel_npz = OOF_SCRATCH / "panel_cache.npz"
    panel_meta = OOF_SCRATCH / "panel_meta.pkl"
    if not (panel_npz.exists() and panel_meta.exists()):
        print(f"\n[OOF] Building panel cache (one-time, ~25 min)...",
              flush=True)
        for attempt in range(1, 5):
            print(f"[OOF] panel-cache attempt {attempt}/4", flush=True)
            rc = subprocess.run([
                sys.executable, "-m",
                "scripts_v17.train.build_landmark_panel_cache",
                "--db", args.db,
                "--max-draft-year", str(args.max_draft_year),
            ], cwd=REPO_ROOT).returncode
            if rc == 0 and panel_npz.exists() and panel_meta.exists():
                break
            print(f"[OOF] panel-cache attempt {attempt} failed "
                  f"(exit {rc}). Sleeping 30s...", flush=True)
            time.sleep(30)
        if not (panel_npz.exists() and panel_meta.exists()):
            print(f"[OOF] panel-cache build failed after 4 attempts.",
                  flush=True)
            sys.exit(1)
        print(f"[OOF] panel cache built. Fold workers will load it.",
              flush=True)
    else:
        print(f"[OOF] reusing existing panel cache "
              f"{panel_npz.name}", flush=True)

    # --- Stage A: partition the universe + write pid lists to disk ---
    fold_pid_files = [OOF_SCRATCH / f"fold{k}_pids.txt"
                      for k in range(args.k)]
    train_pid_files = [OOF_SCRATCH / f"train{k}_pids.txt"
                       for k in range(args.k)]
    if not all(f.exists() for f in fold_pid_files + train_pid_files):
        print(f"[OOF] Partitioning universe into {args.k} folds...",
              flush=True)
        universe = _build_universe(args.db, args.max_entry_year,
                                    args.max_draft_year, val_pid_set)
        print(f"[OOF] Universe: {len(universe):,} pids", flush=True)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(universe))
        fold_arrays = np.array_split(perm, args.k)
        fold_sets = [set(universe[i] for i in arr) for arr in fold_arrays]
        for k in range(args.k):
            fold_pid_files[k].write_text(
                "\n".join(sorted(fold_sets[k])) + "\n")
            train_pids = set()
            for j in range(args.k):
                if j != k:
                    train_pids |= fold_sets[j]
            train_pid_files[k].write_text(
                "\n".join(sorted(train_pids)) + "\n")
            print(f"[OOF] fold {k}: heldout={len(fold_sets[k]):,}, "
                  f"train={len(train_pids):,}", flush=True)
    else:
        print(f"[OOF] reusing existing fold/train pid files", flush=True)

    # --- Stage B: serial subprocess per fold ---
    fold_csv_files = [OOF_SCRATCH / f"fold{k}_long.csv"
                      for k in range(args.k)]
    for k in range(args.k):
        out_csv = fold_csv_files[k]
        if args.skip_folds and out_csv.exists():
            print(f"[fold {k}] skip-folds: {out_csv.name} exists",
                  flush=True)
            continue
        print(f"\n{'#'*70}\n# FOLD {k+1}/{args.k}  ({out_csv.name})\n"
              f"{'#'*70}", flush=True)
        t_fold = time.time()
        max_tries = 4
        rc = -1
        for attempt in range(1, max_tries + 1):
            print(f"[OOF] fold {k} attempt {attempt}/{max_tries}",
                  flush=True)
            rc = subprocess.run([
                sys.executable, "-m",
                "scripts_v17.train.train_v2_0b_oof_fold",
                "--db", args.db,
                "--train-pids", str(train_pid_files[k]),
                "--fold-pids", str(fold_pid_files[k]),
                "--out", str(out_csv),
                "--max-draft-year", str(args.max_draft_year),
                "--max-entry-year", str(args.max_entry_year),
                "--seed", str(args.seed),
            ], cwd=REPO_ROOT).returncode
            if rc == 0 and out_csv.exists():
                break
            print(f"[OOF] fold {k} attempt {attempt} failed (exit {rc}). "
                  f"Sleeping 30s before retry...", flush=True)
            time.sleep(30)
        if rc != 0 or not out_csv.exists():
            print(f"[OOF] fold {k} failed after {max_tries} attempts. "
                  f"Aborting.", flush=True)
            sys.exit(rc or 1)
        print(f"[OOF] fold {k} done in {(time.time()-t_fold)/60:.1f} min",
              flush=True)

    # --- Stage C: stack ---
    print(f"\n[OOF] Stacking {args.k} fold CSVs -> {OOF_STACKED_LONG.name}",
          flush=True)
    dfs = [pd.read_csv(f) for f in fold_csv_files]
    stacked = pd.concat(dfs, ignore_index=True)
    stacked.to_csv(OOF_STACKED_LONG, index=False)
    print(f"[OOF] stacked: {len(stacked):,} rows, "
          f"{stacked.player_id.nunique():,} players", flush=True)

    # --- Stage D: score val using fold-K-1's hazards (re-train one more
    #              time via the fold worker — small cost vs. carrying state) ---
    if args.skip_val_score and OOF_VAL_LONG.exists():
        print(f"[OOF] skip-val-score: reusing {OOF_VAL_LONG.name}",
              flush=True)
    else:
        # Reuse fold-K-1's train_pids as the "hazards saw 75% of universe"
        # source, then score the val pids.
        val_pid_file = OOF_SCRATCH / "val_pids.txt"
        val_pid_file.write_text("\n".join(sorted(val_pid_set)) + "\n")
        print(f"\n[OOF] Scoring {len(val_pid_set):,} val pids with "
              f"fold-{args.k-1} train hazards (val was held out)",
              flush=True)
        rc = subprocess.run([
            sys.executable, "-m",
            "scripts_v17.train.train_v2_0b_oof_fold",
            "--db", args.db,
            "--train-pids", str(train_pid_files[args.k - 1]),
            "--fold-pids", str(val_pid_file),
            "--out", str(OOF_VAL_LONG),
            "--max-draft-year", str(args.max_draft_year),
            "--max-entry-year", str(args.max_entry_year),
            "--seed", str(args.seed),
        ], cwd=REPO_ROOT).returncode
        if rc != 0:
            sys.exit(rc)

    # --- Stage E: fit joint XGB ---
    if args.skip_xgb and XGB_OUT.exists():
        print(f"[OOF] skip-xgb: using existing {XGB_OUT.name}", flush=True)
    else:
        tmp = str(XGB_OUT) + ".tmp"
        print(f"\n[OOF] Training joint XGB on stacked OOF -> "
              f"{XGB_OUT.name}", flush=True)
        rc = subprocess.run([
            sys.executable, "-m", "scripts_v17.train.fit_joint_xgb_v2",
            "--fit", str(OOF_STACKED_LONG),
            "--val", str(OOF_VAL_LONG),
            "--db", args.db,
            "--out", tmp,
        ], cwd=REPO_ROOT).returncode
        if rc != 0:
            sys.exit(rc)
        shutil.move(tmp, XGB_OUT)

    print(f"\n=== v2.0b OOF COMPLETE in {(time.time()-t_start)/60:.1f} min "
          f"===")
    print(f"  fold csvs: {OOF_SCRATCH}/fold[0..{args.k-1}]_long.csv")
    print(f"  stacked:   {OOF_STACKED_LONG}")
    print(f"  val:       {OOF_VAL_LONG}")
    print(f"  XGB:       {XGB_OUT}")


if __name__ == "__main__":
    main()
