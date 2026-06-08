"""Re-score val with the average of all 6 OOF fold hazards.

Why: the original OOF run scored val with just the LAST fold's hazards
(arbitrary choice). The XGB was trained on stacked OOF rows where each
row was scored with a DIFFERENT fold's hazards — so the training set is
a mixture of 6 feature distributions, but val is from just one. That
mismatch hurts val AP across the board.

This script:
  1. Loads each fold's hazards.pkl
  2. Scores val pids with each fold separately → val_long_fold_k.csv
  3. Averages p_*, mean_t_*, sd_t_* across folds per (player, snap) cell
     (label columns are deterministic, unchanged)
  4. Writes results/training/v2.0b_oof_val_long.csv (overwrites the
     last-fold-only version)
  5. Optionally retrains the joint XGB

Resumable: each per-fold val score writes partials to
scratch/v20b_oof/val_partial_fold{k}/snap_NNNN.csv. Reruns skip done
snaps. Aggregation step is fast and idempotent.

Usage:
    python -m scripts_v17.train.rescore_val_avg_oof

    # skip the XGB refit (just rescore val):
    python -m scripts_v17.train.rescore_val_avg_oof --no-refit
"""
from __future__ import annotations

# Threading: respect any pre-set env vars. Was pinned to 1 during the
# BSOD window; the box is stable now.

import argparse
import gc
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from scripts_v17.train.run_v2_0b_oof import (
    PANEL_META, PANEL_NPZ, SCRATCH, VAL_PIDS_PATH, _score_checkpointed,
)

OOF_VAL = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"
OOF_STACKED = REPO_ROOT / "results" / "training" / "v2.0b_oof_stacked_long.csv"


def _load_panel():
    npz = np.load(PANEL_NPZ, allow_pickle=True)
    X_lm = npz["X_lm"]
    pids = npz["pids"].tolist()
    S_yrs = npz["S_yrs"].tolist()
    joined_idx = npz["joined_idx"]
    with PANEL_META.open("rb") as fh:
        meta = pickle.load(fh)
    prospects_list = meta["prospects"]
    stats_by_pid = meta["stats_by_pid"]
    return prospects_list, stats_by_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--no-refit", action="store_true",
                    help="Skip the XGB refit at the end")
    args = ap.parse_args()

    t0 = time.time()
    val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    print(f"[rescore] val pids: {len(val_pid_set):,}, K={args.k}")

    print(f"[rescore] loading panel meta (for prospects + stats)...")
    prospects_all, stats_by_pid = _load_panel()
    print(f"          {len(prospects_all):,} prospects")

    # Step 1: score val with each fold's hazards separately
    fold_csvs = []
    for k in range(args.k):
        hazards_pkl = SCRATCH / f"fold{k}_hazards.pkl"
        if not hazards_pkl.exists():
            sys.exit(f"FATAL: missing {hazards_pkl}")
        fold_val_csv = SCRATCH / f"val_long_fold{k}.csv"
        fold_csvs.append(fold_val_csv)
        if fold_val_csv.exists():
            print(f"[rescore] fold {k}: reusing {fold_val_csv.name}")
            continue
        print(f"\n[rescore] fold {k}: scoring val with "
              f"{hazards_pkl.name}", flush=True)
        with hazards_pkl.open("rb") as fh:
            hazards = pickle.load(fh)
        partial_dir = SCRATCH / f"val_partial_fold{k}"
        n = _score_checkpointed(
            hazards, prospects_all, stats_by_pid, val_pid_set,
            fold_val_csv, partial_dir,
            max_entry_year=args.max_entry_year,
            observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
        )
        print(f"[rescore] fold {k}: wrote {n:,} rows")
        del hazards
        gc.collect()

    # Step 2: average across folds per (player_id, snap_year)
    print(f"\n[rescore] averaging predictions across {args.k} folds...")
    dfs = [pd.read_csv(f) for f in fold_csvs]
    # All folds have identical labels/eligible/trigger columns.
    # Average only the score columns: p_*, mean_t_*, sd_t_*.
    base = dfs[0].copy()
    score_cols = [c for c in base.columns
                  if c.startswith("p_") or c.startswith("mean_t_")
                  or c.startswith("sd_t_")]
    print(f"          averaging {len(score_cols)} score columns")
    key = ["player_id", "snap_year", "snap_offset"]
    for col in score_cols:
        # Stack the col across folds, take mean (NaN-safe).
        stack = np.column_stack([df[col].values for df in dfs])
        base[col] = np.nanmean(stack, axis=1)
    # Sanity: realized/eligible/trigger columns should all match
    # across folds (deterministic from labels). Spot check one:
    for ev in ("MLB_DEBUT", "TOP_100_PROSPECT"):
        col = f"realized_{ev}"
        if col in base.columns:
            mismatches = sum((dfs[i][col] != dfs[0][col]).sum()
                             for i in range(1, len(dfs)))
            if mismatches > 0:
                print(f"  WARN: {col} mismatched across folds "
                      f"({mismatches} rows) — labels should be "
                      f"deterministic")
    # NaN-fill mean_t/sd_t (in case all folds NaN'd)
    for c in base.columns:
        if c.startswith("mean_t_"):
            base[c] = base[c].fillna(15.0)
        elif c.startswith("sd_t_"):
            base[c] = base[c].fillna(0.0)
    OOF_VAL.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(OOF_VAL, index=False)
    print(f"[rescore] wrote {OOF_VAL} ({len(base):,} rows)")

    # Step 3: refit XGB
    if args.no_refit:
        print(f"[rescore] --no-refit: skipping XGB fit")
    else:
        print(f"\n[rescore] retraining joint XGB on OOF stacked + "
              f"averaged val")
        tmp = str(XGB_OUT) + ".tmp"
        rc = subprocess.run([
            sys.executable, "-m", "scripts_v17.train.fit_joint_xgb_v2",
            "--fit", str(OOF_STACKED),
            "--val", str(OOF_VAL),
            "--db", str(REPO_ROOT / "prospects_snapshot.db"),
            "--out", tmp,
        ], cwd=REPO_ROOT).returncode
        if rc != 0:
            sys.exit(rc)
        Path(tmp).replace(XGB_OUT)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"  averaged val:  {OOF_VAL}")
    print(f"  XGB:           {XGB_OUT}")


if __name__ == "__main__":
    main()
