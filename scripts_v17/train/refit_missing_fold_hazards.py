"""Refit hazards for any fold whose hazards.pkl is missing.

Folds 0-2 were trained before hazards-pkl caching was added, so only
their long CSVs exist. The averaged-val rescore needs all K hazards
pkls to score val K times.

This script:
  1. Loads panel cache
  2. For each fold k in 0..K-1 whose hazards pkl is missing:
       reconstruct train_mask from train{k}_pids.txt
       fit_landmark_hazards on that mask
       save fold{k}_hazards.pkl (atomic via .tmp + replace)
  3. Exits

Usage:
    python -m scripts_v17.train.refit_missing_fold_hazards
"""
from __future__ import annotations

# Threading: respect any pre-set env vars. Was pinned to 1 during the
# BSOD window; the box is stable now.

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from scripts_v17.train.run_v2_0b_oof import (
    PANEL_META, PANEL_NPZ, SCRATCH,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()

    # Check which fold pkls are missing
    missing = []
    for k in range(args.k):
        if not (SCRATCH / f"fold{k}_hazards.pkl").exists():
            missing.append(k)
    if not missing:
        print(f"All {args.k} hazards pkls already exist.")
        return 0
    print(f"Missing hazards: {missing}")

    # Load panel
    print(f"Loading panel cache...")
    npz = np.load(PANEL_NPZ, allow_pickle=True)
    X_lm = npz["X_lm"]
    pids = npz["pids"].tolist()
    S_yrs = npz["S_yrs"].tolist()
    joined_idx = npz["joined_idx"]
    with PANEL_META.open("rb") as fh:
        meta = pickle.load(fh)
    prospects_list = meta["prospects"]
    stats_by_pid = meta["stats_by_pid"]
    joined = [prospects_list[i] for i in joined_idx]
    print(f"  X_lm={X_lm.shape}")

    # Refit each missing fold
    for k in missing:
        pkl_path = SCRATCH / f"fold{k}_hazards.pkl"
        train_pid_file = SCRATCH / f"train{k}_pids.txt"
        train_pids = {ln.strip() for ln in
                      train_pid_file.read_text().splitlines() if ln.strip()}
        train_mask = np.array([p in train_pids for p in pids], dtype=bool)
        print(f"\nfold {k}: train_mask {int(train_mask.sum()):,} / "
              f"{len(pids):,} rows")
        t = time.time()
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid,
            train_mask=train_mask, seed=args.seed, verbose=True,
        )
        print(f"fold {k}: hazards fit in {time.time()-t:.0f}s, saving")
        tmp = pkl_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(pkl_path)
        del hazards
        gc.collect()
        print(f"fold {k}: wrote {pkl_path.name}")

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
