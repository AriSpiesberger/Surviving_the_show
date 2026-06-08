"""Finalize v2.0b OOF:
  1. Train ONE hazards_full on the entire 90% universe (all train pids
     pooled — same hazards that production buy-sheet inference will use)
  2. Score val pids with hazards_full → overwrites
     results/training/v2.0b_oof_val_long.csv
  3. Refit joint XGB on stacked OOF (training) + hazards_full-scored val
     (early-stopping val) → models/joint_xgb_v2.0b_oof.pkl
  4. Print val metrics

Training data stays OOF-honest (stacked from 6 fold scorings, each row
scored with hazards that didn't see it). Val is scored with the
hazards-trained-on-everything that will be used in real inference, so
val metrics predict actual buy-list behavior.

All stages resumable:
  - hazards_full skipped if scratch/v20b_oof/hazards_full.pkl exists
  - val scoring skipped per-snap (val_partial_full/snap_NNNN.csv)
  - XGB refit always re-runs (cheap, ~5 min)
"""
from __future__ import annotations

# Threading: respect any pre-set env vars (set OMP_NUM_THREADS=1 in your
# shell to force single-threaded). Was pinned to 1 during the BSOD
# window; the box is stable now.

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

from prospects.classifier.architectures import landmark_survival as lm
from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from scripts_v17.train.run_v2_0b_oof import (
    PANEL_META, PANEL_NPZ, SCRATCH, VAL_PIDS_PATH, _score_checkpointed,
)

HAZARDS_FULL = SCRATCH / "hazards_full.pkl"
OOF_VAL = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
OOF_STACKED = REPO_ROOT / "results" / "training" / "v2.0b_oof_stacked_long.csv"
XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"
VAL_PARTIAL = SCRATCH / "val_partial_full"


def _read_pids(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--no-refit", action="store_true",
                    help="Skip the XGB refit at the end")
    args = ap.parse_args()

    t0 = time.time()
    val_pid_set = _read_pids(VAL_PIDS_PATH)
    print(f"[finalize] val pids held out: {len(val_pid_set):,}")

    # ---- Load panel ----
    print(f"[finalize] loading panel cache...")
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
    print(f"           X_lm={X_lm.shape}")

    # ---- Build full train_mask: union of all K folds (= entire 90% universe) ----
    train_pid_set: set[str] = set()
    for k in range(args.k):
        train_pid_set |= _read_pids(SCRATCH / f"fold{k}_pids.txt")
    # Sanity: train_pid_set should NOT intersect val_pid_set
    overlap = train_pid_set & val_pid_set
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} pids appear in both train universe "
                 f"and val. Aborting to keep val honest.")
    full_train_mask = np.array([p in train_pid_set for p in pids],
                               dtype=bool)
    print(f"[finalize] full train universe: {len(train_pid_set):,} pids, "
          f"{int(full_train_mask.sum()):,} of {len(pids):,} landmark rows")

    # ---- Stage 1: train hazards_full ----
    if HAZARDS_FULL.exists():
        print(f"[finalize] reusing existing {HAZARDS_FULL.name}")
        with HAZARDS_FULL.open("rb") as fh:
            hazards = pickle.load(fh)
    else:
        print(f"\n[finalize] training hazards_full on entire 90% universe "
              f"(this is what prod inference uses)")
        t = time.time()
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid,
            train_mask=full_train_mask, seed=args.seed, verbose=True,
        )
        print(f"[finalize] hazards_full fit in {time.time()-t:.0f}s, saving")
        tmp = HAZARDS_FULL.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(HAZARDS_FULL)

    # ---- Stage 2: score val with hazards_full ----
    # Dedupe joined to prospects_all BEFORE freeing the panel arrays
    seen: set[str] = set()
    prospects_all: list[dict] = []
    for p in joined:
        if p["player_id"] in seen:
            continue
        seen.add(p["player_id"])
        prospects_all.append(p)
    # Free panel arrays — scoring only needs prospects_all + stats_by_pid
    del X_lm, pids, S_yrs, joined_idx, joined, npz
    gc.collect()

    print(f"\n[finalize] scoring {len(val_pid_set):,} val pids with "
          f"hazards_full")
    n = _score_checkpointed(
        hazards, prospects_all, stats_by_pid, val_pid_set, OOF_VAL,
        VAL_PARTIAL,
        max_entry_year=args.max_entry_year,
        observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
    )
    print(f"[finalize] val: {n:,} rows -> {OOF_VAL.name}")

    # ---- Stage 3: refit XGB ----
    if args.no_refit:
        print(f"[finalize] --no-refit: skipping XGB fit")
        return 0

    print(f"\n[finalize] retraining joint XGB")
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

    # ---- Final report ----
    with XGB_OUT.open("rb") as fh:
        bundle = pickle.load(fh)
    print(f"\n===== v2.0b OOF final ({XGB_OUT.name}) =====")
    print(f"  best_iter: {bundle.get('best_iteration')}")
    for r in bundle.get("metrics_val", []):
        print(f"  {r['event']:<22} AP={r['ap']:.3f}  "
              f"lift={r['ap_lift']:.1f}x  AUC={r['auc']:.3f}")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    main()
