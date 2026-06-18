"""Train production landmark hazards on 100% of panel with default HP.

The PROD hazards are the ones used for snap=2026 inference. They train
on the entire panel (val pids included — at inference time val is
irrelevant) so the model has maximum data.

Default HP — no Optuna. Matches what fit_joint_xgb_v2 was trained
against, so the joint XGB head will see in-distribution features at
inference time.

Output:
    models/event_classifiers_v2.0b_prod.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm

PANEL_NPZ = REPO_ROOT / "scratch" / "v20b_oof" / "panel_cache.npz"
PANEL_META = REPO_ROOT / "scratch" / "v20b_oof" / "panel_meta.pkl"
OUT = REPO_ROOT / "models" / "event_classifiers_v2.0b_prod.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="Read the tagged panel cache "
                         "(scratch/v20b_oof_<tag>/) and write tagged prod "
                         "hazards, matching run_v2_0b_oof --tag. The panel is "
                         "already partial-sampled if it was built that way, so "
                         "no separate partial flag is needed here.")
    args = ap.parse_args()

    global PANEL_NPZ, PANEL_META, OUT
    if args.tag:
        scratch = REPO_ROOT / "scratch" / f"v20b_oof_{args.tag}"
        PANEL_NPZ = scratch / "panel_cache.npz"
        PANEL_META = scratch / "panel_meta.pkl"
        OUT = REPO_ROOT / "models" / f"event_classifiers_v2.0b_{args.tag}_prod.pkl"

    if OUT.exists() and not args.force:
        print(f"PROD hazards already exist at {OUT}. --force to overwrite.")
        return 0

    t0 = time.time()
    print(f"Loading panel cache {PANEL_NPZ.name}...")
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
    print(f"  X_lm={X_lm.shape}  prospects={len(prospects_list):,}")

    print(f"\nFitting landmark hazards on 100% of panel (default HP, "
          f"no Optuna)...")
    t = time.time()
    hazards = lm.fit_landmark_hazards(
        X_lm, joined, S_yrs, stats_by_pid,
        train_mask=None,  # 100% — val pids included for prod inference
        seed=args.seed, verbose=True,
    )
    print(f"\nHazards trained in {time.time()-t:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(OUT)
    print(f"\nWrote {OUT}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
