"""Single-fold worker for v2.0b OOF training.

Invoked by train_v2_0b_oof.py — one subprocess per fold so memory is fully
released between folds (the landmark panel + hazards + stats_by_pid are ~1.5GB
combined, holding all K folds' worth simultaneously would OOM).

Usage (invoked by parent, not directly):
    python -m scripts_v17.train.train_v2_0b_oof_fold \\
        --fold-pids fold0_pids.txt \\
        --train-pids notfold0_pids.txt \\
        --out fold0_long.csv

Reads the panel fresh from the DB each invocation. Slow but memory-safe.
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from prospects.storage import ProspectDB
from scripts_v17.train.train_v1_18b_prod import score_pids_with_landmark

CACHE_DIR = REPO_ROOT / "scratch" / "v20b_oof"
PANEL_NPZ = CACHE_DIR / "panel_cache.npz"
PANEL_META = CACHE_DIR / "panel_meta.pkl"


def _read_pid_file(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def _load_panel_or_build(db_path: str, max_draft_year: int):
    """Load panel from cache if present, else build from DB.

    Returns (X_lm, pids, S_yrs, joined, stats_by_pid).
    """
    if PANEL_NPZ.exists() and PANEL_META.exists():
        print(f"[fold worker] loading panel cache from "
              f"{PANEL_NPZ.name}", flush=True)
        t = time.time()
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
        print(f"[fold worker] panel loaded in {time.time()-t:.0f}s  "
              f"X_lm={X_lm.shape}", flush=True)
        return X_lm, pids, S_yrs, joined, stats_by_pid

    db = ProspectDB(db_path)
    print(f"[fold worker] no cache — building landmark panel "
          f"(max_draft_year={max_draft_year})", flush=True)
    t = time.time()
    X_lm, pids, S_yrs, joined, stats_by_pid = lm.build_landmark_panel(
        db, max_draft_year=max_draft_year,
        min_landmark_year=2007, max_landmark_year=MAX_OBS_YEAR - 1,
        include_ifa=True, verbose=True,
    )
    print(f"[fold worker] panel built in {time.time()-t:.0f}s  "
          f"X_lm={X_lm.shape}", flush=True)
    return X_lm, pids, S_yrs, joined, stats_by_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--train-pids", required=True,
                    help="pid list — players the hazards SEE during training")
    ap.add_argument("--fold-pids", required=True,
                    help="pid list — players to score (the OOF heldout fold)")
    ap.add_argument("--out", required=True,
                    help="output scored long CSV path")
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    train_pid_set = _read_pid_file(Path(args.train_pids))
    fold_pid_set = _read_pid_file(Path(args.fold_pids))
    print(f"[fold worker] train pids: {len(train_pid_set):,}, "
          f"fold pids: {len(fold_pid_set):,}", flush=True)

    X_lm, pids, S_yrs, joined, stats_by_pid = _load_panel_or_build(
        args.db, args.max_draft_year)

    # Train mask at landmark-row granularity
    train_mask = np.array([p in train_pid_set for p in pids], dtype=bool)
    print(f"[fold worker] train_mask: {int(train_mask.sum()):,} "
          f"of {len(pids):,} landmark rows in training", flush=True)

    t1 = time.time()
    hazards = lm.fit_landmark_hazards(
        X_lm, joined, S_yrs, stats_by_pid,
        train_mask=train_mask, seed=args.seed, verbose=True,
    )
    print(f"[fold worker] hazards trained in {time.time()-t1:.0f}s",
          flush=True)

    # Dedupe joined → prospects_all for scoring
    seen = set(); prospects_all = []
    for p in joined:
        if p["player_id"] in seen:
            continue
        seen.add(p["player_id"]); prospects_all.append(p)

    # Free panel arrays we don't need for scoring
    del X_lm, pids, S_yrs, joined
    gc.collect()

    t2 = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = score_pids_with_landmark(
        hazards, prospects_all, stats_by_pid, fold_pid_set,
        out_path,
        max_entry_year=args.max_entry_year,
        observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
        verbose=True,
    )
    print(f"[fold worker] wrote {n:,} rows -> {out_path} "
          f"in {time.time()-t2:.0f}s", flush=True)

    # NaN-fill mean_t/sd_t so the XGB / time-to-debut don't choke
    df = pd.read_csv(out_path)
    for c in df.columns:
        if c.startswith("mean_t_"):
            df[c] = df[c].fillna(15.0)
        elif c.startswith("sd_t_"):
            df[c] = df[c].fillna(0.0)
    df.to_csv(out_path, index=False)

    print(f"[fold worker] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
