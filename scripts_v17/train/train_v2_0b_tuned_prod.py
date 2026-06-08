"""Train the v2.0b TUNED production stack and emit a buy list.

  1. Read best hazards HP from the Optuna study
       (sqlite:///scratch/v20b_oof/hazards_study.db).
  2. Train landmark hazards on 100% of the panel with those HP
       -> models/event_classifiers_v2.0b_tuned_prod.pkl
  3. Score the entire current prospect universe at snap=2026 with
     those hazards -> results/scored/snap2026_v2.0b_tuned_prod_long.csv
  4. Apply the existing OOF-honest joint XGB
       (models/joint_xgb_v2.0b_oof.pkl) on top of the snap long.
  5. Filter to the buy-list universe and write
       results/buy_lists/buy_list_v2.0b_TUNED_FINAL.csv
       results/buy_lists/buy_list_v2.0b_TUNED_ALL_SCORED.csv

Resumable per stage (skips on existence). --force-hazards to retrain.
"""
from __future__ import annotations

import argparse
import csv
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
from prospects.classifier.architectures.survival import (
    ELITE_KEY, MAX_OBS_YEAR, STAR_KEY, _trigger_year,
)
from prospects.storage import ProspectDB

# Reuse the snap-scoring helper from train_v2_0b_prod
from scripts_v17.train.train_v2_0b_prod import score_snap_with_landmark

HAZARDS_OUT = REPO_ROOT / "models" / "event_classifiers_v2.0b_tuned_prod.pkl"
SNAP_LONG = (REPO_ROOT / "results" / "scored"
              / "snap2026_v2.0b_tuned_prod_long.csv")
XGB_PKL = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"
TIMING_PKL = REPO_ROOT / "models" / "time_to_debut_v1.18_prod.pkl"
PRICES = REPO_ROOT / "data" / "prices_bowman_chrome_auto_v13.csv"
BUY_ALL = REPO_ROOT / "results" / "buy_lists" / "buy_list_v2.0b_TUNED_ALL_SCORED.csv"
BUY_FINAL = REPO_ROOT / "results" / "buy_lists" / "buy_list_v2.0b_TUNED_FINAL.csv"

OPTUNA_STORAGE = "sqlite:///scratch/v20b_oof/hazards_study.db"
OPTUNA_STUDY = "v20b_hazards_tune"

PANEL_CACHE = REPO_ROOT / "scratch" / "v20b_oof" / "panel_cache.npz"
PANEL_META = REPO_ROOT / "scratch" / "v20b_oof" / "panel_meta.pkl"


def _read_best_hp() -> dict:
    import optuna
    study = optuna.load_study(study_name=OPTUNA_STUDY,
                               storage=OPTUNA_STORAGE)
    print(f"Best Optuna hazards trial: weighted-AP "
          f"{study.best_value:.4f} ({len(study.trials)} trials total)")
    hp = dict(study.best_params)
    for k, v in hp.items():
        print(f"  {k}: {v}")
    return hp


def _load_panel():
    print(f"Loading panel cache {PANEL_CACHE.name}...")
    npz = np.load(PANEL_CACHE, allow_pickle=True)
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
    return X_lm, pids, S_yrs, joined, stats_by_pid, prospects_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-year", type=int, default=2026)
    ap.add_argument("--force-hazards", action="store_true")
    ap.add_argument("--force-snap", action="store_true")
    ap.add_argument("--force-buylist", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    args = ap.parse_args()

    t0 = time.time()

    # ---- Stage 1: read best HP ----
    best_hp = _read_best_hp()

    # ---- Stage 2: train hazards on 100% panel with best HP ----
    if HAZARDS_OUT.exists() and not args.force_hazards:
        print(f"\n[Stage 2] reusing {HAZARDS_OUT.name}")
        with HAZARDS_OUT.open("rb") as fh:
            hazards = pickle.load(fh)
        # Need panel for scoring stage
        _, _, _, _, stats_by_pid, prospects_list = _load_panel()
    else:
        X_lm, pids, S_yrs, joined, stats_by_pid, prospects_list = (
            _load_panel())
        print(f"\n[Stage 2] Training tuned hazards on 100% panel "
              f"({X_lm.shape[0]:,} landmark rows)")
        t = time.time()
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid,
            train_mask=None,           # 100% of panel — PROD
            seed=42, verbose=True,
            hazard_hp=best_hp,
        )
        print(f"  hazards trained in {time.time()-t:.0f}s")
        HAZARDS_OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = HAZARDS_OUT.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(HAZARDS_OUT)
        print(f"  saved {HAZARDS_OUT}")
        # Free panel arrays we don't need for scoring
        del X_lm, pids, S_yrs, joined

    # Dedupe joined → prospects_all for scoring
    seen: set[str] = set()
    prospects_all: list[dict] = []
    for p in prospects_list:
        if p["player_id"] in seen:
            continue
        seen.add(p["player_id"])
        prospects_all.append(p)

    # ---- Stage 3: score snap=2026 with tuned hazards ----
    if SNAP_LONG.exists() and not args.force_snap:
        print(f"\n[Stage 3] reusing {SNAP_LONG.name}")
    else:
        print(f"\n[Stage 3] Scoring snap={args.snap_year} universe "
              f"with tuned hazards")
        n = score_snap_with_landmark(
            hazards, prospects_all, stats_by_pid,
            snap_year=args.snap_year, out_csv=SNAP_LONG,
            horizon=15, verbose=True,
        )
        print(f"  wrote {n:,} rows -> {SNAP_LONG}")

    # ---- Stage 4: build buy list ----
    if BUY_FINAL.exists() and not args.force_buylist:
        print(f"\n[Stage 4] reusing {BUY_FINAL.name}")
    else:
        print(f"\n[Stage 4] Running build_v2.0_buylist.py")
        rc = subprocess.run([
            sys.executable,
            str(REPO_ROOT / "scripts_v17" / "buylist" / "build_v2.0_buylist.py"),
            "--long", str(SNAP_LONG),
            "--xgb", str(XGB_PKL),
            "--timing", str(TIMING_PKL),
            "--prices", str(PRICES),
            "--threshold", str(args.threshold),
            "--db", args.db,
            "--out-all", str(BUY_ALL),
            "--out-final", str(BUY_FINAL),
        ], cwd=REPO_ROOT).returncode
        if rc != 0:
            sys.exit(rc)

    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")
    print(f"  hazards: {HAZARDS_OUT}")
    print(f"  snap:    {SNAP_LONG}")
    print(f"  ALL:     {BUY_ALL}")
    print(f"  FINAL:   {BUY_FINAL}")


if __name__ == "__main__":
    main()
