"""Regenerate the v2.0b OOF stacked + val long CSVs using a CHOSEN set of
tuned hazard hyperparameters (e.g. Optuna trial 149), so the joint-XGB tuner
can be run "over this particular hazard setting".

It reuses the existing panel cache and fold partition from run_v2_0b_oof
(identical OOF split — only the hazard HP changes), refits the K-fold hazards
with the tuned HP into a SEPARATE `_tuned` namespace, and writes:

    results/training/v2.0b_oof_tuned_stacked_long.csv
    results/training/v2.0b_oof_tuned_val_long.csv

The default-HP OOF artifacts (v2.0b_oof_stacked_long.csv etc.) are left
untouched, so you can compare.

HP source: results/training/hazards_tuning_trials.csv, row where number==<trial>
(default 149). Pass --best to instead pick the max-value trial.

Usage (run in YOUR OWN terminal — long job, and it contends with any
still-running hazard study for cores):

    python -m scripts_v17.train.regen_oof_tuned_hazards --trial 149
    # then:
    python -m scripts_v17.train.tune_joint_xgb_v2_oof \\
        --fit results/training/v2.0b_oof_tuned_stacked_long.csv \\
        --val results/training/v2.0b_oof_tuned_val_long.csv \\
        --out-tag tuned_hz149

Every stage is resumable: fold hazards/CSVs and per-snap partials are cached
under scratch/v20b_oof_tuned/. Re-run after a crash and it picks up.
"""
from __future__ import annotations

import argparse
import gc
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
    SCRATCH, TRAIN_DIR, VAL_PIDS_PATH,
    _score_checkpointed, run_one_fold, stage_panel, stage_partition,
)

# Separate namespace so we never clobber the default-HP OOF artifacts.
SCRATCH_TUNED = REPO_ROOT / "scratch" / "v20b_oof_tuned"
OOF_STACKED_TUNED = TRAIN_DIR / "v2.0b_oof_tuned_stacked_long.csv"
OOF_VAL_TUNED = TRAIN_DIR / "v2.0b_oof_tuned_val_long.csv"
TRIALS_CSV = TRAIN_DIR / "hazards_tuning_trials.csv"

HP_KEYS = ["max_iter", "max_depth", "max_leaf_nodes", "learning_rate",
           "min_samples_leaf", "l2_regularization", "max_bins"]
INT_KEYS = {"max_iter", "max_depth", "max_leaf_nodes",
            "min_samples_leaf", "max_bins"}


def load_hazard_hp(trial: int | None, use_best: bool) -> dict:
    if not TRIALS_CSV.exists():
        sys.exit(f"FATAL: {TRIALS_CSV} not found — run the hazard tuner first.")
    df = pd.read_csv(TRIALS_CSV)
    if use_best:
        row = df.loc[df["value"].idxmax()]
    else:
        sub = df[df["number"] == trial]
        if sub.empty:
            sys.exit(f"FATAL: trial {trial} not in {TRIALS_CSV.name} "
                     f"(max trial = {int(df['number'].max())}).")
        row = sub.iloc[0]
    hp = {}
    for k in HP_KEYS:
        v = row[f"params_{k}"]
        hp[k] = int(v) if k in INT_KEYS else float(v)
    print(f"Using hazard HP from trial {int(row['number'])} "
          f"(value={row['value']:.4f}):")
    for k in HP_KEYS:
        print(f"    {k:<18} {hp[k]}")
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--trial", type=int, default=149,
                    help="hazard tuning trial number to use (default 149)")
    ap.add_argument("--best", action="store_true",
                    help="use the max-value trial instead of --trial")
    args = ap.parse_args()

    SCRATCH_TUNED.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("=" * 78)
    print(f"v2.0b OOF REGEN with tuned hazards — K={args.k}, seed={args.seed}")
    print("=" * 78)
    hazard_hp = load_hazard_hp(args.trial, args.best)

    val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    print(f"Val pids held out: {len(val_pid_set):,}")

    # Stage 1 + 2 — reuse cached panel and the SAME fold partition as the
    # default run (stage_partition reads existing scratch/v20b_oof fold files).
    X_lm, pids, S_yrs, joined, stats_by_pid = stage_panel(
        args.db, args.max_draft_year)
    seen: set[str] = set()
    prospects_all: list[dict] = []
    for p in joined:
        if p["player_id"] not in seen:
            seen.add(p["player_id"])
            prospects_all.append(p)
    fold_sets = stage_partition(
        prospects_all, stats_by_pid, val_pid_set,
        args.k, args.seed, args.max_entry_year)

    # Stage 3 — K folds, tuned HP, tuned namespace
    fold_csvs = [SCRATCH_TUNED / f"fold{j}_long.csv" for j in range(args.k)]
    last_hazards = None
    last_train_set: set[str] | None = None
    for j in range(args.k):
        out = fold_csvs[j]
        train_set: set[str] = set()
        for m in range(args.k):
            if m != j:
                train_set |= fold_sets[m]
        last_train_set = train_set
        if out.exists():
            print(f"\n[Stage 3] fold {j+1}/{args.k}: reusing {out.name}")
            last_hazards = None  # force val refit below
            continue
        print(f"\n[Stage 3] fold {j+1}/{args.k} (tuned) -> {out.name}")
        n, hazards = run_one_fold(
            X_lm, pids, S_yrs, joined, stats_by_pid, prospects_all,
            train_set=train_set, score_set=fold_sets[j], out_csv=out,
            max_entry_year=args.max_entry_year, seed=args.seed,
            partial_dir=SCRATCH_TUNED / f"fold{j}_partial",
            hazards_pkl=SCRATCH_TUNED / f"fold{j}_hazards.pkl",
            hazard_hp=hazard_hp,
        )
        print(f"           wrote {n:,} rows")
        last_hazards = hazards

    # Stage 4 — stack
    if OOF_STACKED_TUNED.exists():
        print(f"\n[Stage 4] reusing {OOF_STACKED_TUNED.name}")
    else:
        print(f"\n[Stage 4] stacking -> {OOF_STACKED_TUNED.name}")
        stacked = pd.concat([pd.read_csv(f) for f in fold_csvs],
                            ignore_index=True)
        stacked.to_csv(OOF_STACKED_TUNED, index=False)
        print(f"           {len(stacked):,} rows, "
              f"{stacked.player_id.nunique():,} pids")

    # Stage 5 — val scored with last-fold tuned hazards
    if OOF_VAL_TUNED.exists():
        print(f"\n[Stage 5] reusing {OOF_VAL_TUNED.name}")
    else:
        print(f"\n[Stage 5] scoring {len(val_pid_set):,} val pids (tuned)")
        if last_hazards is None:
            print(f"           refitting last-fold tuned hazards for val")
            train_mask = np.array([p in last_train_set for p in pids],
                                   dtype=bool)
            last_hazards = lm.fit_landmark_hazards(
                X_lm, joined, S_yrs, stats_by_pid,
                train_mask=train_mask, seed=args.seed, verbose=True,
                hazard_hp=hazard_hp,
            )
        n = _score_checkpointed(
            last_hazards, prospects_all, stats_by_pid, val_pid_set,
            OOF_VAL_TUNED, SCRATCH_TUNED / "val_partial",
            max_entry_year=args.max_entry_year,
            observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
        )
        print(f"           wrote {n:,} rows")
        del last_hazards
        gc.collect()

    print(f"\n=== DONE in {(time.time()-t_start)/60:.1f} min ===")
    print(f"  tuned stacked OOF: {OOF_STACKED_TUNED}")
    print(f"  tuned val OOF:     {OOF_VAL_TUNED}")
    print(f"\nNext: tune the joint XGB over these:")
    print(f"  python -m scripts_v17.train.tune_joint_xgb_v2_oof \\")
    print(f"      --fit {OOF_STACKED_TUNED} \\")
    print(f"      --val {OOF_VAL_TUNED} --out-tag tuned_hz{args.trial}")


if __name__ == "__main__":
    main()
