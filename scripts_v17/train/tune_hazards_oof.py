"""Optuna-tune the landmark hazards HP, optimizing per-event val AP.

Why tune hazards (not XGB):
  The hazards are upstream of the XGB. Their noise sets a floor on what
  the XGB can do. Optimizing XGB HP can polish a few percent; optimizing
  hazards moves the whole signal stack.

Per trial:
  1. Train hazards_full on the entire 90% universe with the trial's HP
  2. Score val pids with those hazards (using existing checkpointing)
  3. Compute per-event AP on val using the eligible mask
  4. Return weighted-mean AP across the 4 buy-list events

Search space (HistGradientBoostingClassifier knobs):
  max_iter           int        50..400
  max_depth          int        3..10
  learning_rate      log        0.01..0.20
  min_samples_leaf   int        10..200
  l2_regularization  log        1e-4..1.0
  max_bins           int        64..255

Trials write incrementally to results/training/hazards_tuning_trials.csv
and best params to results/training/hazards_tuning_best.json. The best
trial's hazards are saved to models/hazards_v2.0b_oof_tuned.pkl.

Usage:
    python -m scripts_v17.train.tune_hazards_oof --trials 50

Each trial costs ~5-7 min (hazards fit + val score), so 50 trials ~ 5h.
For a smoke test, run --trials 5 first.
"""
from __future__ import annotations

import os
# Threading defaults: respect any pre-set env vars (let the user pin
# threads via shell), otherwise let HistGB's OpenMP use all cores. Was
# pinned to 1 during the BSOD-instability window; the box is stable now.
# To force single-threaded again, set OMP_NUM_THREADS=1 in your shell.

import argparse
import gc
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import optuna
except ImportError:
    sys.exit("ERROR: optuna not installed. Run: pip install optuna")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from scripts_v17.train.run_v2_0b_oof import (
    PANEL_META, PANEL_NPZ, SCRATCH, VAL_PIDS_PATH, _score_checkpointed,
)

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT",
          "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
EVENT_WEIGHTS = {
    "TOP_100_PROSPECT": 1.0,
    "MLB_DEBUT": 2.0,        # primary buy-list filter
    "ESTABLISHED_MLB": 1.0,
    "STAR_PLUS_ELITE": 1.0,
}

TUNE_DIR = SCRATCH / "hazards_tuning"
TRIALS_CSV = REPO_ROOT / "results" / "training" / "hazards_tuning_trials.csv"
BEST_JSON = REPO_ROOT / "results" / "training" / "hazards_tuning_best.json"
BEST_PKL = REPO_ROOT / "models" / "hazards_v2.0b_oof_tuned.pkl"


def _per_event_metrics(val_df: pd.DataFrame,
                       max_entry: int) -> tuple[float, dict]:
    df = val_df[val_df.entry_year <= max_entry].copy()
    weighted_ap = 0.0
    weight_total = 0.0
    per_event = {}
    for ev in EVENTS:
        elig = df[f"eligible_{ev}"] == 1
        sub = df[elig]
        if len(sub) == 0:
            continue
        y = sub[f"realized_{ev}"].astype(int).values
        p = sub[f"p_{ev}"].astype(float).values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ap = float(average_precision_score(y, p))
        auc = float(roc_auc_score(y, p))
        base = float(y.mean())
        per_event[ev] = {"ap": ap, "auc": auc, "base": base,
                          "n": int(len(y)),
                          "ap_lift": ap / base if base > 0 else None}
        w = EVENT_WEIGHTS[ev]
        weighted_ap += w * ap
        weight_total += w
    obj = weighted_ap / weight_total if weight_total > 0 else 0.0
    return obj, per_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--storage", default=None,
                    help="Optional sqlite URI for Optuna persistence "
                         "(e.g. sqlite:///hazards_study.db)")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Concurrent trials. HistGB releases the GIL "
                         "during fit, so 2-4 scales well on a "
                         "multicore box.")
    ap.add_argument("--n-startup", type=int, default=10,
                    help="Trials TPE does pure random sampling before "
                         "switching to exploit. Higher = more "
                         "exploration. For a 200-trial run, 40-60 is "
                         "a reasonable explore/exploit balance.")
    args = ap.parse_args()

    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PKL.parent.mkdir(parents=True, exist_ok=True)
    TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 78)
    print(f"v2.0b hazards tune — {args.trials} trials, seed={args.seed}")
    print("=" * 78)

    val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    print(f"val pids: {len(val_pid_set):,}")

    # Load panel ONCE — reused across all trials
    print("Loading panel cache...")
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

    # Build full train mask (90% universe)
    train_pid_set: set[str] = set()
    for k in range(args.k):
        train_pid_set |= {ln.strip() for ln in
                          (SCRATCH / f"fold{k}_pids.txt").read_text()
                          .splitlines() if ln.strip()}
    full_train_mask = np.array([p in train_pid_set for p in pids],
                                dtype=bool)
    print(f"full train universe: {len(train_pid_set):,} pids, "
          f"{int(full_train_mask.sum()):,} landmark rows")

    # Dedupe prospects_all for scoring (reused across trials)
    seen: set[str] = set()
    prospects_all: list[dict] = []
    for p in joined:
        if p["player_id"] in seen:
            continue
        seen.add(p["player_id"])
        prospects_all.append(p)

    def objective(trial: "optuna.Trial") -> float:
        hp = {
            "max_iter":          trial.suggest_int("max_iter", 50, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes",
                                                    15, 511),
            "learning_rate":     trial.suggest_float(
                "learning_rate", 1e-2, 2e-1, log=True),
            "min_samples_leaf":  trial.suggest_int(
                "min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.0, 5.0),
            "max_bins":          trial.suggest_int("max_bins", 64, 255),
        }
        trial_dir = TUNE_DIR / f"trial_{trial.number:04d}"
        partial_dir = trial_dir / "val_partial"
        trial_dir.mkdir(parents=True, exist_ok=True)
        t_trial = time.time()
        print(f"\n[trial {trial.number}] hp={hp}", flush=True)
        # Fit hazards with this HP set
        try:
            hazards = lm.fit_landmark_hazards(
                X_lm, joined, S_yrs, stats_by_pid,
                train_mask=full_train_mask, seed=args.seed,
                verbose=False, hazard_hp=hp,
            )
        except TypeError:
            # Backwards-compatible — pass hp via _train_event default
            from prospects.classifier.architectures import (
                landmark_survival as _lm,
            )
            _orig = _lm._train_event
            def _wrapped(Xtr, ytr, seed=args.seed):
                return _orig(Xtr, ytr, seed=seed, hp=hp)
            _lm._train_event = _wrapped
            try:
                hazards = _lm.fit_landmark_hazards(
                    X_lm, joined, S_yrs, stats_by_pid,
                    train_mask=full_train_mask, seed=args.seed,
                    verbose=False,
                )
            finally:
                _lm._train_event = _orig
        # Score val
        val_csv = trial_dir / "val_long.csv"
        _score_checkpointed(
            hazards, prospects_all, stats_by_pid, val_pid_set,
            val_csv, partial_dir,
            max_entry_year=args.max_entry_year,
            observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
        )
        val_df = pd.read_csv(val_csv)
        obj, per_event = _per_event_metrics(val_df, args.max_entry_year)
        elapsed = time.time() - t_trial
        print(f"[trial {trial.number}] obj={obj:.4f}  ({elapsed:.0f}s)")
        for ev, m in per_event.items():
            print(f"  {ev:<22} AP={m['ap']:.3f}  AUC={m['auc']:.3f}")
            trial.set_user_attr(f"{ev}_ap", m["ap"])
            trial.set_user_attr(f"{ev}_auc", m["auc"])
        trial.set_user_attr("wall_sec", elapsed)
        # Persist hazards if it's the new best (cheaper than refit later)
        if not hasattr(objective, "best_obj") or obj > objective.best_obj:
            objective.best_obj = obj
            tmp = BEST_PKL.with_suffix(".pkl.tmp")
            with tmp.open("wb") as fh:
                pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(BEST_PKL)
            print(f"  ✓ new best — saved {BEST_PKL.name}")
        # Persist trials so far
        try:
            study.trials_dataframe().to_csv(TRIALS_CSV, index=False)
        except Exception:
            pass
        # Clean per-trial dir to keep disk tidy
        try:
            shutil.rmtree(trial_dir)
        except Exception:
            pass
        del hazards
        gc.collect()
        return obj

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup,
    )
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        study_name="v20b_hazards_tune",
        storage=args.storage, load_if_exists=bool(args.storage),
    )
    # Inject the known baseline result as a completed FrozenTrial so TPE
    # gets the prior info without re-running the baseline. We measured
    # the non-OOF default HP at weighted-AP = 0.3307 in an earlier run.
    # Bypasses the enqueue_trial race condition we hit with --n-jobs > 1.
    if len(study.trials) == 0:
        from optuna.trial import TrialState, create_trial
        from optuna.distributions import IntDistribution, FloatDistribution
        baseline_params = {
            "max_iter": 200, "max_depth": 6, "max_leaf_nodes": 31,
            "learning_rate": 0.05, "min_samples_leaf": 30,
            "l2_regularization": 0.0, "max_bins": 255,
        }
        baseline_dists = {
            "max_iter": IntDistribution(50, 600),
            "max_depth": IntDistribution(3, 12),
            "max_leaf_nodes": IntDistribution(15, 511),
            "learning_rate": FloatDistribution(1e-2, 2e-1, log=True),
            "min_samples_leaf": IntDistribution(10, 200),
            "l2_regularization": FloatDistribution(0.0, 5.0),
            "max_bins": IntDistribution(64, 255),
        }
        frozen = create_trial(
            params=baseline_params,
            distributions=baseline_dists,
            value=0.3307,             # measured in prior run
            state=TrialState.COMPLETE,
        )
        study.add_trial(frozen)
        print(f"Injected known baseline result (value=0.3307) as "
              f"completed trial — TPE uses it as prior, no re-run.")
    study.optimize(objective, n_trials=args.trials,
                   n_jobs=args.n_jobs, show_progress_bar=False)

    BEST_JSON.write_text(json.dumps({
        "best_objective": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_event_metrics": dict(study.best_trial.user_attrs),
        "n_trials": args.trials,
        "wall_min": (time.time() - t0) / 60,
    }, indent=2))
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"  best obj: {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")
    print(f"  best hazards: {BEST_PKL}")
    print(f"  trials csv:  {TRIALS_CSV}")
    print(f"  best json:   {BEST_JSON}")


if __name__ == "__main__":
    main()
