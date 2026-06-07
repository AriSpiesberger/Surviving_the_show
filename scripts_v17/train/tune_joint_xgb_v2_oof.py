"""Bayesian-optimize the v2.0b joint XGB on the OOF stacked dataset.

The OOF stacked long (`results/training/v2.0b_oof_stacked_long.csv`) carries
~250k rows of HONEST hazard predictions — each row's hazard outputs came
from a model that never saw it during training. That makes per-trial val
metrics genuine: every Optuna trial trains a fresh XGB on the OOF stacked
fit slice and evaluates it on the OOF-scored val slice.

Search space (informed by fit_joint_xgb_v2.py defaults):
  max_depth         int, 4..10
  learning_rate     log-uniform, 0.01..0.10
  min_child_weight  int, 5..100
  reg_lambda        log-uniform, 0.1..10
  subsample         uniform, 0.5..1.0
  colsample_bytree  uniform, 0.5..1.0
  num_rounds        fixed at 1000 with early stopping (25 rounds)

Objective: mean Average Precision across the 4 event heads on val, weighted
toward MLB_DEBUT (the primary buy-list filter).

Outputs:
  models/joint_xgb_v2.0b_oof_tuned.pkl   — best model retrained with best HPs
  results/training/oof_tuning_trials.csv — per-trial metrics for inspection
  results/training/oof_tuning_best.json  — best params + scores

Usage:
    python -m scripts_v17.train.tune_joint_xgb_v2_oof              # 100 trials
    python -m scripts_v17.train.tune_joint_xgb_v2_oof --trials 50  # quicker
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna",
          file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[2]

OOF_STACKED = REPO_ROOT / "results" / "training" / "v2.0b_oof_stacked_long.csv"
OOF_VAL = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
TRIALS_CSV = REPO_ROOT / "results" / "training" / "oof_tuning_trials.csv"
BEST_JSON = REPO_ROOT / "results" / "training" / "oof_tuning_best.json"
TUNED_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof_tuned.pkl"

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "STAR_PLUS_ELITE"]
HAZARD_PROBS = [
    "p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB",
    "p_ELITE", "p_STAR", "p_STAR_PLUS_ELITE",
]
FEAT = (
    HAZARD_PROBS
    + ["age_at_snap_centered", "years_in_pro"]
    + [f"{p}_x_yip_centered" for p in HAZARD_PROBS]
)
AGE_CENTER, YIP_CENTER = 22, 3

# Per-event weights in the objective. MLB_DEBUT carries 2x — that's the
# primary buy-list filter. Other events are co-equal because they describe
# the player's eventual tier and influence the buy-list breakout component.
EVENT_WEIGHTS = {
    "TOP_100_PROSPECT": 1.0,
    "MLB_DEBUT": 2.0,
    "ESTABLISHED_MLB": 1.0,
    "STAR_PLUS_ELITE": 1.0,
}


def _prep(df: pd.DataFrame, db: str, max_entry: int) -> pd.DataFrame:
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(
        birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]],
                  on="player_id", how="left")
    df["age_at_snap_centered"] = (
        (df["snap_year"] - df["birth_year"]).fillna(22.0) - AGE_CENTER
    )
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - YIP_CENTER
    for p in HAZARD_PROBS:
        df[f"{p}_x_yip_centered"] = df[p] * df["yip_centered"]
    df = df[df.entry_year <= max_entry].copy()
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
        df = df[df[f"eligible_{ev}"] == 1]
    needed = FEAT + [f"realized_{e}" for e in EVENTS]
    df = df.dropna(subset=needed)
    return df


def _build_dmatrices(fit_df, val_df, scaler):
    Y_fit = fit_df[[f"realized_{e}" for e in EVENTS]].values.astype(np.float32)
    Y_val = val_df[[f"realized_{e}" for e in EVENTS]].values.astype(np.float32)
    X_fit = scaler.transform(fit_df[FEAT].values.astype(np.float32))
    X_val = scaler.transform(val_df[FEAT].values.astype(np.float32))
    dtrain = xgb.DMatrix(X_fit, label=Y_fit, feature_names=FEAT)
    dval = xgb.DMatrix(X_val, label=Y_val, feature_names=FEAT)
    return dtrain, dval, Y_val


def _eval_metrics(booster, dval, Y_val, best_iter):
    P = booster.predict(dval, iteration_range=(0, best_iter + 1))
    rows = []
    weighted_ap = 0.0
    weight_total = 0.0
    for k, ev in enumerate(EVENTS):
        y = Y_val[:, k].astype(int); p = P[:, k]
        base = float(y.mean())
        ap = float(average_precision_score(y, p)) if y.sum() else float("nan")
        auc = (float(roc_auc_score(y, p))
               if 0 < y.sum() < len(y) else float("nan"))
        brier = float(brier_score_loss(y, p))
        rows.append({
            "event": ev, "base": base, "ap": ap,
            "ap_lift": ap / base if base > 0 else float("nan"),
            "auc": auc, "brier": brier,
        })
        if ap == ap:
            w = EVENT_WEIGHTS[ev]
            weighted_ap += w * ap
            weight_total += w
    obj = weighted_ap / weight_total if weight_total > 0 else 0.0
    return obj, rows


def make_objective(fit_df, val_df, db_path, scaler):
    def objective(trial):
        params = {
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-2, 1e-1, log=True),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 5, 100),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-1, 1e1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            "seed": 42,
            "verbosity": 0,
        }
        dtrain, dval, Y_val = _build_dmatrices(fit_df, val_df, scaler)
        booster = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=25,
            verbose_eval=False,
        )
        obj, rows = _eval_metrics(booster, dval, Y_val, booster.best_iteration)
        # Record per-event metrics on the trial for the trials CSV.
        for r in rows:
            trial.set_user_attr(f"{r['event']}_ap", r["ap"])
            trial.set_user_attr(f"{r['event']}_auc", r["auc"])
            trial.set_user_attr(f"{r['event']}_brier", r["brier"])
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return obj
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--storage", default=None,
                    help="Optional sqlite URI for Optuna persistence "
                         "(e.g. sqlite:///oof_study.db). Lets you resume "
                         "by re-running.")
    args = ap.parse_args()

    if not OOF_STACKED.exists() or not OOF_VAL.exists():
        sys.exit(f"FATAL: missing OOF inputs. Run train_v2_0b_oof first.\n"
                 f"  expected: {OOF_STACKED}\n"
                 f"            {OOF_VAL}")

    print("=" * 78)
    print(f"v2.0b OOF XGB Bayesian opt — {args.trials} trials")
    print("=" * 78)
    print(f"Loading OOF stacked: {OOF_STACKED.name}", flush=True)
    fit_df = _prep(pd.read_csv(OOF_STACKED), args.db, args.max_entry)
    print(f"  fit: {len(fit_df):,} rows, "
          f"{fit_df.player_id.nunique():,} unique players", flush=True)
    print(f"Loading OOF val: {OOF_VAL.name}", flush=True)
    val_df = _prep(pd.read_csv(OOF_VAL), args.db, args.max_entry)
    print(f"  val: {len(val_df):,} rows, "
          f"{val_df.player_id.nunique():,} unique players", flush=True)

    # Single scaler fit on fit slice (no leakage)
    scaler = StandardScaler().fit(fit_df[FEAT].values.astype(np.float32))

    # Set up Optuna
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    storage = args.storage  # None means in-memory study
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler, pruner=pruner,
        study_name="v20b_oof_xgb",
        storage=storage, load_if_exists=bool(storage),
    )

    t0 = time.time()
    study.optimize(
        make_objective(fit_df, val_df, args.db, scaler),
        n_trials=args.trials,
        show_progress_bar=False,
    )
    elapsed_min = (time.time() - t0) / 60
    print(f"\nOptimization done in {elapsed_min:.1f} min. "
          f"Best objective = {study.best_value:.4f}", flush=True)

    # Persist trial results
    trials_df = study.trials_dataframe()
    TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(TRIALS_CSV, index=False)
    print(f"Wrote {TRIALS_CSV}", flush=True)

    best_params = dict(study.best_params)
    best_attrs = dict(study.best_trial.user_attrs)
    BEST_JSON.write_text(json.dumps({
        "best_objective": float(study.best_value),
        "best_params": best_params,
        "best_event_metrics": best_attrs,
        "n_trials": args.trials,
        "wall_min": elapsed_min,
    }, indent=2))
    print(f"Wrote {BEST_JSON}", flush=True)

    # Retrain final model with best params on the fit slice, save
    print("\nRetraining final model with best params...", flush=True)
    dtrain, dval, Y_val = _build_dmatrices(fit_df, val_df, scaler)
    final_params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
        "verbosity": 0,
        **best_params,
    }
    booster = xgb.train(
        final_params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=25,
        verbose_eval=50,
    )
    obj_final, rows_final = _eval_metrics(
        booster, dval, Y_val, booster.best_iteration)
    print(f"\nFinal model val objective = {obj_final:.4f}")
    print(f"{'event':<22} {'base%':>7} {'AP':>7} {'lift':>7} {'AUC':>6} "
          f"{'Brier':>7}")
    for r in rows_final:
        print(f"{r['event']:<22} {r['base']*100:>6.2f} {r['ap']:>7.3f} "
              f"{r['ap_lift']:>7.2f} {r['auc']:>6.3f} "
              f"{r['brier']:>7.4f}")

    TUNED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TUNED_OUT.open("wb") as fh:
        pickle.dump({
            "model": booster, "scaler": scaler,
            "feature_names": list(FEAT),
            "events": list(EVENTS),
            "best_iteration": int(booster.best_iteration),
            "version": "v2.0b_oof_tuned",
            "kind": "joint_xgb_multi_output_tree_tuned",
            "metrics_val": rows_final,
            "params": final_params,
            "tuning": {
                "objective_value": float(obj_final),
                "n_trials": args.trials,
                "best_params_from_study": best_params,
            },
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {TUNED_OUT}")


if __name__ == "__main__":
    main()
