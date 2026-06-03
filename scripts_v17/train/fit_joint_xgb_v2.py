"""v2.0: joint multi-output XGBoost with shared-trunk multi_output_tree.

Each boosting round grows ONE tree with 4 leaf values (one per event head).
Tree splits minimize summed gradient across all 4 outputs — features that
help only the rare classes (TOP_100, STAR+) can ride on splits chosen for
the dense classes (MLB_DEBUT, ESTABLISHED). That's the "shared trunk" we
were missing in v1.18, where the four L1-logistics each train alone.

Features: same 14-d set as v1.18 (6 honest hazard probs + age + yip +
                                    6 yip-interactions).
Targets : binary realized_<event> for the 4 events.

Training: honest fit slice (v1.17h_fit_long.csv).
Honest val (v1.17h_val_long.csv) is used only as the early-stopping
holdout — no model selection is done off it, AP/Brier/lift reported.

Output:
  models/joint_xgb_v2.0h.pkl  = {"model": xgb.Booster, "scaler": StandardScaler,
                                  "feature_names": [...], "events": [...],
                                  "best_iteration": int}
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="v1.17h_fit_long.csv")
    ap.add_argument("--val", default="v1.17h_val_long.csv")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-rounds", type=int, default=500)
    ap.add_argument("--early-stop", type=int, default=25)
    ap.add_argument("--min-child-weight", type=int, default=30)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/joint_xgb_v2.0h.pkl")
    args = ap.parse_args()

    print(f"Loading {args.fit}")
    fit = _prep(pd.read_csv(args.fit), args.db, args.max_entry)
    val = _prep(pd.read_csv(args.val), args.db, args.max_entry)
    print(f"  fit: {len(fit):,} rows, {fit.player_id.nunique():,} players")
    print(f"  val: {len(val):,} rows, {val.player_id.nunique():,} players")

    print(f"\nPositives per event (honest fit):")
    for ev in EVENTS:
        n = int(fit[f"realized_{ev}"].sum())
        print(f"  {ev:<22} pos={n:,}  base={n/len(fit):.2%}")

    Y_fit = fit[[f"realized_{e}" for e in EVENTS]].values.astype(np.float32)
    Y_val = val[[f"realized_{e}" for e in EVENTS]].values.astype(np.float32)
    scaler = StandardScaler().fit(fit[FEAT].values.astype(np.float32))
    X_fit = scaler.transform(fit[FEAT].values.astype(np.float32))
    X_val = scaler.transform(val[FEAT].values.astype(np.float32))

    dtrain = xgb.DMatrix(X_fit, label=Y_fit, feature_names=FEAT)
    dval = xgb.DMatrix(X_val, label=Y_val, feature_names=FEAT)

    params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.l2,
        "seed": args.seed,
        "verbosity": 1,
    }
    print(f"\nTraining XGBoost (multi_output_tree, max_depth={args.max_depth}, "
          f"lr={args.lr}, early_stop={args.early_stop})...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=args.num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stop,
        verbose_eval=25,
    )
    best_iter = booster.best_iteration
    print(f"\nbest_iteration = {best_iter}")

    # Honest val metrics
    P_val = booster.predict(dval, iteration_range=(0, best_iter + 1))
    print(f"\n===== HONEST VAL (n={len(val):,}) =====")
    print(f"{'event':<22} {'base%':>7} {'AU-PR':>7} {'AP_lift':>8} "
          f"{'AUC':>6} {'Brier':>7}")
    rows = []
    for k, ev in enumerate(EVENTS):
        y = Y_val[:, k].astype(int)
        p = P_val[:, k]
        base = float(y.mean())
        ap = float(average_precision_score(y, p)) if y.sum() else float("nan")
        auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
        brier = float(brier_score_loss(y, p))
        lift = ap / base if base else float("nan")
        rows.append({"event": ev, "base": base, "ap": ap, "ap_lift": lift,
                     "auc": auc, "brier": brier})
        print(f"{ev:<22} {base*100:>6.2f} {ap:>7.3f} {lift:>8.2f} "
              f"{auc:>6.3f} {brier:>7.4f}")

    with open(args.out, "wb") as fh:
        pickle.dump({
            "model": booster, "scaler": scaler,
            "feature_names": list(FEAT),
            "events": list(EVENTS),
            "best_iteration": int(best_iter),
            "version": "v2.0",
            "kind": "joint_xgb_multi_output_tree",
            "metrics_val": rows,
            "params": params,
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
