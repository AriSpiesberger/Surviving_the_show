"""v2.1c: conditional refinement of the landmark hazard model.

Replaces fit_joint_xgb_v2.py. Instead of a terminal head that collapses the
hazard trajectory into one scalar per event, this trains a joint multi-output
XGB that, given a player's full hazard curves + baseline + a target horizon h,
outputs the refined cumulative P(event by snap+h). Sweeping h=1..H_MAX at
inference yields a per-year trajectory vector (see prospects.classifier.joint_cond).

Key differences from v2.0:
  - Training rows are (player-snap, h) pairs for h in 1..H_MAX, kept only where
    years_fwd >= h (per-horizon right-censoring, built in — no --censor-window).
  - Labels are realized_by_h (cumulative-by-horizon), not realized-ever.
  - Features add the full hk1..hk10 curves, the hazard model's own cumulative
    answer at h (haz_cum_h_<event>), and h itself.

Output bundle (models/joint_xgb_*.pkl):
  {"model": Booster, "scaler": StandardScaler, "feature_names": FEAT_COND,
   "events": EVENTS, "best_iteration": int, "h_max": int, "publish_h": int,
   "version": "v2.1c", "kind": "joint_xgb_cond_horizon", "metrics_val": [...]}

Usage:
    python -m scripts_v17.train.fit_joint_xgb_cond \
        --fit results/training/v2.0b_oof_stacked_long.csv \
        --val results/training/v2.0b_oof_val_long.csv \
        --db prospects_snapshot.db --out models/joint_xgb_v2.1c.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.joint_cond import (  # noqa: E402
    EVENTS, FEAT_COND, H_MAX, PUBLISH_H, expand_long, prep_base,
)

# Eligibility gate (matches v2.0): drop snaps where the event already fired by
# the snap year for the two dense early events. Applied at the player-snap level.
ELIGIBLE_GATE = ("TOP_100_PROSPECT", "MLB_DEBUT")


def _prep_train(df: pd.DataFrame, db: str, max_entry: int) -> pd.DataFrame:
    df = prep_base(df, db, max_entry=max_entry)
    for ev in ELIGIBLE_GATE:
        col = f"eligible_{ev}"
        if col in df.columns:
            df = df[df[col] == 1]
    return df.copy()


def _assemble(base: pd.DataFrame, h_max: int):
    """Long-expand to resolved (row, h) pairs, drop FEAT_COND NaNs, and rebuild
    the cumulative-by-h labels on the surviving frame (using the per-row h that
    expand_long stamped) so X and Y stay perfectly aligned."""
    long_df, _ = expand_long(base, h_max=h_max)
    long_df = long_df.dropna(subset=FEAT_COND).reset_index(drop=True)
    if not len(long_df):
        return long_df, np.empty((0, len(EVENTS)), np.float32)
    snap = long_df["snap_year"].astype(float)
    hcol = long_df["h"].astype(float)
    Y = np.empty((len(long_df), len(EVENTS)), dtype=np.float32)
    for k, ev in enumerate(EVENTS):
        trig = pd.to_numeric(long_df[f"trigger_{ev}"], errors="coerce")
        Y[:, k] = (trig.notna() & (trig > snap)
                   & (trig <= snap + hcol)).to_numpy(np.float32)
    return long_df, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="results/training/v2.0b_oof_stacked_long.csv")
    ap.add_argument("--val", default="results/training/v2.0b_oof_val_long.csv")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--h-max", type=int, default=H_MAX)
    ap.add_argument("--publish-h", type=int, default=PUBLISH_H)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-rounds", type=int, default=500)
    ap.add_argument("--early-stop", type=int, default=25)
    ap.add_argument("--min-child-weight", type=int, default=30)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/joint_xgb_v2.1c.pkl")
    args = ap.parse_args()

    print(f"Loading {args.fit}")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, args.max_entry)
    val_base = _prep_train(pd.read_csv(args.val), args.db, args.max_entry)

    fit, Y_fit = _assemble(fit_base, args.h_max)
    val, Y_val = _assemble(val_base, args.h_max)
    print(f"  fit: {len(fit):,} (row,h) rows, "
          f"{fit.player_id.nunique():,} players, h in 1..{args.h_max}")
    print(f"  val: {len(val):,} (row,h) rows, {val.player_id.nunique():,} players")

    print(f"\nPositives per event (cumulative-by-h, fit):")
    for k, ev in enumerate(EVENTS):
        n = int(Y_fit[:, k].sum())
        print(f"  {ev:<22} pos={n:,}  base={n/max(len(fit),1):.2%}")

    scaler = StandardScaler().fit(fit[FEAT_COND].values.astype(np.float32))
    X_fit = scaler.transform(fit[FEAT_COND].values.astype(np.float32))
    X_val = scaler.transform(val[FEAT_COND].values.astype(np.float32))

    dtrain = xgb.DMatrix(X_fit, label=Y_fit, feature_names=FEAT_COND)
    dval = xgb.DMatrix(X_val, label=Y_val, feature_names=FEAT_COND)

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
    print(f"\nTraining conditional XGBoost (multi_output_tree, "
          f"max_depth={args.max_depth}, lr={args.lr})...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=args.num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stop,
        verbose_eval=25,
    )
    best_iter = booster.best_iteration
    print(f"\nbest_iteration = {best_iter}")

    # Honest val metrics, headline at the publish horizon.
    P_val = booster.predict(dval, iteration_range=(0, best_iter + 1))
    pub = min(args.publish_h, args.h_max)
    pub_mask = (val["h"].astype(int) == pub).to_numpy()
    print(f"\n===== HONEST VAL @ h={pub} (n={int(pub_mask.sum()):,}) =====")
    print(f"{'event':<22} {'base%':>7} {'AU-PR':>7} {'AP_lift':>8} "
          f"{'AUC':>6} {'Brier':>7}")
    rows = []
    for k, ev in enumerate(EVENTS):
        y = Y_val[pub_mask, k].astype(int)
        p = P_val[pub_mask, k]
        base = float(y.mean()) if len(y) else float("nan")
        ap = float(average_precision_score(y, p)) if y.sum() else float("nan")
        auc = (float(roc_auc_score(y, p))
               if 0 < y.sum() < len(y) else float("nan"))
        brier = float(brier_score_loss(y, p)) if len(y) else float("nan")
        lift = ap / base if base else float("nan")
        rows.append({"event": ev, "horizon": pub, "base": base, "ap": ap,
                     "ap_lift": lift, "auc": auc, "brier": brier})
        print(f"{ev:<22} {base*100:>6.2f} {ap:>7.3f} {lift:>8.2f} "
              f"{auc:>6.3f} {brier:>7.4f}")

    # Per-horizon AP curve (shows trajectory quality across h).
    print(f"\n===== VAL AP by horizon =====")
    print(f"{'h':>3} " + " ".join(f"{ev[:10]:>11}" for ev in EVENTS))
    for h in range(1, args.h_max + 1):
        m = (val["h"].astype(int) == h).to_numpy()
        cells = []
        for k in range(len(EVENTS)):
            y = Y_val[m, k].astype(int)
            p = P_val[m, k]
            ap = average_precision_score(y, p) if y.sum() else float("nan")
            cells.append(f"{ap:>11.3f}")
        print(f"{h:>3} " + " ".join(cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({
            "model": booster, "scaler": scaler,
            "feature_names": list(FEAT_COND),
            "events": list(EVENTS),
            "best_iteration": int(best_iter),
            "h_max": int(args.h_max),
            "publish_h": int(args.publish_h),
            "version": "v2.1c",
            "kind": "joint_xgb_cond_horizon",
            "metrics_val": rows,
            "params": params,
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
