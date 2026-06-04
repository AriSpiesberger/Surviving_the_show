"""v2.x joint hazards — XGBoost multi_output_tree across 5 event heads.

Replaces the v1.17 architecture of 6 independent HistGradientBoosting
classifiers with ONE booster outputting 5 hazards per row.

Heads:
  TOP_100_PROSPECT   year_top_100
  MLB_DEBUT          mlb_debut_year
  ESTABLISHED_MLB    year_established_mlb
  STAR_PLUS_ELITE    min(year_all_star_once, year_all_star_three,
                          year_major_award, year_hof_trajectory)
  EXIT_BASEBALL      last_active_year (the year the player stopped playing)

Per-row labels: realized_<event> at year t = 1 iff trigger_<event> == t.
For already-triggered events, label = 0 (it triggered previously, not now).
The model learns from features which events are still in play.

Row inclusion: every panel row where year <= last_active_year (= player was
in baseball that year). Right-censoring is implicit: still-active players'
event labels stay at 0 because we haven't observed the trigger yet.

Split: 80% train / 10% cal / 10% val from the v1.17 lasso_fit / lasso_val
player files. Hazards train on the 80%.
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

from prospects.features.scouting import FEATURE_NAMES, N_FEATURES

EVENTS = [
    "TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
    "STAR_PLUS_ELITE", "EXIT_BASEBALL",
]

TRIGGER_COLS = {
    "TOP_100_PROSPECT": "year_top_100",
    "MLB_DEBUT": "mlb_debut_year",
    "ESTABLISHED_MLB": "year_established_mlb",
    # STAR_PLUS_ELITE is composite (min of components)
    # EXIT_BASEBALL is derived from last-active
}
STAR_PLUS_COLS = (
    "year_all_star_once", "year_all_star_three",
    "year_major_award", "year_hof_trajectory",
)


def _int_or_none(v):
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _star_plus_trigger(p: dict) -> int | None:
    cands = [v for v in (_int_or_none(p.get(c)) for c in STAR_PLUS_COLS)
             if v is not None]
    return min(cands) if cands else None


def _load_pids(path: str) -> set[str]:
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panels/panel_v1.17.npz")
    ap.add_argument("--joined", default="panels/panel_v1.17.joined.pkl")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--fit-players", required=True,
                    help="10% cal slice (v1.17 lasso_fit_players.txt)")
    ap.add_argument("--val-players", required=True,
                    help="10% val slice (v1.17 lasso_val_players.txt)")
    ap.add_argument("--num-rounds", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-child-weight", type=int, default=30)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/hazards_xgb_v2.x.pkl")
    args = ap.parse_args()

    print(f"Loading panel {args.panel}")
    with np.load(args.panel, allow_pickle=True) as d:
        X = d["X"].astype(np.float32, copy=False)
        pids = np.asarray(d["pids"])
        years = np.asarray(d["years"], dtype=int)
    assert X.shape[1] == N_FEATURES
    print(f"  panel: {X.shape[0]:,} rows × {X.shape[1]} features, "
          f"{len(set(pids.tolist())):,} players")

    print(f"Loading joined {args.joined}")
    with open(args.joined, "rb") as fh:
        joined = pickle.load(fh)
    assert len(joined) == X.shape[0]

    print(f"Loading stats_max_by_pid from {args.db}")
    c = sqlite3.connect(args.db)
    sm = pd.read_sql(
        "SELECT player_id, MAX(season_year) AS y "
        "FROM season_stats GROUP BY player_id", c)
    c.close()
    stats_max = dict(zip(sm["player_id"], sm["y"].astype(int)))

    # Per-player trigger years (cached)
    print("Computing per-player trigger years...")
    trig_by_pid: dict[str, dict[str, int | None]] = {}
    last_active_by_pid: dict[str, int | None] = {}
    for p in joined:
        pid = p["player_id"]
        if pid in trig_by_pid:
            continue
        trig_by_pid[pid] = {
            ev: _int_or_none(p.get(col))
            for ev, col in TRIGGER_COLS.items()
        }
        trig_by_pid[pid]["STAR_PLUS_ELITE"] = _star_plus_trigger(p)
        fy = _int_or_none(p.get("final_mlb_year"))
        sm_v = stats_max.get(pid)
        if fy is None and sm_v is None:
            last_active_by_pid[pid] = None
        elif fy is None:
            last_active_by_pid[pid] = int(sm_v)
        elif sm_v is None:
            last_active_by_pid[pid] = int(fy)
        else:
            last_active_by_pid[pid] = int(max(fy, sm_v))
        # EXIT trigger = last_active_year (the player's last in-baseball year)
        trig_by_pid[pid]["EXIT_BASEBALL"] = last_active_by_pid[pid]

    cal_pids = _load_pids(args.fit_players)
    val_pids = _load_pids(args.val_players)
    print(f"  cal slice (10%): {len(cal_pids):,}  "
          f"val slice (10%): {len(val_pids):,}")

    print("Building per-row labels for 5 heads...")
    n = X.shape[0]
    Y = np.zeros((n, len(EVENTS)), dtype=np.float32)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        pid = pids[i]
        yr = int(years[i])
        la = last_active_by_pid.get(pid)
        if la is None or yr > la:
            # Player exited baseball before this row's year — skip.
            continue
        keep[i] = True
        trigs = trig_by_pid[pid]
        for k, ev in enumerate(EVENTS):
            t = trigs.get(ev)
            if t is not None and t == yr:
                Y[i, k] = 1.0

    n_kept = int(keep.sum())
    print(f"  kept {n_kept:,} / {n:,} rows (active-in-baseball)")
    for k, ev in enumerate(EVENTS):
        pos = int(Y[keep, k].sum())
        print(f"  {ev:<20}  realized=1: {pos:,} "
              f"({pos / max(n_kept, 1):.2%})")

    # 80% train / 10% cal / 10% val by player
    is_train = np.array([p not in cal_pids and p not in val_pids
                          for p in pids], dtype=bool)
    is_val = np.array([p in val_pids for p in pids], dtype=bool)
    train_mask = keep & is_train
    val_mask = keep & is_val
    print(f"  train rows: {int(train_mask.sum()):,}  "
          f"val rows: {int(val_mask.sum()):,}")

    # NOTE: XGBoost's hist tree_method handles NaN natively — no impute.
    dtrain = xgb.DMatrix(X[train_mask], label=Y[train_mask],
                          feature_names=list(FEATURE_NAMES))
    dval = xgb.DMatrix(X[val_mask], label=Y[val_mask],
                        feature_names=list(FEATURE_NAMES))

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
    print(f"\nTraining XGBoost (multi_output_tree, "
          f"max_depth={args.max_depth}, lr={args.lr}, "
          f"early_stop={args.early_stop})...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=args.num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stop,
        verbose_eval=25,
    )
    best_iter = booster.best_iteration
    print(f"\nbest_iteration = {best_iter}")

    print(f"\n===== HONEST VAL (n={int(val_mask.sum()):,}) =====")
    P_val = booster.predict(dval, iteration_range=(0, best_iter + 1))
    Y_val = Y[val_mask]
    print(f"{'event':<22} {'base%':>7} {'AU-PR':>7} {'AP_lift':>8} "
          f"{'AUC':>6} {'Brier':>7}")
    rows_metrics = []
    for k, ev in enumerate(EVENTS):
        y = Y_val[:, k].astype(int)
        p = P_val[:, k]
        base = float(y.mean())
        ap = float(average_precision_score(y, p)) if y.sum() else float("nan")
        auc = (float(roc_auc_score(y, p)) if 0 < y.sum() < len(y)
               else float("nan"))
        brier = float(brier_score_loss(y, p))
        lift = ap / base if base else float("nan")
        rows_metrics.append({
            "event": ev, "base": base, "ap": ap, "ap_lift": lift,
            "auc": auc, "brier": brier, "n_pos": int(y.sum()),
        })
        print(f"{ev:<22} {base*100:>6.2f} {ap:>7.3f} {lift:>8.2f} "
              f"{auc:>6.3f} {brier:>7.4f}")

    with open(args.out, "wb") as fh:
        pickle.dump({
            "model": booster,
            "feature_names": list(FEATURE_NAMES),
            "events": list(EVENTS),
            "best_iteration": int(best_iter),
            "version": "v2.x_joint_hazards",
            "kind": "xgb_multi_output_tree",
            "params": params,
            "metrics_val": rows_metrics,
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
