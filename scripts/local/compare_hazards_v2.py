"""Compare v1.17h independent hazards vs v2.x joint hazards on the honest
val cohort, same panel rows."""
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects.features.scouting import FEATURE_NAMES

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "STAR_PLUS_ELITE"]


def main():
    v = pd.read_csv("v1.17h_val_long.csv")
    print(f"v1.17h val long: {len(v):,} rows, "
          f"{v.player_id.nunique():,} players")

    with open("models/hazards_xgb_v2.x.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    booster = bundle["model"]

    with np.load("panels/panel_v1.17.npz", allow_pickle=True) as d:
        X = d["X"].astype(np.float32, copy=False)
        pids = np.asarray(d["pids"])
        years = np.asarray(d["years"], dtype=int)
    idx = {(pids[i], int(years[i])): i for i in range(len(pids))}

    rows = np.array([idx.get((pid, int(yr)), -1)
                     for pid, yr in zip(v.player_id, v.snap_year)])
    mask = rows >= 0
    print(f"Matched panel rows: {int(mask.sum()):,} / {len(v):,}")

    X_v = X[rows[mask]]
    v_match = v[mask].reset_index(drop=True)
    dval = xgb.DMatrix(X_v, feature_names=list(FEATURE_NAMES))
    P = booster.predict(dval,
                         iteration_range=(0, bundle["best_iteration"] + 1))
    xgb_idx = {ev: bundle["events"].index(ev) for ev in EVENTS}

    print()
    print(f"{'event':<22} {'base%':>7} {'v1.17h AP':>10} {'v2.x AP':>9} "
          f"{'delta':>7} {'v1.17h AUC':>11} {'v2.x AUC':>9}")
    for ev in EVENTS:
        y = v_match[f"realized_{ev}"].values.astype(int)
        p_old = v_match[f"p_{ev}"].values
        p_new = P[:, xgb_idx[ev]]
        base = y.mean()
        if y.sum() == 0:
            continue
        ap_old = average_precision_score(y, p_old)
        ap_new = average_precision_score(y, p_new)
        auc_old = roc_auc_score(y, p_old)
        auc_new = roc_auc_score(y, p_new)
        print(f"{ev:<22} {base*100:>6.2f} {ap_old:>10.3f} {ap_new:>9.3f} "
              f"{ap_new - ap_old:>+7.3f} {auc_old:>11.3f} {auc_new:>9.3f}")


if __name__ == "__main__":
    main()
