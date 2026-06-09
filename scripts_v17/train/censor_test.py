"""Censoring experiment: does training the joint XGB on resolved-exposure rows
(keep all positives + negatives with >= W forward years) lift debut magnitude
without tanking AP? W=0 is the current censored baseline.

    python -m scripts_v17.train.censor_test
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from scripts_v17.train.fit_joint_xgb_v2 import EVENTS, FEAT, _prep

DB = "prospects_snapshot.db"
PARAMS = dict(tree_method="hist", multi_strategy="multi_output_tree",
              objective="binary:logistic", eval_metric="logloss",
              max_depth=6, learning_rate=0.05, min_child_weight=30,
              reg_lambda=1.0, seed=42)

fit0 = _prep(pd.read_csv("results/training/v2.0b_oof_stacked_long.csv"), DB, 2020).reset_index(drop=True)
val = _prep(pd.read_csv("results/training/v2.0b_oof_val_long.csv"), DB, 2020).reset_index(drop=True)
snap = _prep(pd.read_csv("results/scored/snap2026_v1.18b_landmark_long.csv"), DB, 2030).reset_index(drop=True)
lvl = pd.read_csv("results/buy_lists/buy_list_v2.0b_ALL_SCORED.csv")[["player_id", "cur_level_2026"]]
snap = snap.merge(lvl, on="player_id", how="left")

Yv = val[[f"realized_{e}" for e in EVENTS]].values.astype(np.float32)
rcols = [f"realized_{e}" for e in EVENTS]
print(f"{'W':>3} {'n_train':>9} {'debutAP':>8} {'estAP':>7} {'starAP':>7} "
      f"{'AAA_med':>8} {'A_med':>7} {'Pecko':>7}")
for W in [0, 4, 6, 8, 10]:
    if W == 0:
        fit = fit0
    else:
        pos = fit0[rcols].sum(axis=1) > 0
        fit = fit0[(fit0.years_fwd >= W) | pos].reset_index(drop=True)
    sc = StandardScaler().fit(fit[FEAT].values.astype(np.float32))
    Y = fit[rcols].values.astype(np.float32)
    dtr = xgb.DMatrix(sc.transform(fit[FEAT].values.astype(np.float32)), label=Y, feature_names=list(FEAT))
    dv = xgb.DMatrix(sc.transform(val[FEAT].values.astype(np.float32)), label=Yv, feature_names=list(FEAT))
    b = xgb.train(PARAMS, dtr, 500, evals=[(dv, "v")], early_stopping_rounds=25, verbose_eval=False)
    Pv = b.predict(dv, iteration_range=(0, b.best_iteration + 1))
    aps = [average_precision_score(Yv[:, k], Pv[:, k]) for k in range(len(EVENTS))]
    Ps = b.predict(xgb.DMatrix(sc.transform(snap[FEAT].values.astype(np.float32)), feature_names=list(FEAT)),
                   iteration_range=(0, b.best_iteration + 1))
    snap["deb"] = Ps[:, EVENTS.index("MLB_DEBUT")]
    aaa = snap[snap.cur_level_2026 == "AAA"].deb.median()
    a = snap[snap.cur_level_2026 == "A"].deb.median()
    pk = snap[snap.player_id == "draft_2023_ethan_pecko_r6p194"].deb
    pkv = float(pk.iloc[0]) if len(pk) else float("nan")
    print(f"{W:>3} {len(fit):>9,} {aps[1]:>8.3f} {aps[2]:>7.3f} {aps[3]:>7.3f} "
          f"{aaa:>8.3f} {a:>7.3f} {pkv:>7.3f}")
