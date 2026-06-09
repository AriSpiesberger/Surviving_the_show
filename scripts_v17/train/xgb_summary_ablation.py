"""XGB-stage ablation: add a compact CURRENT point-in-time scouting summary
directly to the joint XGB and see if it helps beyond the hazards.

The joint XGB normally sees only hazard probs + age/yip. Here we append the
player's scouting state AS OF the snapshot (latest grade with season <= snap,
via backward merge_asof -- no lookahead, no cumulative/peak derivations, just
the occurrent values): fv, ovr_rank, eta_gap, risk, is_scouted.

Runs on the grades-ON OOF (hazards already reflect grades). XGB stage only, so
it's fast -- no hazard rerun.

Usage:  python -m scripts_v17.train.xgb_summary_ablation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts_v17.train.fit_joint_xgb_v2 import EVENTS, FEAT, _prep

EVENT_WEIGHTS = {"TOP_100_PROSPECT": 1.0, "MLB_DEBUT": 2.0,
                 "ESTABLISHED_MLB": 1.0, "STAR_PLUS_ELITE": 1.0}

REPO = Path(__file__).resolve().parents[2]
DB = str(REPO / "prospects_snapshot.db")
SCOUT = REPO / "scratch" / "fangraphs_board" / "scouting_grades_pointintime.csv"
FIT = REPO / "results" / "training" / "v2.0b_oof_stacked_long_gradeson.csv"
VAL = REPO / "results" / "training" / "v2.0b_oof_val_long_gradeson.csv"

SUMMARY = ["scout_fv", "scout_ovr_rank", "scout_eta_gap", "scout_risk",
           "scout_is_scouted"]
XGB_PARAMS = dict(tree_method="hist", multi_strategy="multi_output_tree",
                  objective="binary:logistic", eval_metric="logloss",
                  max_depth=6, eta=0.05, seed=42, verbosity=0)


def scout_table():
    s = pd.read_csv(SCOUT, low_memory=False)
    s = (s.sort_values(["player_id", "season", "source"])
         .drop_duplicates(["player_id", "season"], keep="first"))
    return s[["player_id", "season", "fv", "ovr_rank", "eta", "risk"]].sort_values("season")


def attach_summary(long_df, s):
    m = pd.merge_asof(long_df.sort_values("snap_year"), s,
                      left_on="snap_year", right_on="season", by="player_id",
                      direction="backward")
    m["scout_fv"] = m["fv"]
    m["scout_ovr_rank"] = m["ovr_rank"]
    m["scout_risk"] = m["risk"]
    m["scout_eta_gap"] = m["eta"] - m["snap_year"]
    m["scout_is_scouted"] = m["season"].notna().astype(float)
    return m


def fit_eval(fit_df, val_df, feat):
    Ytr = fit_df[[f"realized_{e}" for e in EVENTS]].values.astype(np.float32)
    dtr = xgb.DMatrix(fit_df[feat].values.astype(np.float32), label=Ytr,
                      feature_names=feat)
    dva = xgb.DMatrix(val_df[feat].values.astype(np.float32),
                      label=val_df[[f"realized_{e}" for e in EVENTS]].values.astype(np.float32),
                      feature_names=feat)
    bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=1000, evals=[(dva, "v")],
                    early_stopping_rounds=25, verbose_eval=False)
    P = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    out = val_df.copy()
    for k, e in enumerate(EVENTS):
        out[f"xp_{e}"] = P[:, k]
    return out


def wap(df):
    """Return {event: {base,ap,auc,n}} + weighted-AP."""
    o, s, w = {}, 0.0, 0.0
    for ev in EVENTS:
        sub = df[df.get(f"eligible_{ev}", 0) == 1]
        y = sub[f"realized_{ev}"].astype(int).values
        p = sub[f"xp_{ev}"].astype(float).values
        if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
            continue
        ap = float(average_precision_score(y, p))
        o[ev] = {"base": float(y.mean()), "ap": ap,
                 "auc": float(roc_auc_score(y, p)), "n": int(len(y))}
        s += EVENT_WEIGHTS[ev] * ap
        w += EVENT_WEIGHTS[ev]
    o["WEIGHTED"] = {"ap": s / w if w else float("nan")}
    return o


def main():
    fit = _prep(pd.read_csv(FIT), DB, 2020).reset_index(drop=True)
    val = _prep(pd.read_csv(VAL), DB, 2020).reset_index(drop=True)
    s = scout_table()
    fit_m, val_m = attach_summary(fit, s), attach_summary(val, s)
    print(f"summary cols: {SUMMARY}")
    print(f"val rows scouted: {val_m.scout_is_scouted.mean():.0%}\n")

    base = fit_eval(fit, val, FEAT)
    summ = fit_eval(fit_m, val_m, FEAT + SUMMARY)

    for slc in ("full", "2013+"):
        a = wap(base if slc == "full" else base[base.entry_year >= 2013])
        b = wap(summ if slc == "full" else summ[summ.entry_year >= 2013])
        print(f"=== XGB | {slc} val ===")
        print(f"{'event':<20}{'base%':>7}{'n':>8} | {'AP_no':>7}{'AP_sum':>8}{'dAP':>8}"
              f" | {'AUC_no':>7}{'AUC_sum':>8}")
        for ev in EVENTS:
            if ev not in a and ev not in b:
                continue
            ra, rb = a.get(ev, {}), b.get(ev, {})
            print(f"{ev:<20}{rb.get('base', ra.get('base', float('nan')))*100:>6.1f}"
                  f"{rb.get('n', ra.get('n', 0)):>8,} | "
                  f"{ra.get('ap', float('nan')):>7.3f}{rb.get('ap', float('nan')):>8.3f}"
                  f"{rb.get('ap', float('nan'))-ra.get('ap', float('nan')):>+8.3f} | "
                  f"{ra.get('auc', float('nan')):>7.3f}{rb.get('auc', float('nan')):>8.3f}")
        print(f"{'WEIGHTED-AP':<20}{'':>15} | {a['WEIGHTED']['ap']:>7.3f}"
              f"{b['WEIGHTED']['ap']:>8.3f}{b['WEIGHTED']['ap']-a['WEIGHTED']['ap']:>+8.3f}\n")


if __name__ == "__main__":
    main()
