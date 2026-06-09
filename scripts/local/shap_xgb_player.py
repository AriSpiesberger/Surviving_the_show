"""Joint-XGB SHAP for one player: does scout_fv/scout_ovr_rank drive the FINAL
est/star output (the layer the hazard SHAP can't see)?

Usage: python -m scripts.local.shap_xgb_player <player_id>
"""
from __future__ import annotations

import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from scripts_v17.train.fit_joint_xgb_v2 import (
    AGE_CENTER, EVENTS, HAZARD_PROBS, YIP_CENTER,
)
from prospects.features.scouting_grades import (
    SCOUTING_SUMMARY_COLS, attach_scouting_summary,
)

PID = sys.argv[1] if len(sys.argv) > 1 else "draft_2024_ethan_anderson_r2p61"
XGB = "models/joint_xgb_v2.0b_oof.pkl"
LONG = "results/scored/snap2026_v1.18b_landmark_long.csv"
DB = "prospects_snapshot.db"
SHOW = {"MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"}


def build_feat(df):
    c = sqlite3.connect(DB)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(birth.birth_date, errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = (df.snap_year - df.birth_year).fillna(22.0) - AGE_CENTER
    df["years_in_pro"] = df.snap_offset
    df["yip_centered"] = df.snap_offset - YIP_CENTER
    for p in HAZARD_PROBS:
        df[f"{p}_x_yip_centered"] = df[p] * df["yip_centered"]
    return attach_scouting_summary(df)


def main():
    b = pickle.load(open(XGB, "rb"))
    booster, scaler, feat, bi = b["model"], b["scaler"], b["feature_names"], b["best_iteration"]
    long = pd.read_csv(LONG, low_memory=False)
    bg = build_feat(long.sample(min(300, len(long)), random_state=0))
    Xbg = scaler.transform(bg[feat].values.astype(np.float32))
    row = build_feat(long[long.player_id == PID])
    if row.empty:
        print(f"{PID} not in {LONG}"); return
    Xp = scaler.transform(row[feat].values.astype(np.float32))

    def f(X, k):
        return booster.predict(xgb.DMatrix(X, feature_names=list(feat)),
                               iteration_range=(0, bi + 1))[:, k]

    print(f"{PID}\n  XGB features: {len(feat)} (scout summary: {SCOUTING_SUMMARY_COLS})\n")
    for k, ev in enumerate(EVENTS):
        if ev not in SHOW:
            continue
        expl = shap.Explainer(lambda X, k=k: f(X, k), Xbg, feature_names=list(feat))
        sv = expl(Xp).values[0]
        pred = float(f(Xp, k)[0])
        print(f"=== {ev}  XGB p={pred:.3f} ===")
        for i in np.argsort(-np.abs(sv)):
            star = "  <<< SCOUT" if feat[i] in SCOUTING_SUMMARY_COLS else ""
            print(f"  {feat[i]:<26}{row[feat[i]].values[0]:>10.3f}  shap={sv[i]:+.4f}{star}")
        print()


if __name__ == "__main__":
    main()
