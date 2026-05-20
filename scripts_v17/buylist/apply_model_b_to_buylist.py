"""Apply Model B (v1.14n) to the buy list: P(cup/utility/regular/breakout | debut).

Inputs:
    - buy_list_v14n_FINAL.csv  (score>=0.40, non-R1, non-top100 — produced upstream)
    - models/model_b_outcomes_v1.14n.pkl  (trained on v1.14n fit+val debutees)
    - p_debut comes from the FILTERED VAL CELL RATE (calibrated empirical rate),
      not from raw p_MLB_DEBUT.

Output:
    buy_list_v14n_FINAL_with_outcomes.csv  with columns:
        p_debut, p_no_debut
        p_cup_given_debut, p_utility_given_debut, p_regular_given_debut, p_breakout_given_debut
        p_cup, p_utility, p_regular, p_breakout   (unconditional)
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

BL = "buy_list_v14n_FINAL.csv"
MODEL = "models/model_b_outcomes_v1.14n.pkl"
VAL_MATRIX = "scratch/log_bins/full_matrix_yip_x_bin.csv"  # already has val rates per cell
POSITION_LOOKUP = "models/player_position_from_stats.csv"
OUT = "buy_list_v14n_FINAL_with_outcomes.csv"

BUCKET_DUMMIES = ["b_R1", "b_R10+", "b_R2-R3", "b_R4-R10"]
POS_DUMMIES = ["pos_IF", "pos_OF", "pos_OTH"]


def pos_group(p):
    p = str(p).upper()
    if p == "C": return "C"
    if p in ("1B", "2B", "3B", "SS"): return "IF"
    if p in ("LF", "CF", "RF", "OF", "DH"): return "OF"
    return "OTH"


def main():
    df = pd.read_csv(BL)
    print(f"buy list: {len(df)} rows")

    # Calibrated P(debut) — empirical val rate for this player's (yip, score_bin)
    matrix = pd.read_csv(VAL_MATRIX)
    # Reconstruct same 0.1 score bins as the matrix uses
    edges = list(np.round(np.arange(-0.5, 1.0, 0.1), 2)) + [1.5, 2.0, 3.0, 5.0]
    labels = [f"{edges[i]:>+.2f} to {edges[i+1]:>+.2f}" for i in range(len(edges)-1)]
    df["sb"] = pd.cut(df["lasso_score"], bins=edges, labels=labels).astype(str)
    cal = matrix[["sb","snap_offset","val_rate","val_n"]].rename(
        columns={"val_rate":"p_debut", "val_n":"val_cell_n"})
    df = df.merge(cal, on=["sb","snap_offset"], how="left")
    print(f"  matched p_debut from val table for "
          f"{df.p_debut.notna().sum()}/{len(df)} rows")
    # Fallback: raw hazard p_MLB_DEBUT (clipped)
    df["p_debut"] = df["p_debut"].fillna(df["p_MLB_DEBUT"].clip(0.0, 0.95))
    df["p_no_debut"] = 1.0 - df["p_debut"]

    # Model B features (conditional on debut)
    with open(MODEL, "rb") as fh:
        m = pickle.load(fh)
    clf, sc, feat_names, classes = m["model"], m["scaler"], m["feature_names"], m["classes"]
    # classes order: cup, utility, regular, breakout
    print(f"  classes: {classes}")

    pos_lk = pd.read_csv(POSITION_LOOKUP)
    df = df.merge(pos_lk, on="player_id", how="left")
    df["position_corrected"] = df["pos_seasonstats"].fillna(df["primary_position"])
    df["pos_grp"] = df["position_corrected"].apply(pos_group)

    eps = 1e-6
    haz_map = {
        "logit_p_TOP_100_PROSPECT": "p_TOP_100_PROSPECT",
        "logit_p_MLB_DEBUT": "p_MLB_DEBUT",
        "logit_p_ESTABLISHED_MLB": "p_ESTABLISHED_MLB",
        "logit_p_STAR_PLUS_ELITE": "p_STAR_PLUS_ELITE",
    }
    feat = pd.DataFrame()
    for logit_col, src_col in haz_map.items():
        p = df[src_col].clip(eps, 1 - eps)
        feat[logit_col] = np.log(p / (1 - p))
    feat["yrs_pre_debut"] = df["snap_offset"].astype(float)
    for b in BUCKET_DUMMIES:
        feat[b] = (df["bucket"] == b.replace("b_", "")).astype(float)
    for p in POS_DUMMIES:
        feat[p] = (df["pos_grp"] == p.replace("pos_", "")).astype(float)
    feat = feat[feat_names]

    P = clf.predict_proba(sc.transform(feat.values))
    for i, cls in enumerate(classes):
        df[f"p_{cls}_given_debut"] = P[:, i]
        df[f"p_{cls}"] = df[f"p_{cls}_given_debut"] * df["p_debut"]

    cols_out = ["name","player_id","snap_offset","bucket","primary_position","current_org","draft_year",
                "lasso_score","sb","p_debut","val_cell_n","p_no_debut",
                "p_cup","p_utility","p_regular","p_breakout",
                "p_cup_given_debut","p_utility_given_debut","p_regular_given_debut","p_breakout_given_debut",
                "p_MLB_DEBUT","p_TOP_100_PROSPECT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE",
                "age_at_snap"]
    cols_out = [c for c in cols_out if c in df.columns]
    df_out = df[cols_out].sort_values(["snap_offset","p_breakout","p_regular"],
                                      ascending=[True, False, False])
    df_out.to_csv(OUT, index=False)
    print(f"\nSaved {OUT} ({len(df_out)} rows)\n")

    print("=== top players by P(breakout) ===")
    print(df_out.sort_values("p_breakout", ascending=False).head(20)[
        ["name","snap_offset","bucket","p_debut","p_breakout","p_regular","p_utility","p_cup"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
