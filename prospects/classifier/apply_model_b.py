"""Apply Model B to buy list — adds P(cup/utility/regular/breakout | debut)."""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

MODEL = "model_b_outcomes_v1.15.pkl"
POSITION_LOOKUP = "player_position_from_stats.csv"
BUCKET_DUMMIES = ["b_R1", "b_R10+", "b_R2-R3", "b_R4-R10"]
POS_DUMMIES = ["pos_IF", "pos_OF", "pos_OTH"]


def pos_group(p):
    p = str(p).upper()
    if p == "C": return "C"
    if p in ("1B", "2B", "3B", "SS"): return "IF"
    if p in ("LF", "CF", "RF", "OF", "DH"): return "OF"
    # Everything else — P/RHP/LHP/SP/RP/multi-position — matches training OTH bucket
    return "OTH"


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    df = pd.read_csv(src)
    print(f"Loaded {len(df):,} rows from {src}")

    pos_lk = pd.read_csv(POSITION_LOOKUP)
    df = df.merge(pos_lk, on="player_id", how="left")
    # Prefer season_stats position; fall back to prospects.primary_position
    df["position_corrected"] = df["pos_seasonstats"].fillna(df["primary_position"])
    n_fixed = (df["pos_seasonstats"].notna()
               & (df["pos_seasonstats"] != df["primary_position"])).sum()
    print(f"  position corrected from season_stats for {n_fixed:,} rows")

    with open(MODEL, "rb") as fh:
        m = pickle.load(fh)
    clf, sc, feature_names, classes = m["model"], m["scaler"], m["feature_names"], m["classes"]

    haz_map = {
        "logit_p_TOP_100_PROSPECT": "p_TOP_100_PROSPECT_raw",
        "logit_p_MLB_DEBUT": "p_MLB_DEBUT_raw",
        "logit_p_ESTABLISHED_MLB": "p_ESTABLISHED_MLB_raw",
        "logit_p_STAR_PLUS_ELITE": "p_STAR_PLUS_ELITE_raw",
    }
    eps = 1e-6
    feat = pd.DataFrame()
    for logit_col, src_col in haz_map.items():
        p = df[src_col].clip(eps, 1 - eps)
        feat[logit_col] = np.log(p / (1 - p))
    feat["yrs_pre_debut"] = 0.0
    for b in BUCKET_DUMMIES:
        feat[b] = (df["bucket"] == b.replace("b_", "")).astype(float)
    df["pos_grp"] = df["position_corrected"].apply(pos_group)
    for p in POS_DUMMIES:
        feat[p] = (df["pos_grp"] == p.replace("pos_", "")).astype(float)
    feat = feat[feature_names]

    proba = clf.predict_proba(sc.transform(feat.values))
    for i, cls in enumerate(classes):
        df[f"p_{cls}_given_debut"] = proba[:, i]

    p_debut = df["p_MLB_DEBUT_raw"].clip(0, 1)
    df["p_no_debut"] = 1 - p_debut
    for cls in classes:
        df[f"p_{cls}"] = p_debut * df[f"p_{cls}_given_debut"]

    POP = {"cup": 15.0, "utility": 25.0, "regular": 50.0, "breakout": 150.0,
           "no_debut": 3.0}
    df["EV_dollars"] = (
        df["p_no_debut"] * POP["no_debut"]
        + df["p_cup"] * POP["cup"]
        + df["p_utility"] * POP["utility"]
        + df["p_regular"] * POP["regular"]
        + df["p_breakout"] * POP["breakout"]
    )

    entry = df["ebay_price_median"].fillna(df["ebay_price_mean"])
    df["entry_price"] = entry
    df["edge_dollars"] = df["EV_dollars"] - entry

    df = df.sort_values("EV_dollars", ascending=False).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

    print("\nTop 25 by EV:")
    cols = ["name", "bucket", "primary_position", "buy_score",
            "p_no_debut", "p_cup", "p_utility", "p_regular", "p_breakout",
            "entry_price", "EV_dollars", "edge_dollars"]
    show = df[cols].head(25).copy()
    for c in ["p_no_debut", "p_cup", "p_utility", "p_regular", "p_breakout"]:
        show[c] = (show[c] * 100).round(1)
    for c in ["entry_price", "EV_dollars", "edge_dollars"]:
        show[c] = show[c].round(1)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
