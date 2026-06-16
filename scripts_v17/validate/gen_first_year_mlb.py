"""First-year-MLB ceiling list: recent debutees (2025-26), scored for
ESTABLISHED / STAR from their final pre-debut data, with eBay card prices.

These players have left the prospect buy universe (they debuted), so we score
each at snap = debut_year - 1 (their last prospect season) with the production
hazards + the W=0 ceiling XGB — i.e. the model's pre-debut ceiling projection.

    python -m scripts_v17.validate.gen_first_year_mlb
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3
import xgboost as xgb

from prospects.storage import ProspectDB
from scripts_v17.train.train_v2_0b_prod import score_snap_with_landmark
from scripts_v17.train.fit_joint_xgb_v2 import HAZARD_PROBS, AGE_CENTER, YIP_CENTER
from prospects.features.scouting_grades import attach_scouting_summary

DB = "prospects_snapshot.db"
HAZ = "models/event_classifiers_v2.0b_prod.pkl"
CEIL = "models/joint_xgb_v2.0b_ceiling_w0.pkl"
DEBUT_YEARS = [2025, 2026]
OUT = "results/buy_lists/first_year_mlb_ceiling.csv"


def main():
    db = ProspectDB(DB)
    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb, o.year_top_100,
                   o.year_top_25, o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.events_json,
                   o.final_mlb_year
            FROM prospects p LEFT JOIN career_outcomes o ON o.player_id=p.player_id
            WHERE o.mlb_debut_year IN (%s)
        """ % ",".join("?" * len(DEBUT_YEARS)), DEBUT_YEARS).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    print(f"first-year-MLB players (debut {DEBUT_YEARS}): {len(prospects)}")

    hazards = pickle.load(open(HAZ, "rb"))
    longs = []
    for D in DEBUT_YEARS:
        cohort = [p for p in prospects if p.get("mlb_debut_year") == D]
        if not cohort:
            continue
        out = Path(f"scratch/fy_{D}.csv")
        score_snap_with_landmark(hazards, cohort, stats_by_pid,
                                 snap_year=D - 1, out_csv=out, verbose=False)
        longs.append(pd.read_csv(out))
    big = pd.concat(longs, ignore_index=True)

    c = sqlite3.connect(DB)
    meta = pd.read_sql("SELECT player_id, birth_date, primary_position, "
                       "current_org FROM prospects", c)
    c.close()
    meta["birth_year"] = pd.to_datetime(meta.birth_date, errors="coerce").dt.year
    big = big.drop(columns=[x for x in ("primary_position", "current_org")
                            if x in big.columns]).merge(meta, on="player_id", how="left")
    big["age_at_snap_centered"] = (big.snap_year - big.birth_year).fillna(22.) - AGE_CENTER
    big["years_in_pro"] = big.snap_offset
    big["yip_centered"] = big.snap_offset - YIP_CENTER
    for p in HAZARD_PROBS:
        big[f"{p}_x_yip_centered"] = big[p] * big.yip_centered
    big = attach_scouting_summary(big)

    bd = pickle.load(open(CEIL, "rb"))
    X = bd["scaler"].transform(big[bd["feature_names"]].values.astype(np.float32))
    P = bd["model"].predict(xgb.DMatrix(X, feature_names=list(bd["feature_names"])),
                            iteration_range=(0, bd["best_iteration"] + 1))
    ev = bd["events"]
    big["p_ESTABLISHED_MLB"] = P[:, ev.index("ESTABLISHED_MLB")]
    big["p_STAR_PLUS_ELITE"] = P[:, ev.index("STAR_PLUS_ELITE")]

    pr = pd.read_csv("data/prices_bowman_chrome_auto_v13.csv")
    if "denominator" in pr.columns:
        pr = pr[pr.denominator.astype(str).isin(["0", "0.0"])]
    if "has_market" in pr.columns:
        pr = pr[pr.has_market.astype(str) == "1"]
    pr = pr.rename(columns={"price_median": "ebay_price_median",
                            "n_listings": "ebay_n_listings"})
    pcols = ["player_id"] + [c for c in ("ebay_price_median", "ebay_n_listings")
                             if c in pr.columns]
    big = big.merge(pr[pcols].drop_duplicates("player_id"), on="player_id", how="left")

    out_cols = ["player_id", "name", "mlb_debut_year", "primary_position",
                "current_org", "age_at_snap_centered", "p_ESTABLISHED_MLB",
                "p_STAR_PLUS_ELITE", "ebay_price_median", "ebay_n_listings"]
    res = big[[c for c in out_cols if c in big.columns]].copy()
    res["age"] = (res.age_at_snap_centered + AGE_CENTER).round().astype(int)
    res = res.drop(columns=["age_at_snap_centered"]).sort_values(
        "p_ESTABLISHED_MLB", ascending=False)
    res.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(res)} rookies)")
    print(res.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
