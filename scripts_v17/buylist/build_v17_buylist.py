"""Final v1.17 buy list builder.

Pipeline:
1. Load snap=2026 hazards from snap2026_v17_all_long.csv (already produced
   by v1.17_prod hazards on full panel).
2. Apply debut_lasso_universe_v1.17.pkl → lasso_score.
3. Apply top100_lasso_v1.17.pkl → top100_score.
4. Apply model_b_outcomes_v1.17.pkl → P(cup/utility/regular/breakout | debut).
5. Compute UNIVERSE percentile ranks for p_breakout and top100_score.
6. Other probabilities stay as raw values.
7. Merge 2026 MiLB stats (PA-weighted across levels).
8. Merge eBay prices.
9. Universe filter: non-R1 AND never-top-100.
10. Per-yip threshold filter:
    yip=1: lasso>=1.86; yip=2: >=2.51; yip=3: >=3.90; yip=4: >=3.19; yip=5: >=3.79

Output: buy_list_v1.17_FINAL.csv
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd

FEAT = [
    "p_TOP_100_PROSPECT","p_MLB_DEBUT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE",
    "age_at_snap_centered","years_in_pro",
    "p_TOP_100_PROSPECT_x_yip_centered","p_MLB_DEBUT_x_yip_centered",
    "p_ESTABLISHED_MLB_x_yip_centered","p_STAR_PLUS_ELITE_x_yip_centered",
]
YIP_THRESHOLDS = {0: 4.691, 1: 1.938, 2: 2.754, 3: 4.226, 4: 4.853}  # HONEST, on overall_score
LEVEL_RANK = {'RK':1,'DSL':1,'FCL':1,'CPX':1,'ROK':1,'A-':2,'A':3,'A+':4,'AA':5,'AAA':6,'MLB':7}


def bucket(r):
    if int(r.is_international or 0)==1: return "IFA"
    if pd.isna(r.draft_round): return "IFA"
    dr = int(r.draft_round)
    if dr==1: return "R1"
    if dr<=3: return "R2-R3"
    if dr<=10: return "R4-R10"
    return "R10+"


def pos_group(p):
    p = str(p).upper()
    if p == "C": return "C"
    if p in ("1B","2B","3B","SS"): return "IF"
    if p in ("LF","CF","RF","OF","DH"): return "OF"
    return "OTH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-long",
                    default="results/scored/snap2026_v17_all_long.csv",
                    help="Output of score_panel_v17.py for the snap year.")
    ap.add_argument("--debut-lasso",
                    default="models/debut_lasso_universe_v1.17_prod.pkl")
    ap.add_argument("--top100-lasso",
                    default="models/top100_lasso_v1.17_prod.pkl")
    ap.add_argument("--model-b",
                    default="models/model_b_outcomes_v1.17_prod.pkl")
    ap.add_argument("--out-prefix",
                    default="results/buy_lists/buy_list_v1.17",
                    help="Output files: <prefix>_ALL_SCORED.csv, "
                         "<prefix>_FINAL.csv, plus a dated copy under "
                         "results/buy_lists/history/.")
    ap.add_argument("--db", default="prospects_snapshot.db")
    args = ap.parse_args()
    db = args.db

    print(f"Loading scored snap prospects from {args.snap_long}...")
    df = pd.read_csv(args.snap_long)
    df = df[df.snap_year == 2026].copy()
    df = df.drop_duplicates("player_id")
    print(f"  {len(df):,} unique players")

    # Merge prospect meta + bucket
    c = sqlite3.connect(db)
    meta = pd.read_sql("""SELECT p.player_id, p.primary_position, p.current_org,
                                p.draft_year, p.draft_round, p.is_international,
                                p.birth_date,
                                o.year_top_100 AS first_top100_yr,
                                o.mlb_debut_year
                         FROM prospects p
                         LEFT JOIN career_outcomes o ON o.player_id = p.player_id""", c)
    pos_lk = pd.read_csv("models/player_position_from_stats.csv")
    stats_all = pd.read_sql("""SELECT player_id, season_year, level, pa, ip,
                                      avg, obp, slg, iso, k_pct, bb_pct,
                                      home_runs, stolen_bases,
                                      era, k9, bb9, whip, fip
                              FROM season_stats""", c)
    c.close()

    df = df.merge(meta, on="player_id", how="left", suffixes=("","_m"))
    df["bucket"] = df.apply(bucket, axis=1)
    df["birth_year"] = pd.to_datetime(df["birth_date"], errors="coerce").dt.year
    df["age_at_snap"] = (df["snap_year"] - df["birth_year"]).fillna(22.0)
    df["age_at_snap_centered"] = df["age_at_snap"] - 22
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - 3
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]

    print(f"Applying debut lasso: {args.debut_lasso}")
    with open(args.debut_lasso,"rb") as fh:
        m_debut = pickle.load(fh)
    df["lasso_score"] = m_debut["lasso"].predict(m_debut["scaler"].transform(df[FEAT].values))

    print(f"Applying top100 lasso: {args.top100_lasso}")
    with open(args.top100_lasso,"rb") as fh:
        m_top100 = pickle.load(fh)
    df["top100_score"] = m_top100["lasso"].predict(m_top100["scaler"].transform(df[FEAT].values))

    print(f"Applying model B: {args.model_b}")
    with open(args.model_b,"rb") as fh:
        m_b = pickle.load(fh)
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
    feat_b = pd.DataFrame()
    for lc, sc_col in haz_map.items():
        p = df[sc_col].clip(eps, 1-eps).values
        feat_b[lc] = np.log(p / (1-p))
    feat_b["yrs_pre_debut"] = df["snap_offset"].astype(float).values
    for b in ("b_R1","b_R10+","b_R2-R3","b_R4-R10"):
        feat_b[b] = (df["bucket"] == b.replace("b_","")).astype(float).values
    for p in ("pos_IF","pos_OF","pos_OTH"):
        feat_b[p] = (df["pos_grp"] == p.replace("pos_","")).astype(float).values
    feat_b = feat_b[m_b["feature_names"]]
    P = m_b["model"].predict_proba(m_b["scaler"].transform(feat_b.values))
    for i, cls in enumerate(m_b["classes"]):
        df[f"p_{cls}_given_debut"] = P[:, i]

    # Use calibrated p_debut from val cell rate per (yip, lasso_score_bin) — fallback to clipped raw
    # Simpler: use raw p_MLB_DEBUT clipped to [0, 0.95] as the conditional probability
    df["p_debut"] = df["p_MLB_DEBUT"].clip(0.0, 0.95)
    df["p_no_debut"] = 1.0 - df["p_debut"]
    for cls in m_b["classes"]:
        df[f"p_{cls}"] = df[f"p_{cls}_given_debut"] * df["p_debut"]

    # 4b. Composite overall: debut amplified by breakout (α=2) and top100 (β=0.75)
    ALPHA, BETA = 1.06, 2.60
    df["overall_score"] = df["lasso_score"] * (
        1.0 + ALPHA * df["p_breakout_given_debut"] + BETA * df["p_TOP_100_PROSPECT"]
    )

    # 5. Universe filter for percentile computation + buy list
    universe = (df["bucket"] != "R1") & (df["first_top100_yr"].isna())
    print(f"Universe: {universe.sum():,} of {len(df):,} players")

    # 6. Percentile ranks within universe for p_breakout and top100_score
    uni_df = df[universe].copy()
    df["pct_breakout"] = np.nan
    df["pct_top100"] = np.nan
    df.loc[universe, "pct_breakout"] = uni_df["p_breakout"].rank(pct=True) * 100
    df.loc[universe, "pct_top100"] = uni_df["top100_score"].rank(pct=True) * 100

    # 7. Merge 2026 stats. Two views per player:
    #    (a) PA/IP-weighted BLEND across all 2026 levels (y2026_*) — historical
    #        default; useful for season-totals like HR/SB and for stable-level
    #        guys.
    #    (b) CURRENT-LEVEL-ONLY (y2026_cur_*) — uses only rows at the player's
    #        highest 2026 level. For recently-promoted players (e.g. an AA guy
    #        with 13 PA at AAA) the blended slash is dominated by the lower
    #        level while cur_level_2026 says AAA; (b) makes that mismatch
    #        legible. recently_promoted=True flags <50 PA / <15 IP at the
    #        current level when they ALSO have meaningful stats at a lower
    #        level the same season.
    print("Merging 2026 MiLB stats (blended + current-level-only)...")
    s26 = stats_all[stats_all.season_year == 2026].copy()
    s26["pa"] = s26["pa"].fillna(0); s26["ip"] = s26["ip"].fillna(0)
    s26["lvl_rank"] = s26["level"].str.upper().map(LEVEL_RANK).fillna(0)
    # Highest level
    s26 = s26.sort_values(["player_id","lvl_rank"], ascending=[True, False])
    high = s26.groupby("player_id").first().reset_index()[["player_id","level","lvl_rank"]]
    high = high.rename(columns={"level":"cur_level_2026",
                                "lvl_rank":"_cur_lvl_rank"})
    sums = s26.groupby("player_id").agg(
        y2026_pa=("pa","sum"), y2026_ip=("ip","sum"),
        y2026_home_runs=("home_runs","sum"),
        y2026_stolen_bases=("stolen_bases","sum"),
    ).reset_index()

    def wavg(s, val, w):
        d = s[[val, w]].dropna(subset=[val])
        d = d[d[w] > 0]
        if len(d) == 0: return np.nan
        return float((d[val] * d[w]).sum() / d[w].sum())

    # Build BLEND + CUR rows side by side
    s26_with_high = s26.merge(high[["player_id","_cur_lvl_rank"]], on="player_id", how="left")
    rows = []
    for pid, grp in s26_with_high.groupby("player_id"):
        cur_mask = grp["lvl_rank"] == grp["_cur_lvl_rank"].iloc[0]
        cur = grp[cur_mask]
        lower = grp[~cur_mask]
        r = {"player_id": pid}
        # Blended (PA/IP-weighted across all levels)
        for v in ("avg","obp","slg","iso","k_pct","bb_pct"):
            r[f"y2026_{v}"] = wavg(grp, v, "pa")
        for v in ("era","k9","bb9","whip","fip"):
            r[f"y2026_{v}"] = wavg(grp, v, "ip")
        # Current-level-only
        r["y2026_cur_pa"] = float(cur["pa"].sum())
        r["y2026_cur_ip"] = float(cur["ip"].sum())
        for v in ("avg","obp","slg","iso","k_pct","bb_pct"):
            r[f"y2026_cur_{v}"] = wavg(cur, v, "pa")
        for v in ("era","k9","bb9","whip","fip"):
            r[f"y2026_cur_{v}"] = wavg(cur, v, "ip")
        # Promotion flag: small sample at cur level AND meaningful lower-level
        # sample the same season => "their headline blend is mostly lower level"
        lower_pa = float(lower["pa"].sum()); lower_ip = float(lower["ip"].sum())
        cur_pa = r["y2026_cur_pa"]; cur_ip = r["y2026_cur_ip"]
        is_hitter = cur_pa + lower_pa > cur_ip + lower_ip  # rough
        if is_hitter:
            r["recently_promoted"] = bool(cur_pa < 50 and lower_pa >= 50)
        else:
            r["recently_promoted"] = bool(cur_ip < 15 and lower_ip >= 15)
        rows.append(r)
    wstats = pd.DataFrame(rows)
    df = df.merge(high.drop(columns=["_cur_lvl_rank"]), on="player_id", how="left")
    df = df.merge(sums, on="player_id", how="left")
    df = df.merge(wstats, on="player_id", how="left")

    # 8. Apply per-yip threshold filter (within universe)
    def passes(r):
        if not (r["bucket"] != "R1" and pd.isna(r["first_top100_yr"])):
            return False
        yip = r["snap_offset"]
        thresh = YIP_THRESHOLDS.get(yip)
        if thresh is None:
            return False
        return r["overall_score"] >= thresh
    df["passes_filter"] = df.apply(passes, axis=1)
    print(f"Players passing yip-threshold filter (universe): {df['passes_filter'].sum():,}")

    # 9. Merge eBay prices
    print("Merging eBay prices...")
    try:
        prices = pd.read_csv("data/prices_bowman_chrome_auto_v13.csv")
        prices = prices[prices["denominator"].astype(str).isin(["0","0.0"])]
        prices = prices[prices["has_market"].astype(str) == "1"]
        price_cols = ["card_year","n_listings","price_min","price_p25","price_median",
                      "price_mean","price_p75","price_max","top_listing_url","top_listing_title"]
        price_cols = [c for c in price_cols if c in prices.columns]
        prices_keep = prices[["player_id"] + price_cols].drop_duplicates("player_id")
        prices_keep.columns = ["player_id"] + [f"ebay_{c}" for c in price_cols]
        df = df.merge(prices_keep, on="player_id", how="left")
    except FileNotFoundError as e:
        print(f"  no prices file: {e}")

    # EV calculation
    EV = {"cup": 5, "utility": 25, "regular": 75, "breakout": 300, "no": 1}
    df["EV_dollars"] = (df["p_cup"]*EV["cup"] + df["p_utility"]*EV["utility"]
                       + df["p_regular"]*EV["regular"] + df["p_breakout"]*EV["breakout"]
                       + df["p_no_debut"]*EV["no"])
    if "ebay_price_p25" in df.columns:
        df["entry_price"] = pd.to_numeric(df["ebay_price_p25"], errors="coerce").fillna(
            pd.to_numeric(df["ebay_price_median"], errors="coerce"))
    else:
        df["entry_price"] = np.nan
    df["edge_dollars"] = df["EV_dollars"] - df["entry_price"]

    # Output
    out_cols = [
        "player_id","name","bucket","primary_position","current_org","cur_level_2026",
        "draft_year","draft_round","is_international","age_at_snap","years_in_pro",
        "overall_score","lasso_score","top100_score","pct_breakout","pct_top100",
        "p_MLB_DEBUT","p_TOP_100_PROSPECT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE",
        "p_debut","p_no_debut","p_cup","p_utility","p_regular","p_breakout",
        "p_cup_given_debut","p_utility_given_debut","p_regular_given_debut","p_breakout_given_debut",
        "y2026_pa","y2026_ip","y2026_avg","y2026_obp","y2026_slg","y2026_iso",
        "y2026_k_pct","y2026_bb_pct","y2026_home_runs","y2026_stolen_bases",
        "y2026_era","y2026_k9","y2026_bb9","y2026_whip","y2026_fip",
        "y2026_cur_pa","y2026_cur_ip","y2026_cur_avg","y2026_cur_obp",
        "y2026_cur_slg","y2026_cur_iso","y2026_cur_k_pct","y2026_cur_bb_pct",
        "y2026_cur_era","y2026_cur_k9","y2026_cur_bb9","y2026_cur_whip",
        "y2026_cur_fip","recently_promoted",
        "ebay_price_median","ebay_price_p25","ebay_n_listings","ebay_top_listing_url",
        "EV_dollars","entry_price","edge_dollars",
        "passes_filter",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    import os, time
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "history"), exist_ok=True)
    all_path = f"{args.out_prefix}_ALL_SCORED.csv"
    final_path = f"{args.out_prefix}_FINAL.csv"
    today = time.strftime("%Y-%m-%d")
    history_dir = os.path.join(out_dir, "history") if out_dir else "history"
    final_history = os.path.join(history_dir,
                                 f"buy_list_v1.17_FINAL_{today}.csv")
    all_history = os.path.join(history_dir,
                               f"buy_list_v1.17_ALL_SCORED_{today}.csv")

    def _safe_to_csv(df_out, path):
        """Write CSV, falling back to a timestamped sidecar if the target is
        locked (Excel/Numbers/IDE open on the file). Returns the path actually
        written. Failing the whole weekly run because someone left the buy
        list open in Excel is not acceptable for an autonomous deploy."""
        import time
        try:
            df_out.to_csv(path, index=False)
            return path
        except PermissionError:
            stamp = time.strftime("%Y-%m-%d_%H%M%S")
            base, ext = path.rsplit(".", 1)
            fallback = f"{base}.{stamp}.{ext}"
            df_out.to_csv(fallback, index=False)
            print(f"  WARN {path} is locked (open in Excel?); wrote {fallback}")
            return fallback

    # Full annotated scored set (everyone scored) + dated copy
    full = df[out_cols].sort_values("overall_score", ascending=False)
    print(f"saved {_safe_to_csv(full, all_path)} ({len(full):,} players)")
    _safe_to_csv(full, all_history)

    # Final filtered buy list (passes universe + yip threshold) + dated copy
    final = df[df["passes_filter"]].copy().sort_values("overall_score", ascending=False)
    final["buy_rank"] = np.arange(1, len(final)+1)
    out_cols_final = ["buy_rank"] + out_cols
    final = final[out_cols_final]
    print(f"saved {_safe_to_csv(final, final_path)} ({len(final):,} players passing filter)")
    _safe_to_csv(final, final_history)

    # Headline
    print(f"\n=== FINAL BUY LIST  (v1.17, universe + per-yip threshold) ===")
    print(f"  yip composition:")
    print(final["years_in_pro"].value_counts().sort_index().to_string())
    print(f"\n  bucket composition:")
    print(final["bucket"].value_counts().to_string())
    print(f"\n  top 20 by lasso_score:")
    show = ["buy_rank","name","years_in_pro","bucket","primary_position","cur_level_2026",
            "overall_score","lasso_score","p_MLB_DEBUT","pct_top100","pct_breakout","p_breakout",
            "y2026_pa","y2026_ip","y2026_avg","y2026_era","y2026_k9",
            "entry_price","EV_dollars","edge_dollars"]
    show = [c for c in show if c in final.columns]
    print(final[show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
