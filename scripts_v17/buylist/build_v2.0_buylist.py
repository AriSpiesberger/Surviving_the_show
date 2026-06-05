"""v2.0 production buy list builder.

Same pipeline as v1.18, but the per-event scoring engine is the joint
XGBoost booster (models/joint_xgb_v2.0_prod.pkl) instead of the per-event
lasso bundle. v2.0's logistic outputs are well-calibrated (honest val
ECE 0.001-0.016), so we filter by an absolute P(MLB_DEBUT) >= 0.60
threshold instead of per-yip cutoffs.

Universe filters (same as v1.18):
  - year_top_100 IS NULL  (drop ever-top-100)
  - eligible_MLB_DEBUT == 1  (drop pre-snap debutees)
  - cur_level_2026 != "MLB"  (drop currently-MLB-level)

Output: slim CSV with logistic probs + time-to-debut + eBay prices.
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_LONG = "results/scored/snap2026_v17_all_long.csv"
DEFAULT_XGB = "models/joint_xgb_v2.0_prod.pkl"
DEFAULT_TIMING = "models/time_to_debut_v1.18_prod.pkl"
DEFAULT_PRICES = "data/prices_bowman_chrome_auto_v13.csv"
DEFAULT_DB = "prospects_snapshot.db"
AGE_CENTER, YIP_CENTER = 22, 3


def _add_feats(df, db):
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(
        birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]],
                  on="player_id", how="left")
    df["age_at_snap"] = (df["snap_year"] - df["birth_year"]).fillna(22.0)
    df["age_at_snap_centered"] = df["age_at_snap"] - AGE_CENTER
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - YIP_CENTER
    for col in [c for c in df.columns if c.startswith("p_")]:
        if f"{col}_x_yip_centered" not in df.columns:
            df[f"{col}_x_yip_centered"] = df[col] * df["yip_centered"]
    return df


def _score_xgb(df, xgb_pkl):
    with open(xgb_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    feat = bundle["feature_names"]
    scaler = bundle["scaler"]
    booster = bundle["model"]
    best_iter = bundle.get("best_iteration")
    X = scaler.transform(df[feat].values.astype(np.float32))
    d = xgb.DMatrix(X, feature_names=list(feat))
    if best_iter is not None:
        P = booster.predict(d, iteration_range=(0, best_iter + 1))
    else:
        P = booster.predict(d)
    return {ev: P[:, k] for k, ev in enumerate(bundle["events"])}


def _score_timing(df, timing_pkl, p_debut):
    with open(timing_pkl, "rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    df = df.copy()
    if "p_debut_lasso" in feat:
        df["p_debut_lasso"] = p_debut
    for col in feat:
        if col not in df.columns and col.endswith("_x_yip_centered"):
            base = col[:-len("_x_yip_centered")]
            if base in df.columns:
                df[col] = df[base] * df["yip_centered"]
    return lasso.predict(sc.transform(df[feat].values))


def _join_prospect_meta(df, db):
    c = sqlite3.connect(db)
    meta = pd.read_sql(
        "SELECT player_id, primary_position, current_org FROM prospects", c)
    outcomes = pd.read_sql(
        "SELECT player_id, year_top_100 FROM career_outcomes", c)
    c.close()
    df = df.merge(meta, on="player_id", how="left")
    df = df.merge(outcomes, on="player_id", how="left")
    return df


def _join_current_level(df, db):
    ranks = {"RK": 0, "A-": 1, "A": 2, "A+": 3,
             "AA": 4, "AAA": 5, "MLB": 6}
    labels = {v: k for k, v in ranks.items()}
    c = sqlite3.connect(db)
    s = pd.read_sql(
        "SELECT player_id, season_year, level FROM season_stats", c)
    c.close()
    s = s.dropna(subset=["season_year"])
    s["season_year"] = s["season_year"].astype(int)
    s["rank"] = s["level"].astype(str).str.upper().map(ranks)
    s = s.dropna(subset=["rank"])
    s["rank"] = s["rank"].astype(int)
    hi = (s.groupby(["player_id", "season_year"])["rank"].max()
            .rename("cur_rank").reset_index())
    df = df.merge(hi, left_on=["player_id", "snap_year"],
                   right_on=["player_id", "season_year"], how="left")
    df["cur_level_2026"] = df["cur_rank"].map(labels).fillna("NONE")
    return df.drop(columns=["season_year", "cur_rank"], errors="ignore")


def _join_prices(df, prices_csv):
    """Join eBay prices into df. Supports two source shapes:

      A) Already-prefixed buy-list output ({ebay_price_median,
         ebay_price_p25, ebay_n_listings, ebay_top_listing_url}) — the
         legacy default (results/buy_lists/buy_list_v1.17_FINAL.csv).
         Covers ~300 players (the v1.17-filtered set).

      B) Raw eBay price file (data/prices_bowman_chrome_auto_v13.csv) with
         {price_median, price_p25, n_listings, top_listing_url, denominator,
         has_market}. Covers ~10k players (the full crawl). We filter to
         base/raw rows (denominator == 0, has_market == 1) and apply the
         ebay_ prefix so the downstream output schema is unchanged.

    Auto-detects by column presence so both paths work without a flag.
    """
    if not prices_csv:
        return df
    try:
        p = pd.read_csv(prices_csv)
    except FileNotFoundError:
        print(f"  (prices file not found: {prices_csv} — skipping)")
        return df

    # Path A: already-prefixed
    if "ebay_price_median" in p.columns:
        keep = [c for c in ["player_id", "ebay_price_median",
                              "ebay_price_p25", "ebay_n_listings",
                              "ebay_top_listing_url"]
                if c in p.columns]
        if "player_id" not in keep:
            return df
        print(f"  joined prices from {prices_csv}: "
              f"{p['player_id'].nunique():,} players with prices")
        return df.merge(p[keep], on="player_id", how="left")

    # Path B: raw price file (broad eBay crawl)
    if "price_median" in p.columns:
        if "denominator" in p.columns:
            p = p[p["denominator"].astype(str).isin(["0", "0.0"])]
        if "has_market" in p.columns:
            p = p[p["has_market"].astype(str) == "1"]
        rename = {"price_median": "ebay_price_median",
                  "price_p25":    "ebay_price_p25",
                  "n_listings":   "ebay_n_listings",
                  "top_listing_url": "ebay_top_listing_url"}
        cols = ["player_id"] + [c for c in rename if c in p.columns]
        p = p[cols].drop_duplicates("player_id").rename(columns=rename)
        print(f"  joined prices from {prices_csv}: "
              f"{p['player_id'].nunique():,} players with prices "
              f"(filtered to base+raw)")
        return df.merge(p, on="player_id", how="left")

    print(f"  (prices file {prices_csv} has neither ebay_* nor price_* cols)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", "--snap-long", dest="long",
                    default=DEFAULT_LONG)
    ap.add_argument("--xgb", default=DEFAULT_XGB)
    ap.add_argument("--timing", default=DEFAULT_TIMING)
    ap.add_argument("--prices", default=DEFAULT_PRICES)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="Flat P(MLB_DEBUT) threshold for FINAL list")
    ap.add_argument("--sort-by", default="p_MLB_DEBUT")
    ap.add_argument("--events", nargs="+",
                    default=["TOP_100_PROSPECT", "MLB_DEBUT",
                              "ESTABLISHED_MLB", "STAR_PLUS_ELITE"])
    ap.add_argument("--out-all",
                    default="results/buy_lists/buy_list_v2.0_ALL_SCORED.csv")
    ap.add_argument("--out-final",
                    default="results/buy_lists/buy_list_v2.0_FINAL.csv")
    args = ap.parse_args()

    print(f"Loading {args.long}")
    df = pd.read_csv(args.long)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} players")

    df = _add_feats(df, args.db)
    df = _join_prospect_meta(df, args.db)
    df = _join_current_level(df, args.db)
    print(f"Scoring with {args.xgb}")
    scores = _score_xgb(df, args.xgb)
    for ev, arr in scores.items():
        df[f"p_lasso_{ev}"] = arr   # holds v2.0 calibrated probability
    p_debut = scores["MLB_DEBUT"]
    print(f"Scoring time-to-debut with {args.timing}")
    df["time_to_debut"] = _score_timing(df, args.timing, p_debut)

    if args.prices:
        print(f"Joining eBay prices from {args.prices}")
        df = _join_prices(df, args.prices)

    # Universe filters
    n0 = len(df)
    df = df[df["year_top_100"].isna()].copy()
    print(f"Drop ever-top-100: {n0:,} -> {len(df):,}  "
          f"({n0-len(df):,} removed)")
    n_r1 = len(df)
    df = df[df["bucket"] != "R1"].copy()
    print(f"Drop R1 picks (R1 picks are 'top-100' by another name): "
          f"{n_r1:,} -> {len(df):,}  ({n_r1-len(df):,} removed)")
    if "eligible_MLB_DEBUT" in df.columns:
        n1 = len(df)
        df = df[df["eligible_MLB_DEBUT"] == 1].copy()
        print(f"Drop pre-snap debutees: {n1:,} -> {len(df):,}  "
              f"({n1-len(df):,} removed)")
    n2 = len(df)
    df = df[df["cur_level_2026"] != "MLB"].copy()
    print(f"Drop currently-MLB: {n2:,} -> {len(df):,}  "
          f"({n2-len(df):,} removed)")

    # Apply flat threshold on P(MLB_DEBUT)
    df["passes_filter"] = df["p_lasso_MLB_DEBUT"] >= args.threshold
    print(f"P(MLB_DEBUT) >= {args.threshold}: "
          f"{int(df['passes_filter'].sum()):,} pass")

    keep = ["player_id", "name", "bucket", "draft_year", "draft_round",
            "primary_position", "current_org", "cur_level_2026",
            "age_at_snap", "years_in_pro"]
    keep += [f"p_lasso_{ev}" for ev in args.events]
    keep += ["time_to_debut", "passes_filter"]
    for c in ("ebay_price_median", "ebay_price_p25", "ebay_n_listings",
              "ebay_top_listing_url"):
        if c in df.columns:
            keep.append(c)
    # Rename p_lasso_* -> p_<event> for slim output
    rename = {f"p_lasso_{ev}": f"p_{ev}" for ev in args.events}
    out = df[keep].rename(columns=rename).copy()
    sort_col = rename.get(args.sort_by, args.sort_by)
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)

    out.to_csv(args.out_all, index=False)
    print(f"\nWrote {args.out_all}  rows={len(out):,}")

    final = out[out["passes_filter"]].copy()
    final.to_csv(args.out_final, index=False)
    print(f"Wrote {args.out_final}  rows={len(final):,}")
    if len(final):
        print(f"\nFINAL by level:")
        for lv, n in final.groupby("cur_level_2026").size().items():
            print(f"  {lv:<10} n={int(n):,}")
        print(f"  with eBay price: "
              f"{final['ebay_price_median'].notna().sum() if 'ebay_price_median' in final.columns else 0:,}")


if __name__ == "__main__":
    main()
