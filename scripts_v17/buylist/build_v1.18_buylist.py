"""v1.18 production buy list builder.

Inputs:
  - snap2026_v17_all_long.csv  : 2026 prospects scored by prod hazards
  - lasso_logits_v1.18_prod.pkl: per-event L1-logistic bundle
  - time_to_debut_v1.18_prod.pkl: time-to-debut Lasso regression
  - buy_list_v1.17_FINAL.csv   : (optional) source of eBay prices, joined
                                  by player_id

Output: slim CSV with ONLY:
  - identity:   player_id, name, bucket, draft_year, draft_round,
                primary_position, current_org, cur_level_2026, age,
                years_in_pro
  - logistic:   p_TOP_100_PROSPECT, p_MLB_DEBUT, p_ESTABLISHED_MLB,
                p_STAR_PLUS_ELITE     (calibrated probabilities)
  - timing:     time_to_debut         (predicted years)
  - card value: ebay_price_median, ebay_price_p25, ebay_n_listings,
                ebay_top_listing_url

Sort by p_MLB_DEBUT descending.
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd

DEFAULT_LONG = "results/scored/snap2026_v17_all_long.csv"
DEFAULT_BUNDLE = "models/lasso_logits_v1.18_prod.pkl"
DEFAULT_TIMING = "models/time_to_debut_v1.18_prod.pkl"
DEFAULT_PRICES = "results/buy_lists/buy_list_v1.17_FINAL.csv"
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


def _score_bundle(df, bundle_pkl, events):
    with open(bundle_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    out = {}
    for ev in events:
        art = bundle["per_event"][ev]
        sc, lasso, feat = art["scaler"], art["lasso"], art["feature_names"]
        out[ev] = lasso.predict_proba(sc.transform(df[feat].values))[:, 1]
    return out


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
    """Pull primary_position, current_org from prospects table, and
    year_top_100 from career_outcomes (used to filter out players who have
    already appeared on the top-100 — they're outside our buy universe)."""
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
    """Highest level played in snap_year (from season_stats)."""
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
    df["cur_level_2026"] = (df["cur_rank"].map(labels).fillna("NONE"))
    return df.drop(columns=["season_year", "cur_rank"], errors="ignore")


def _join_prices(df, prices_csv):
    if prices_csv is None:
        return df
    try:
        p = pd.read_csv(prices_csv)
    except FileNotFoundError:
        print(f"  (prices file not found: {prices_csv} — skipping)")
        return df
    keep = [c for c in ["player_id", "ebay_price_median", "ebay_price_p25",
                          "ebay_n_listings", "ebay_top_listing_url"]
            if c in p.columns]
    if "player_id" not in keep:
        return df
    return df.merge(p[keep], on="player_id", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", "--snap-long", dest="long",
                    default=DEFAULT_LONG)
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--timing", default=DEFAULT_TIMING)
    ap.add_argument("--prices", default=DEFAULT_PRICES,
                    help="Existing buy list to source eBay prices from "
                         "(joined by player_id). Pass '' to skip.")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--events", nargs="+",
                    default=["TOP_100_PROSPECT", "MLB_DEBUT",
                              "ESTABLISHED_MLB", "STAR_PLUS_ELITE"])
    ap.add_argument("--sort-by", default="p_MLB_DEBUT")
    ap.add_argument("--thresholds",
                    default="results/val_v18h_2026-06-02/"
                            "MLB_DEBUT_thresholds_at_p60.csv",
                    help="Per-yip threshold CSV (yip, threshold). Applied "
                         "to p_MLB_DEBUT to derive `passes_filter`. Empty "
                         "= no filter.")
    ap.add_argument("--out-all",
                    default="results/buy_lists/buy_list_v1.18_ALL_SCORED.csv")
    ap.add_argument("--out-final",
                    default="results/buy_lists/buy_list_v1.18_FINAL.csv")
    args = ap.parse_args()

    print(f"Loading {args.long}")
    df = pd.read_csv(args.long)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} players")

    df = _add_feats(df, args.db)
    df = _join_prospect_meta(df, args.db)
    df = _join_current_level(df, args.db)
    print(f"Scoring with {args.bundle}")
    scores = _score_bundle(df, args.bundle, args.events)
    for ev, arr in scores.items():
        df[f"p_{ev}"] = arr
    print(f"Scoring time-to-debut with {args.timing}")
    df["time_to_debut"] = _score_timing(df, args.timing,
                                          scores["MLB_DEBUT"])

    if args.prices:
        print(f"Joining eBay prices from {args.prices}")
        df = _join_prices(df, args.prices)

    keep = ["player_id", "name", "bucket", "draft_year", "draft_round",
            "primary_position", "current_org", "cur_level_2026",
            "age_at_snap", "years_in_pro"]
    keep += [f"p_{ev}" for ev in args.events]
    keep += ["time_to_debut"]

    # Universe filter (1/2): drop players who have already appeared on the
    # top-100 list. They're outside the buy universe — the model wasn't
    # built to rank them and the market has already priced them.
    n_before = len(df)
    df = df[df["year_top_100"].isna()].copy()
    print(f"Universe filter (drop ever-top-100): "
          f"{n_before:,} -> {len(df):,}  "
          f"({n_before - len(df):,} removed)")

    # Universe filter (2/3): drop players who already debuted before the
    # snap year. `eligible_MLB_DEBUT == 1` means the player had NOT debuted
    # by the start of snap_year (carried from the hazard scoring).
    if "eligible_MLB_DEBUT" in df.columns:
        n_before = len(df)
        df = df[df["eligible_MLB_DEBUT"] == 1].copy()
        print(f"Universe filter (eligible_MLB_DEBUT==1, drop pre-snap "
              f"debutees): {n_before:,} -> {len(df):,}  "
              f"({n_before - len(df):,} removed)")

    # Universe filter (3/3): drop players currently at MLB level. Catches
    # current-season call-ups whose mlb_debut_year hasn't propagated to the
    # snapshot DB yet (so eligible_MLB_DEBUT is still 1 even though they're
    # already in The Show).
    n_before = len(df)
    df = df[df["cur_level_2026"] != "MLB"].copy()
    print(f"Universe filter (drop currently-MLB): "
          f"{n_before:,} -> {len(df):,}  "
          f"({n_before - len(df):,} removed)")

    # Apply per-yip threshold filter if provided.
    if args.thresholds:
        try:
            thr_df = pd.read_csv(args.thresholds)
            thr_map = dict(zip(thr_df["yip"].astype(int),
                                thr_df["threshold"]))
            print(f"Loaded thresholds: {thr_map}")

            def _passes(row):
                t = thr_map.get(int(row["years_in_pro"]))
                if t is None or pd.isna(t):
                    return False
                return row["p_MLB_DEBUT"] >= t
            df["threshold"] = df["years_in_pro"].astype(int).map(thr_map)
            df["passes_filter"] = df.apply(_passes, axis=1)
            keep += ["threshold", "passes_filter"]
        except FileNotFoundError:
            print(f"  (thresholds file missing: {args.thresholds})")

    for c in ("ebay_price_median", "ebay_price_p25", "ebay_n_listings",
              "ebay_top_listing_url"):
        if c in df.columns:
            keep.append(c)

    all_df = df[keep].copy()
    if args.sort_by in all_df.columns:
        all_df = all_df.sort_values(args.sort_by, ascending=False)
    all_df.to_csv(args.out_all, index=False)
    print(f"\nWrote {args.out_all}")
    print(f"  rows={len(all_df):,}  cols={len(all_df.columns)}")

    if "passes_filter" in all_df.columns:
        final = all_df[all_df["passes_filter"]].copy()
        final.to_csv(args.out_final, index=False)
        print(f"\nWrote {args.out_final}")
        print(f"  rows={len(final):,}  cols={len(final.columns)}")
        # Per-yip count of FINAL list
        print(f"\n  final count by yip:")
        for yip, n in final.groupby("years_in_pro").size().items():
            t = final.loc[final.years_in_pro == yip,
                          "threshold"].iloc[0]
            print(f"    yip={int(yip)}  n={int(n):,}  thr={t:.3f}")
        wp = final["ebay_price_median"].notna().sum() \
            if "ebay_price_median" in final.columns else 0
        print(f"  with eBay price: {wp:,}")


if __name__ == "__main__":
    main()
