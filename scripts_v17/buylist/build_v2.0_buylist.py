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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prospects.classifier.joint_cond import predict_trajectory, prep_base

DEFAULT_LONG = "results/scored/snap2026_v17_all_long.csv"
DEFAULT_XGB = "models/joint_xgb_v2.0b_prod.pkl"
DEFAULT_TIMING = "models/time_to_debut_v2.0b_prod.pkl"  # retrained on v2.0b haz
DEFAULT_PRICES = "data/prices_bowman_chrome_auto_v13.csv"
DEFAULT_DB = "prospects_snapshot.db"
AGE_CENTER, YIP_CENTER = 22, 3


def _add_feats(df, db):
    # Horizon-independent feature prep shared with the trainer/eval. Adds
    # age_at_snap_centered, years_in_pro, yip_centered, hazard x yip interactions
    # and the point-in-time scouting summary. We additionally keep the
    # non-centered age_at_snap for the slim output schema.
    df = prep_base(df, db)
    df["age_at_snap"] = df["age_at_snap_centered"] + AGE_CENTER
    return df


def _score_xgb(df, xgb_pkl):
    """Score the conditional model and return the publish-horizon (h=6) cumulative
    probability per event, as {event: ndarray}. The full per-year trajectory
    (xp_<event>_h{1..H}) is computed internally; the buy list is a single-horizon
    artifact so we surface the h=PUBLISH_H slice the bundle was trained to publish."""
    with open(xgb_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    scored = predict_trajectory(bundle, df)
    return {ev: scored[f"xp_{ev}"].to_numpy() for ev in bundle["events"]}


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
    ap.add_argument("--xgb-ceiling", default=None,
                    help="Legacy: single model for both est+star.")
    ap.add_argument("--xgb-est", default=None,
                    help="v2.1: model for ESTABLISHED_MLB (e.g. est@9).")
    ap.add_argument("--xgb-star", default=None,
                    help="v2.1: model for STAR_PLUS_ELITE (e.g. star@12).")
    ap.add_argument("--timing", default=DEFAULT_TIMING)
    ap.add_argument("--prices", default=DEFAULT_PRICES)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="Flat P(MLB_DEBUT within debut-horizon yrs) threshold "
                         "for the FINAL list")
    ap.add_argument("--debut-horizon", type=int, default=3,
                    help="Buy thesis = P(MLB_DEBUT within this many years). The "
                         "FINAL filter, sort, and the output p_MLB_DEBUT column "
                         "all use this horizon's cumulative slice (default 3y).")
    ap.add_argument("--max-yip", type=int, default=None,
                    help="Drop players with > this many years of service "
                         "(snap_offset). e.g. 4 = only <=4 yrs in pro.")
    ap.add_argument("--yip-thresholds", default=None,
                    help="JSON file {yip: P(debut) threshold} for per-yip "
                         "precision-calibrated cutoffs (overrides --threshold).")
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
    print(f"Scoring conditional trajectory with {args.xgb}")
    with open(args.xgb, "rb") as fh:
        bundle = pickle.load(fh)
    scored = predict_trajectory(bundle, df)  # adds xp_<ev>_h{1..H} + xp_<ev> (h6)
    dh = args.debut_horizon
    debut_col = f"xp_MLB_DEBUT_h{dh}"
    if debut_col not in scored.columns:
        raise SystemExit(f"FATAL: model h_max < debut-horizon={dh} "
                         f"(missing {debut_col})")
    # Ceiling/context events reported at the publish horizon (h=6).
    for ev in args.events:
        df[f"p_lasso_{ev}"] = scored[f"xp_{ev}"].to_numpy()
    # Buy thesis: P(MLB debut within `dh` years) drives the filter, sort and the
    # output p_MLB_DEBUT column. Keep the h=6 debut prob as a context column.
    df["p_lasso_MLB_DEBUT"] = scored[debut_col].to_numpy()
    df["p_lasso_MLB_DEBUT_6y"] = scored["xp_MLB_DEBUT_h6"].to_numpy()
    print(f"  buy thesis = P(MLB_DEBUT <= {dh}y) [{debut_col}]; "
          f"ceiling events at h=6")
    # Time-to-debut: feed the publish-horizon (h6) debut prob, the distribution
    # the timing model was trained against.
    print(f"Scoring time-to-debut with {args.timing}")
    df["time_to_debut"] = _score_timing(df, args.timing,
                                        scored["xp_MLB_DEBUT"].to_numpy())

    if args.prices:
        print(f"Joining eBay prices from {args.prices}")
        df = _join_prices(df, args.prices)

    # Universe filters
    n0 = len(df)
    # Point-in-time: drop only players who were already top-100 AS OF the snap
    # year. A player who first makes top-100 *after* the snap was not yet known,
    # so he legitimately belongs in the buy universe at that snap (the
    # buy-before-pop case). At snap=present this reduces to "drop ever-top-100".
    was_top100 = (df["year_top_100"].notna()
                  & (df["year_top_100"] <= df["snap_year"]))
    df = df[~was_top100].copy()
    print(f"Drop top-100-as-of-snap: {n0:,} -> {len(df):,}  "
          f"({n0-len(df):,} removed)")
    # NOTE: R1 picks are kept — they belong in the buy universe unless they're
    # already a known (point-in-time) top-100 prospect, handled by the filter
    # above. (Previously dropped wholesale as "top-100 by another name".)
    if "eligible_MLB_DEBUT" in df.columns:
        n1 = len(df)
        df = df[df["eligible_MLB_DEBUT"] == 1].copy()
        print(f"Drop pre-snap debutees: {n1:,} -> {len(df):,}  "
              f"({n1-len(df):,} removed)")
    n2 = len(df)
    df = df[df["cur_level_2026"] != "MLB"].copy()
    print(f"Drop currently-MLB: {n2:,} -> {len(df):,}  "
          f"({n2-len(df):,} removed)")
    if args.max_yip is not None and "snap_offset" in df.columns:
        n3 = len(df)
        df = df[df["snap_offset"] <= args.max_yip].copy()
        print(f"Drop >{args.max_yip} yrs service (snap_offset): "
              f"{n3:,} -> {len(df):,}  ({n3-len(df):,} removed)")

    # Threshold: per-yip map (calibrated precision) or a flat cutoff
    if args.yip_thresholds:
        import json as _json
        ymap = {int(k): float(v)
                for k, v in _json.load(open(args.yip_thresholds)).items()}
        thr_row = df["snap_offset"].map(ymap)
        # A yip with no calibrated threshold can't reach the target precision
        # at any cutoff -> nobody at that yip passes.
        df["passes_filter"] = thr_row.notna() & (df["p_lasso_MLB_DEBUT"] >= thr_row)
        print(f"Per-yip thresholds {ymap} on P(debut<={dh}y): "
              f"{int(df['passes_filter'].sum()):,} pass")
    else:
        df["passes_filter"] = df["p_lasso_MLB_DEBUT"] >= args.threshold
        print(f"P(MLB_DEBUT <= {dh}y) >= {args.threshold}: "
              f"{int(df['passes_filter'].sum()):,} pass")

    keep = ["player_id", "name", "bucket", "draft_year", "draft_round",
            "primary_position", "current_org", "cur_level_2026",
            "age_at_snap", "years_in_pro"]
    keep += [f"p_lasso_{ev}" for ev in args.events]
    keep += ["p_lasso_MLB_DEBUT_6y"]  # context: P(debut<=6y) alongside the 3y thesis
    keep += ["time_to_debut", "passes_filter"]
    for c in ("ebay_price_median", "ebay_price_p25", "ebay_n_listings",
              "ebay_top_listing_url"):
        if c in df.columns:
            keep.append(c)
    # Rename p_lasso_* -> p_<event> for slim output. p_MLB_DEBUT holds the
    # debut-horizon (3y) thesis; p_MLB_DEBUT_6y is the longer-window context.
    rename = {f"p_lasso_{ev}": f"p_{ev}" for ev in args.events}
    rename["p_lasso_MLB_DEBUT_6y"] = "p_MLB_DEBUT_6y"
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
