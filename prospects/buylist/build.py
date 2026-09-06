"""v2.0 production buy list builder.

Same pipeline as v1.18, but the per-event scoring engine is the joint
XGBoost booster (models/joint_xgb_v2.0_prod.pkl) instead of the per-event
lasso bundle. v2.0's logistic outputs are well-calibrated (honest val
ECE 0.001-0.016). The final list is filtered by the established per-yip
precision cutoff (BUYLIST_PRECISION, default 0.60): for each years-in-pro,
keep only picks above the score where empirical precision clears the target.
Thresholds are computed on the current model's val slice each run (never
stale). --precision 0 falls back to a flat probability --threshold.

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

from prospects import config
from prospects.model.joint import AGE_CENTER, YIP_CENTER, predict_trajectory, prep_base

_RUN = config.run()  # runs/current unless RUN_TAG overrides
DEFAULT_LONG = str(_RUN.snap_long(2026))
DEFAULT_XGB = str(_RUN.joint_xgb)
DEFAULT_TIMING = str(_RUN.timing)
DEFAULT_PRICES = str(config.BUYLIST_PRICES)
DEFAULT_DB = str(config.model_db())

# Established buy-list knob: target per-yip debut precision. The buy list keeps
# a pick only where empirical precision at its years-in-pro clears this. Set to
# 0 (or --precision 0) to fall back to the flat --threshold instead.
BUYLIST_PRECISION = 0.60


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
    ap.add_argument("--calibrators",
                    default=str(_RUN.calibrators)
                    if _RUN.calibrators.exists() else None,
                    help="prob_calibrators_v2.0b.pkl (from fit_prob_calibrators). "
                         "When set, the output p_<event> columns are isotonic-"
                         "calibrated to true probabilities (raw kept as "
                         "p_<event>_raw). The debut filter then operates on the "
                         "calibrated debut prob, so --threshold reads as a real "
                         "probability.")
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD,
                    help="Flat P(MLB_DEBUT within debut-horizon yrs) fallback "
                         "threshold, used only when --precision 0.")
    ap.add_argument("--precision", type=float, default=BUYLIST_PRECISION,
                    help=f"Target per-yip debut PRECISION for the final list "
                         f"(default {BUYLIST_PRECISION}). The established cutoff: "
                         f"computes per-yip thresholds on the current model so "
                         f"each yip's picks clear this precision. --precision 0 "
                         f"uses the flat --threshold instead.")
    ap.add_argument("--debut-horizon", type=int,
                    default=config.DEFAULT_DEBUT_HORIZON,
                    help="Buy thesis = P(MLB_DEBUT within this many years). The "
                         "FINAL filter, sort, and the output p_MLB_DEBUT column "
                         "all use this horizon's cumulative slice (default 3y).")
    ap.add_argument("--debut-horizons", default="2,3,4,6",
                    help="Comma list of debut horizons to emit as "
                         "p_MLB_DEBUT_<h>y columns (the trajectory, not one "
                         "number). The --debut-horizon thesis and 6y are always "
                         "included. Default 2,3,4,6.")
    ap.add_argument("--max-yip", type=int, default=3,
                    help="Drop players with > this many years of service "
                         "(snap_offset). Default 3 — we only buy through yip 3. "
                         "Pass -1 to disable the cap.")
    ap.add_argument("--include-ifa", action="store_true",
                    help="Keep IFA (ifa_*) prospects in the buy list. Off by "
                         "default: IFAs aren't in the draft-keyed training "
                         "panel, so their scores are untrustworthy extrapolation.")
    ap.add_argument("--yip-thresholds", default=None,
                    help="JSON file {yip: P(debut) threshold} for per-yip "
                         "precision-calibrated cutoffs. This is the proper "
                         "production filter — each yip cohort held to ~the "
                         "target precision rather than a flat cutoff. If "
                         "omitted, thresholds are computed fresh from "
                         "--precision on the current model.")
    ap.add_argument("--sort-by", default="p_MLB_DEBUT")
    ap.add_argument("--events", nargs="+",
                    default=["TOP_100_PROSPECT", "MLB_DEBUT",
                              "ESTABLISHED_MLB", "STAR_PLUS_ELITE"])
    ap.add_argument("--out-all", default=str(_RUN.buy_list_all))
    ap.add_argument("--out-final", default=str(_RUN.buy_list_final))
    args = ap.parse_args()
    Path(args.out_all).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.long}")
    df = pd.read_csv(args.long)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} players")

    df = _add_feats(df, args.db)
    df = _join_prospect_meta(df, args.db)
    df = _join_current_level(df, args.db)
    print(f"Scoring conditional trajectory with {args.xgb}")
    from prospects.model.joint2 import (
        apply_calibrators_frame, cdf_timing, is_bag_bundle, load_calibrators,
        make_cal_fn, score_trajectory,
    )
    # v2.2 bag bundles (raw features attached from the DB) and legacy v2.1c
    # scaler bundles both come back with the same xp_<ev>_h{1..H} schema.
    scored, bundle = score_trajectory(args.xgb, df, args.db)
    dh = args.debut_horizon
    debut_col = f"xp_MLB_DEBUT_h{dh}"
    if debut_col not in scored.columns:
        raise SystemExit(f"FATAL: model h_max < debut-horizon={dh} "
                         f"(missing {debut_col})")
    # Optional isotonic calibration: map raw conditional-XGB scores -> true
    # probabilities using per-(event, horizon) calibrators fit on the held-out
    # val. Keeps the raw score as p_lasso_<col>_raw.
    cal = None
    _cal_fn = None
    if args.calibrators:
        cal_bundle = load_calibrators(args.calibrators)
        cal = cal_bundle["calibrators"]
        _cal_fn = make_cal_fn(cal_bundle, scored)
        print(f"  calibrating probabilities with {Path(args.calibrators).name} "
              f"({len(cal)} calibrators, "
              f"{cal_bundle.get('kind', 'per-(event,h)')})")

    def _cal(values, event, h):
        """Calibrate a raw score vector for (event, h); identity if no cal.
        Dispatches across legacy per-(event,h) and v2.2 h/yip formats."""
        if _cal_fn is None:
            return values
        return _cal_fn(values, event, h)

    # Ceiling/context events reported at the publish horizon (h=6).
    for ev in args.events:
        if ev == "MLB_DEBUT":
            continue  # debut handled below across multiple horizons
        raw = scored[f"xp_{ev}"].to_numpy()
        if cal is not None:
            df[f"p_lasso_{ev}_raw"] = raw
        df[f"p_lasso_{ev}"] = _cal(raw, ev, 6)
    # DEBUT IS A TRAJECTORY, not one number: emit P(MLB debut <= h) at every
    # requested horizon (default 2y/3y/4y/6y), each with its own calibrator.
    # dh (default 3y) is the thesis that drives the filter/sort.
    debut_hs = sorted({int(x) for x in str(args.debut_horizons).split(",")}
                      | {dh, 6})
    for h in debut_hs:
        col = f"xp_MLB_DEBUT_h{h}"
        if col not in scored.columns:
            continue
        raw = scored[col].to_numpy()
        if cal is not None:
            df[f"p_lasso_MLB_DEBUT_{h}y_raw"] = raw
        df[f"p_lasso_MLB_DEBUT_{h}y"] = _cal(raw, "MLB_DEBUT", h)
    # Thesis column (filter/sort) = the debut-horizon slice.
    df["p_lasso_MLB_DEBUT"] = df[f"p_lasso_MLB_DEBUT_{dh}y"]
    if cal is not None:
        df["p_lasso_MLB_DEBUT_raw"] = df[f"p_lasso_MLB_DEBUT_{dh}y_raw"]
    print(f"  buy thesis = P(MLB_DEBUT <= {dh}y) [{debut_col}]; "
          f"debut horizons emitted: {', '.join(f'{h}y' for h in debut_hs)}; "
          f"ceiling events at h=6"
          + ("  [CALIBRATED]" if cal is not None else ""))
    # Time-to-debut. v2.2 bag bundles: read it off the calibrated debut
    # trajectory (CDF median + q25-q75 window) — the same object the
    # probabilities come from, so "when" and "whether" can never disagree.
    # Legacy bundles keep the Lasso timing model.
    if is_bag_bundle(bundle):
        print(f"Timing from the calibrated debut CDF (median + q25-q75)")
        traj = (apply_calibrators_frame(scored, cal_bundle)
                if args.calibrators else scored)
        tim = cdf_timing(traj, "MLB_DEBUT")
        df["time_to_debut"] = tim["t_med"].to_numpy()
        df["debut_eta_lo"] = tim["t_q25"].to_numpy()
        df["debut_eta_hi"] = tim["t_q75"].to_numpy()
    else:
        print(f"Scoring time-to-debut with {args.timing}")
        df["time_to_debut"] = _score_timing(df, args.timing,
                                            scored["xp_MLB_DEBUT"].to_numpy())

    if args.prices:
        print(f"Joining eBay prices from {args.prices}")
        df = _join_prices(df, args.prices)

    # Universe filters
    # Drop IFAs by default: the training panel is draft-keyed, so IFAs were
    # never in training — scoring them is extrapolation and their probabilities
    # aren't trustworthy (they dominated an early buy list spuriously). Opt back
    # in with --include-ifa once IFAs are trained on.
    if not args.include_ifa:
        n_ifa = len(df)
        df = df[~df["player_id"].astype(str).str.startswith("ifa_")].copy()
        print(f"Drop IFAs (not in training): {n_ifa:,} -> {len(df):,}  "
              f"({n_ifa-len(df):,} removed)")
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
    if args.max_yip is not None and args.max_yip >= 0 \
            and "snap_offset" in df.columns:
        n3 = len(df)
        df = df[df["snap_offset"] <= args.max_yip].copy()
        print(f"Drop >{args.max_yip} yrs service (snap_offset): "
              f"{n3:,} -> {len(df):,}  ({n3-len(df):,} removed)")

    # Threshold. The ESTABLISHED default is the per-yip precision cutoff: for
    # each years-in-pro, buy only above the score where empirical precision
    # >= BUYLIST_PRECISION. Because base rates fall with yip, a flat probability
    # cutoff is wrong (it only ever clears advanced arms); precision-per-yip is
    # the right knob. We compute it on the CURRENT model's val slice (calibrated
    # space, matching p_lasso_MLB_DEBUT) so it's never stale.
    import json as _json
    ymap = None
    src = None
    if args.yip_thresholds:                       # explicit pre-generated file
        ymap = {int(k): float(v)
                for k, v in _json.load(open(args.yip_thresholds)).items()}
        src = f"file {Path(args.yip_thresholds).name}"
    elif args.precision and args.precision > 0:   # default: compute on current model
        try:
            from prospects.model.thresholds import compute_yip_thresholds
            thr = compute_yip_thresholds(
                str(_RUN.oof_val_long), args.xgb, horizon=dh,
                target=args.precision, calibrators=args.calibrators,
                db=args.db, verbose=False)
            ymap = {int(k): float(v) for k, v in thr.items()}
            out = _RUN.yip_thresholds(int(round(args.precision * 100)))
            out.parent.mkdir(parents=True, exist_ok=True)
            _json.dump(thr, open(out, "w"), indent=2)
            src = (f"P{int(round(args.precision*100))} per-yip, computed on "
                   f"current model -> {out.name}")
        except Exception as e:
            print(f"  [threshold] precision cutoff unavailable ({type(e).__name__}: "
                  f"{e}); falling back to flat --threshold {args.threshold}")

    if ymap is not None:
        thr_row = df["snap_offset"].map(ymap)
        # A yip with no calibrated threshold can't reach the target precision
        # at any cutoff -> nobody at that yip passes.
        df["passes_filter"] = thr_row.notna() & (df["p_lasso_MLB_DEBUT"] >= thr_row)
        print(f"Per-yip precision cutoff [{src}] on P(debut<={dh}y): "
              f"{int(df['passes_filter'].sum()):,} pass  {ymap}")
    else:
        df["passes_filter"] = df["p_lasso_MLB_DEBUT"] >= args.threshold
        print(f"Flat P(MLB_DEBUT <= {dh}y) >= {args.threshold}: "
              f"{int(df['passes_filter'].sum()):,} pass")

    debut_hs = sorted({int(x) for x in str(args.debut_horizons).split(",")}
                      | {dh, 6})
    keep = ["player_id", "name", "bucket", "draft_year", "draft_round",
            "primary_position", "current_org", "cur_level_2026",
            "age_at_snap", "years_in_pro"]
    # ceiling events + the debut thesis col + the full debut trajectory
    keep += [f"p_lasso_{ev}" for ev in args.events if ev != "MLB_DEBUT"]
    keep += ["p_lasso_MLB_DEBUT"]
    keep += [f"p_lasso_MLB_DEBUT_{h}y" for h in debut_hs]
    keep += ["time_to_debut", "passes_filter"]
    keep += [c for c in ("debut_eta_lo", "debut_eta_hi") if c in df.columns]
    for c in ("ebay_price_median", "ebay_price_p25", "ebay_n_listings",
              "ebay_top_listing_url"):
        if c in df.columns:
            keep.append(c)
    # Rename p_lasso_* -> p_<event>. p_MLB_DEBUT = the thesis (dh) slice;
    # p_MLB_DEBUT_<h>y = the calibrated debut probability at each horizon.
    rename = {f"p_lasso_{ev}": f"p_{ev}"
              for ev in args.events if ev != "MLB_DEBUT"}
    rename["p_lasso_MLB_DEBUT"] = "p_MLB_DEBUT"
    for h in debut_hs:
        rename[f"p_lasso_MLB_DEBUT_{h}y"] = f"p_MLB_DEBUT_{h}y"
    # When calibrated, carry the raw scores alongside (p_<event>_raw).
    if cal is not None:
        for ev in args.events:
            if ev == "MLB_DEBUT":
                continue
            rc = f"p_lasso_{ev}_raw"
            if rc in df.columns:
                keep.append(rc); rename[rc] = f"p_{ev}_raw"
        for h in debut_hs:
            rc = f"p_lasso_MLB_DEBUT_{h}y_raw"
            if rc in df.columns:
                keep.append(rc); rename[rc] = f"p_MLB_DEBUT_{h}y_raw"
    out = df[keep].rename(columns=rename).copy()
    sort_col = rename.get(args.sort_by, args.sort_by)
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)

    out.to_csv(args.out_all, index=False)
    print(f"\nWrote {args.out_all}  rows={len(out):,}")

    final = out[out["passes_filter"]].copy()
    try:
        final.to_csv(args.out_final, index=False)
        print(f"Wrote {args.out_final}  rows={len(final):,}")
    except PermissionError:
        # The final CSV is routinely open in Excel; a locked file must not
        # fail the (weekly) build. Park the fresh list next to it instead.
        alt = str(Path(args.out_final).with_suffix("")) + "_new.csv"
        final.to_csv(alt, index=False)
        print(f"WARN: {args.out_final} is locked (open in Excel?) — wrote "
              f"{alt} instead. Close the file and rename it over.")
    if len(final):
        print(f"\nFINAL by level:")
        for lv, n in final.groupby("cur_level_2026").size().items():
            print(f"  {lv:<10} n={int(n):,}")
        print(f"  with eBay price: "
              f"{final['ebay_price_median'].notna().sum() if 'ebay_price_median' in final.columns else 0:,}")


if __name__ == "__main__":
    main()
