"""Canonical validation for the BUY UNIVERSE: non-R1, never top-100, entry<=2020.

What this script does:
  1. Load the held-out val long file (lasso-val players, never seen by hazards
     or lasso during training).
  2. Filter to UNIVERSE:
       - entry_year <= 2020 (mature outcomes through 2026)
       - bucket != R1 (R1 = round 1 draftees are too expensive)
       - never was a top-100 prospect (year_top_100 IS NULL in career_outcomes)
  3. Apply the lasso to compute score.
  4. Bin by ABSOLUTE lasso_score (not percentile) into 0.25-wide buckets.
  5. Report realized rate × n per (yip, score_bucket).
  6. Same analysis on the 2021-entry cohort (draftees + IFAs) at multi-snap
     to see how the model performs walk-forward on actual operational picks.

Usage:
    python validate_universe.py \\
        --long v1.17_val_long.csv \\
        --lasso models/lasso_v1.17_td.pkl \\
        --cohort2021-long v17_cohort2021_long.csv \\
        --out-prefix v17 \\
        --bucket-width 0.25
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd

LASSO_FEATURES = [
    "p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB", "p_STAR_PLUS_ELITE",
    "age_at_snap_centered", "years_in_pro",
    "p_TOP_100_PROSPECT_x_yip_centered", "p_MLB_DEBUT_x_yip_centered",
    "p_ESTABLISHED_MLB_x_yip_centered", "p_STAR_PLUS_ELITE_x_yip_centered",
]


def bucket(r):
    if int(r.get("is_international", 0) or 0) == 1:
        return "IFA"
    dr = r.get("draft_round")
    if pd.isna(dr) or dr is None:
        return "IFA"
    dr = int(dr)
    if dr == 1: return "R1"
    if dr <= 3: return "R2-R3"
    if dr <= 10: return "R4-R10"
    return "R10+"


def add_features_and_score(df, lasso_pkl, db="prospects_snapshot.db",
                           age_center=22, yip_center=3):
    # Normalize p_<event>_raw → p_<event> (walkforward variant)
    drop_cal = [c for c in df.columns
                if c.startswith("p_") and not c.endswith("_raw")
                and (c + "_raw") in df.columns]
    if drop_cal:
        df = df.drop(columns=drop_cal)
    rename = {c: c[:-4] for c in df.columns if c.endswith("_raw") and c.startswith("p_")}
    if rename:
        df = df.rename(columns=rename)
    if "realized_MLB_DEBUT" not in df.columns:
        for c in ("realized_after_snap_MLB_DEBUT",):
            if c in df.columns:
                df["realized_MLB_DEBUT"] = df[c].astype(int)
                break

    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"] - df["birth_year"]).fillna(22.0) - age_center)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - yip_center
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"):
        if f"p_{ev}" in df.columns:
            df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]

    with open(lasso_pkl, "rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    df["lasso_score"] = lasso.predict(sc.transform(df[feat].values))
    return df


def apply_universe_filter(df, db="prospects_snapshot.db"):
    """
    UNIVERSE:
      - bucket != R1
      - never was top-100 (year_top_100 IS NULL)
    """
    c = sqlite3.connect(db)
    meta = pd.read_sql("""
        SELECT p.player_id, p.draft_round, p.is_international,
               o.year_top_100
        FROM prospects p
        LEFT JOIN career_outcomes o ON o.player_id = p.player_id
    """, c)
    c.close()
    if "bucket" not in df.columns:
        df = df.merge(meta[["player_id","draft_round","is_international"]],
                      on="player_id", how="left", suffixes=("","_meta"))
        df["bucket"] = df.apply(bucket, axis=1)
    # Mark top100
    df = df.merge(meta[["player_id","year_top_100"]],
                  on="player_id", how="left", suffixes=("","_o"))
    n0 = len(df)
    df = df[df["bucket"] != "R1"].copy()
    n1 = len(df)
    df = df[df["year_top_100"].isna()].copy()
    n2 = len(df)
    print(f"  universe filter: {n0:,} -> drop R1 -> {n1:,} -> drop top100 -> {n2:,}")
    return df


def score_bucket_table(df, edges, labels, event="MLB_DEBUT"):
    df = df.copy()
    df["sb"] = pd.cut(df["lasso_score"], bins=edges, labels=labels, include_lowest=True)
    rows = []
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 5: continue
        base = sub[f"realized_{event}"].mean()
        for lbl in labels:
            cell = sub[sub.sb == lbl]
            if len(cell) == 0: continue
            n = len(cell); tp = int(cell[f"realized_{event}"].sum())
            rows.append({"yip": int(yip), "sb": lbl, "n": n, "tp": tp,
                         "rate": tp/n if n else np.nan,
                         "base_rate": base,
                         "lift": (tp/n)/base if base>0 else np.nan,
                         "score_lo": float(cell.lasso_score.min()),
                         "score_hi": float(cell.lasso_score.max())})
    return pd.DataFrame(rows)


def display_table(tbl, labels, title):
    rate = tbl.pivot(index="sb", columns="yip", values="rate").reindex(labels)
    nmat = tbl.pivot(index="sb", columns="yip", values="n").reindex(labels)
    keep = [c for c in rate.columns if c <= 10]
    print(f"\n=== {title} ===")
    print(f"  realized MLB_DEBUT rate (%) by (score_bucket, yip):")
    print(rate[keep].map(lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  ").to_string())
    print(f"\n  n per cell:")
    print(nmat[keep].map(lambda x: f"{int(x):>5d}" if pd.notna(x) else "     ").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, help="held-out val long file")
    ap.add_argument("--lasso", required=True)
    ap.add_argument("--cohort2021-long", default=None,
                    help="walkforward long file for 2021-entry cohort")
    ap.add_argument("--out-prefix", default="val")
    ap.add_argument("--bucket-width", type=float, default=0.25)
    ap.add_argument("--score-min", type=float, default=-0.5)
    ap.add_argument("--score-max", type=float, default=5.0)
    ap.add_argument("--db", default="prospects_snapshot.db")
    args = ap.parse_args()

    edges = [-99.0] + list(np.arange(args.score_min, args.score_max + 0.001, args.bucket_width)) + [99.0]
    edges = sorted(set([round(e,3) for e in edges]))
    labels = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        if lo == -99.0: labels.append(f"<{hi:.2f}")
        elif hi == 99.0: labels.append(f">={lo:.2f}")
        else: labels.append(f"{lo:.2f}-{hi:.2f}")

    print("="*80)
    print(f"UNIVERSE: non-R1, never top-100, entry_year <= 2020")
    print(f"Bucket edges: {edges}")
    print("="*80)

    # --- 1. Held-out val slice ---
    print(f"\n[1] HELD-OUT VAL SLICE  ({args.long})")
    val = pd.read_csv(args.long)
    val = val[val.entry_year <= 2020].copy()
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT"):
        if f"eligible_{ev}" in val.columns:
            val = val[val[f"eligible_{ev}"]==1]
    val = apply_universe_filter(val, db=args.db)
    val = add_features_and_score(val, args.lasso, db=args.db)
    print(f"  scored: {len(val):,} rows, {val.player_id.nunique():,} players")
    val_tbl = score_bucket_table(val, edges, labels)
    val_tbl.to_csv(f"val_{args.out_prefix}_UNIVERSE_score_buckets.csv", index=False)
    display_table(val_tbl, labels,
                  f"VAL slice (held-out, UNIVERSE-filtered)")

    # --- 2. 2021-entry walkforward (operational test set) ---
    if args.cohort2021_long:
        print(f"\n\n[2] 2021-ENTRY COHORT  ({args.cohort2021_long})")
        c21 = pd.read_csv(args.cohort2021_long)
        c21 = apply_universe_filter(c21, db=args.db)
        c21 = add_features_and_score(c21, args.lasso, db=args.db)
        print(f"  scored: {len(c21):,} rows, {c21.player_id.nunique():,} players")
        c21_tbl = score_bucket_table(c21, edges, labels)
        c21_tbl.to_csv(f"val_{args.out_prefix}_UNIVERSE_2021_score_buckets.csv", index=False)
        display_table(c21_tbl, labels,
                      f"2021-entry cohort (UNIVERSE-filtered, all snaps)")

        # Per-snap top-N realized
        print(f"\n  Per-snap top-N realized debut (UNIVERSE):")
        print(f"  {'snap':>5} {'n':>5} {'base':>6} {'top10':>10} {'top25':>10} {'top50':>10} {'top100':>11}")
        for snap_yr in sorted(c21.snap_year.unique()):
            sub = c21[c21.snap_year == snap_yr].sort_values("lasso_score", ascending=False)
            if len(sub) < 10: continue
            base = sub.realized_MLB_DEBUT.mean()
            def topn(n):
                k = min(n, len(sub))
                if k == 0: return ""
                tp = sub.head(k).realized_MLB_DEBUT.sum()
                return f"{tp}/{k} {tp/k:.0%}"
            print(f"  {int(snap_yr):>5} {len(sub):>5} {base*100:>5.1f}% "
                  f"{topn(10):>10} {topn(25):>10} {topn(50):>10} {topn(100):>11}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
