"""Score-bucket validation: bin by absolute lasso_score, not percentile rank.

For each (yip, score_bucket) cell, reports:
  - n players in cell
  - realized MLB_DEBUT rate
  - lift vs yip base rate
  - score min/max

Also runs the 2021 draftee look-back: same score-bucket scheme applied
to 2021 cohort at snap=2022 and snap=2023, listing ALL players per bucket
(not just top-2%).

Usage:
    python validate_score_buckets.py \\
        --long v1.17_val_long.csv \\
        --lasso models/lasso_v1.17_td.pkl \\
        --cohort2021-long walkforward_v17_2021_long.csv \\
        --out-prefix v17 \\
        --bucket-width 0.5
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd


def add_feats(df, age_center=22, yip_center=3, db="prospects_snapshot.db"):
    df = df.copy()
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - age_center)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - yip_center
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        if f"p_{ev}" not in df.columns:
            continue
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    return df


def score(long_csv, lasso_pkl, max_entry_year=None):
    df = pd.read_csv(long_csv)
    # walkforward variant: collapse p_<event>_raw → p_<event>
    drop_cal = [c for c in df.columns
                if c.startswith("p_") and not c.endswith("_raw")
                and (c + "_raw") in df.columns]
    df = df.drop(columns=drop_cal)
    rename = {c: c[:-4] for c in df.columns if c.endswith("_raw") and c.startswith("p_")}
    df = df.rename(columns=rename)

    if max_entry_year is not None and "entry_year" in df.columns:
        df = df[df.entry_year <= max_entry_year].copy()
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT"):
        col = f"eligible_{ev}"
        if col in df.columns:
            df = df[df[col]==1]

    df = add_feats(df)
    with open(lasso_pkl,"rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    df["lasso_score"] = lasso.predict(sc.transform(df[feat].values))

    # walkforward sometimes uses realized_after_snap_MLB_DEBUT instead of realized_MLB_DEBUT
    if "realized_MLB_DEBUT" not in df.columns:
        for c in ("realized_after_snap_MLB_DEBUT",):
            if c in df.columns:
                df["realized_MLB_DEBUT"] = df[c].astype(int)
                break
    return df


def score_bucket_table(df, edges, labels, event="MLB_DEBUT"):
    df = df.copy()
    df["bucket"] = pd.cut(df["lasso_score"], bins=edges, labels=labels, include_lowest=True)
    rows = []
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 5:
            continue
        base = sub[f"realized_{event}"].mean()
        for lbl in labels:
            cell = sub[sub.bucket == lbl]
            n = len(cell)
            if n == 0:
                continue
            tp = int(cell[f"realized_{event}"].sum())
            rows.append({
                "event": event, "yip": int(yip), "bucket": lbl,
                "n": n, "tp": tp, "rate": tp/n if n else np.nan,
                "base_rate": base, "lift": (tp/n)/base if base>0 and n else np.nan,
                "score_lo": float(cell.lasso_score.min()),
                "score_hi": float(cell.lasso_score.max()),
                "score_mean": float(cell.lasso_score.mean()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--lasso", required=True)
    ap.add_argument("--cohort2021-long", default=None)
    ap.add_argument("--out-prefix", default="val")
    ap.add_argument("--bucket-width", type=float, default=0.5,
                    help="width of each score bucket (default 0.5)")
    ap.add_argument("--score-min", type=float, default=-0.5)
    ap.add_argument("--score-max", type=float, default=5.0)
    ap.add_argument("--db", default="prospects_snapshot.db")
    args = ap.parse_args()

    # Build bucket edges
    edges = [-99.0] + list(np.arange(args.score_min, args.score_max + 0.001, args.bucket_width)) + [99.0]
    edges = sorted(set([round(e,3) for e in edges]))
    labels = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        if lo == -99.0:
            labels.append(f"<{hi:.2f}")
        elif hi == 99.0:
            labels.append(f">={lo:.2f}")
        else:
            labels.append(f"{lo:.2f}-{hi:.2f}")

    print(f"Score buckets: {labels}\n")

    # --- val slice ---
    print(f"=== VAL SLICE  ({args.long}) ===")
    val = score(args.long, args.lasso)
    print(f"  rows: {len(val):,}  players: {val.player_id.nunique():,}")
    val_tbl = score_bucket_table(val, edges, labels)
    val_tbl.to_csv(f"val_{args.out_prefix}_score_buckets.csv", index=False)

    # Pivot for display
    rate = val_tbl.pivot(index="bucket", columns="yip", values="rate").reindex(labels)
    nmat = val_tbl.pivot(index="bucket", columns="yip", values="n").reindex(labels)
    keep = [c for c in rate.columns if c <= 10]
    print(f"\n--- Realized MLB_DEBUT rate by (score_bucket, yip) ---")
    print(rate[keep].map(lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  ").to_string())
    print(f"\n--- n per cell ---")
    print(nmat[keep].map(lambda x: f"{int(x):>5d}" if pd.notna(x) else "     ").to_string())
    print(f"\nsaved val_{args.out_prefix}_score_buckets.csv")

    # --- 2021 lookback ---
    if args.cohort2021_long:
        print(f"\n=== 2021 COHORT LOOK-BACK  ({args.cohort2021_long}) ===")
        c21 = score(args.cohort2021_long, args.lasso)
        print(f"  rows: {len(c21):,}  players: {c21.player_id.nunique():,}")
        c21_tbl = score_bucket_table(c21, edges, labels)
        c21_tbl.to_csv(f"val_{args.out_prefix}_2021_score_buckets.csv", index=False)

        rate21 = c21_tbl.pivot(index="bucket", columns="yip", values="rate").reindex(labels)
        n21 = c21_tbl.pivot(index="bucket", columns="yip", values="n").reindex(labels)
        keep21 = [c for c in rate21.columns if c <= 10]
        print(f"\n--- 2021 cohort realized MLB_DEBUT rate by (score_bucket, yip) ---")
        print(rate21[keep21].map(lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  ").to_string())
        print(f"\n--- n per cell ---")
        print(n21[keep21].map(lambda x: f"{int(x):>5d}" if pd.notna(x) else "     ").to_string())
        print(f"\nsaved val_{args.out_prefix}_2021_score_buckets.csv")

        # ALL players per (yip in {1,2}, score_bucket) with realized
        c21["bucket"] = pd.cut(c21["lasso_score"], bins=edges, labels=labels, include_lowest=True)
        for snap_yip in (1, 2):
            sub = c21[c21.snap_offset == snap_yip].sort_values("lasso_score", ascending=False)
            sub_out = sub[["player_id","name","snap_year","snap_offset","bucket","lasso_score",
                           "realized_MLB_DEBUT","p_MLB_DEBUT","p_TOP_100_PROSPECT"]]
            fname = f"val_{args.out_prefix}_2021_yip{snap_yip}_all_buckets.csv"
            sub_out.to_csv(fname, index=False)
            print(f"  saved {fname}  ({len(sub_out)} players)")


if __name__ == "__main__":
    main()
