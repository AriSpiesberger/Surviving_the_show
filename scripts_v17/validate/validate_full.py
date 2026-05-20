"""Granular per-yip × per-percentile validation.

Reports per-yip per-event the realized outcome rate in fine-grained percentile
slabs:

  TOP HEAD     (precision matters most here)
    0.0-0.5%   0.5-1.0%   1.0-1.5%   1.5-2.0%
  MID
    2-3%   3-4%   4-5%
  WIDER
    5-10%   10-20%   20-50%   bottom 50%

For each (yip, slab) cell, emits:
    score floor, score ceiling, mean score, n, realized rate, expected lift
    (rate/base_rate)

And a LOOK-BACK retrospective:
    Across all val players in slab, list the actual outcomes (cup/utility/
    regular/breakout if model_b available) and the names of the top picks
    to spot-check.

Output per event:
    val_<prefix>_<event>_pct_slabs.csv      machine-readable per-cell metrics
    val_<prefix>_<event>_top_lookback.csv   names + outcomes of top picks per yip

Console: per-yip × slab table for the primary event (default MLB_DEBUT).

Usage:
    python validate_full.py \\
        --long v1.14n_val_long.csv \\
        --lasso lasso_v1.14n_td.pkl \\
        --model-b model_b_outcomes_v1.14n.pkl \\
        --out-prefix v14n
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

EVENTS = ["MLB_DEBUT", "TOP_100_PROSPECT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
SLABS = [
    (0.0, 0.5, "0.0-0.5%"),
    (0.5, 1.0, "0.5-1.0%"),
    (1.0, 1.5, "1.0-1.5%"),
    (1.5, 2.0, "1.5-2.0%"),
    (2.0, 3.0, "2-3%"),
    (3.0, 4.0, "3-4%"),
    (4.0, 5.0, "4-5%"),
    (5.0, 10.0, "5-10%"),
    (10.0, 20.0, "10-20%"),
    (20.0, 50.0, "20-50%"),
    (50.0, 100.0, "bottom 50%"),
]


def score_with_lasso(long_csv, lasso_pkl, age_center=22, yip_center=3, db="prospects_snapshot.db"):
    df = pd.read_csv(long_csv)
    df = df[df.entry_year <= 2020].copy()
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT"):
        df = df[df[f"eligible_{ev}"]==1]
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - age_center)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - yip_center
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    with open(lasso_pkl,"rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    df["lasso_score"] = lasso.predict(sc.transform(df[feat].values))
    return df


def slab_metrics(scores, y, lo_pct, hi_pct):
    n = len(scores)
    lo = max(0, int(round(n*lo_pct/100)))
    hi = min(n, int(round(n*hi_pct/100)))
    if hi <= lo:
        return None
    order = np.argsort(-scores)
    idx = order[lo:hi]
    seg_scores = scores[idx]; seg_y = y[idx]
    return {
        "n": len(idx),
        "score_lo": float(seg_scores.min()),
        "score_hi": float(seg_scores.max()),
        "score_mean": float(seg_scores.mean()),
        "rate": float(seg_y.mean()),
        "tp": int(seg_y.sum()),
        "idx": idx,
    }


def per_yip_slabs(df, event, slabs):
    rows = []
    base_by_yip = {}
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 100: continue
        y = sub[f"realized_{event}"].values.astype(int)
        if y.sum() == 0: continue
        base = y.mean(); base_by_yip[yip] = base
        scores = sub["lasso_score"].values
        for lo, hi, label in slabs:
            r = slab_metrics(scores, y, lo, hi)
            if r is None: continue
            rows.append({
                "event": event, "yip": yip, "slab": label,
                "slab_lo": lo, "slab_hi": hi,
                "n": r["n"], "tp": r["tp"], "rate": r["rate"],
                "base_rate": base, "lift": r["rate"]/base if base>0 else np.nan,
                "score_lo": r["score_lo"], "score_hi": r["score_hi"],
                "score_mean": r["score_mean"],
            })
    return pd.DataFrame(rows), base_by_yip


def lookback_top_picks(df, event, slabs, top_slab_idx=4, model_b=None, scaler_b=None,
                      classes_b=None, feat_names_b=None):
    """Capture names and full outcome distribution for the union of the top
    `top_slab_idx` slabs (default 4 = 0-2%) per yip."""
    rows = []
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 100: continue
        y = sub[f"realized_{event}"].values.astype(int)
        if y.sum() == 0: continue
        scores = sub["lasso_score"].values
        n = len(sub)
        # Top union of slabs
        top_hi = max(slabs[i][1] for i in range(min(top_slab_idx, len(slabs))))
        k = max(1, int(round(n*top_hi/100)))
        order = np.argsort(-scores)[:k]
        seg = sub.iloc[order].copy()
        seg["realized"] = seg[f"realized_{event}"].astype(int).values
        # Slab assignment within this top
        seg["slab"] = ""
        n_total = n
        for j, idx in enumerate(order):
            pct = (j+1) / n_total * 100
            for lo, hi, label in slabs[:top_slab_idx]:
                if lo < pct <= hi:
                    seg.iloc[j, seg.columns.get_loc("slab")] = label
                    break
        # Add model_b outcomes if provided
        if model_b is not None:
            # logit features
            eps = 1e-6
            feat_df = pd.DataFrame()
            haz_map = {
                "logit_p_TOP_100_PROSPECT": "p_TOP_100_PROSPECT",
                "logit_p_MLB_DEBUT": "p_MLB_DEBUT",
                "logit_p_ESTABLISHED_MLB": "p_ESTABLISHED_MLB",
                "logit_p_STAR_PLUS_ELITE": "p_STAR_PLUS_ELITE",
            }
            for lc, sc_col in haz_map.items():
                p = seg[sc_col].clip(eps, 1-eps).values
                feat_df[lc] = np.log(p / (1-p))
            feat_df["yrs_pre_debut"] = seg["snap_offset"].astype(float).values
            buckets = ["b_R1","b_R10+","b_R2-R3","b_R4-R10"]
            for b in buckets:
                feat_df[b] = (seg["bucket"] == b.replace("b_","")).astype(float).values
            # pos_grp default OTH
            feat_df["pos_IF"] = 0.0; feat_df["pos_OF"] = 0.0; feat_df["pos_OTH"] = 0.0
            try:
                feat_df = feat_df[feat_names_b]
                P = model_b.predict_proba(scaler_b.transform(feat_df.values))
                for i, c in enumerate(classes_b):
                    seg[f"p_b_{c}"] = P[:, i]
            except Exception as e:
                print(f"  WARN model_b apply failed: {e}")
        rows.append(seg)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["snap_offset","slab","player_id","name","bucket","lasso_score",
            f"realized_{event}","realized",
            "p_MLB_DEBUT","p_TOP_100_PROSPECT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE"]
    keep += [c for c in out.columns if c.startswith("p_b_")]
    keep = [c for c in keep if c in out.columns]
    return out[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--lasso", required=True)
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--out-prefix", default="val")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--events", nargs="+", default=EVENTS)
    ap.add_argument("--cohort2021-long", default=None,
                    help="Path to long file scored on 2021 draftees (e.g. "
                         "walkforward_v1.14n_2021_long.csv). If set, runs a "
                         "look-back at snap=2022 and snap=2023.")
    args = ap.parse_args()

    df = score_with_lasso(args.long, args.lasso, db=args.db)
    print(f"scored val: {len(df):,} rows, {df.player_id.nunique():,} players")

    # Optional model B
    model_b = scaler_b = classes_b = feat_names_b = None
    if args.model_b:
        with open(args.model_b, "rb") as fh:
            mb = pickle.load(fh)
        model_b = mb["model"]; scaler_b = mb["scaler"]
        classes_b = mb["classes"]; feat_names_b = mb["feature_names"]
        print(f"loaded model B: classes={classes_b}")

    for ev in args.events:
        print(f"\n{'='*78}\nEVENT: {ev}\n{'='*78}")
        slab_df, base_by_yip = per_yip_slabs(df, ev, SLABS)
        if slab_df.empty:
            print("  (no data)"); continue
        out_slab = f"val_{args.out_prefix}_{ev}_pct_slabs.csv"
        slab_df.to_csv(out_slab, index=False)
        print(f"saved {out_slab}")

        # Pretty pivot per event
        rate = slab_df.pivot(index="slab", columns="yip", values="rate").reindex([s[2] for s in SLABS])
        n    = slab_df.pivot(index="slab", columns="yip", values="n").reindex([s[2] for s in SLABS])
        sc_lo = slab_df.pivot(index="slab", columns="yip", values="score_lo").reindex([s[2] for s in SLABS])
        print(f"\n--- realized {ev} rate (%) by yip × slab ---")
        print(rate.map(lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  ").to_string())
        print(f"\n--- n per cell ---")
        print(n.map(lambda x: f"{int(x):>5d}" if pd.notna(x) else "     ").to_string())
        print(f"\n--- score floor per cell ---")
        print(sc_lo.map(lambda x: f"{x:>+6.3f}" if pd.notna(x) else "       ").to_string())
        print(f"\n--- base rate by yip ---")
        for y, b in sorted(base_by_yip.items()):
            print(f"  yip={y}: {b:.2%}")

        # Look-back: actual top picks
        lb = lookback_top_picks(df, ev, SLABS, top_slab_idx=4,
                                model_b=model_b, scaler_b=scaler_b,
                                classes_b=classes_b, feat_names_b=feat_names_b)
        if not lb.empty:
            out_lb = f"val_{args.out_prefix}_{ev}_top_lookback.csv"
            lb.to_csv(out_lb, index=False)
            print(f"\nsaved {out_lb}  ({len(lb)} top-2% rows)")
            # Show summary: per yip × slab → realized rate
            agg = lb.groupby(["snap_offset","slab"]).agg(
                n=("realized","size"), realized=("realized","sum")
            ).reset_index()
            agg["rate"] = agg["realized"]/agg["n"]
            print("\nlook-back per yip × slab (top 0-2%):")
            print(agg.to_string(index=False))

    if args.cohort2021_long:
        cohort2021_lookback(args.cohort2021_long, args.lasso, args.out_prefix, db=args.db)


def _unused_cohort2021_lookback_placeholder():
    pass


def cohort2021_lookback(long_csv, lasso_pkl, prefix, db="prospects_snapshot.db"):
    """Score 2021 draftees with the same lasso at snap=2022 and snap=2023,
    bin by per-yip percentile slab, and report realized rates so far."""
    print(f"\n{'='*78}\n2021 DRAFTEE LOOK-BACK — apply same lasso at snap 2022/2023\n{'='*78}")
    df = pd.read_csv(long_csv)
    print(f"  loaded {len(df):,} rows, {df.player_id.nunique():,} players")
    # Walk-forward file already has cumulative event probs at each snap
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - 22)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - 3
    # walkforward long has BOTH p_<event> (Beta-calibrated) AND p_<event>_raw.
    # Lasso was trained on raw hazards, so drop the calibrated ones first then
    # rename _raw -> base name to match feature names.
    drop_cal = [c for c in df.columns
                if c.startswith("p_") and not c.endswith("_raw")
                and (c + "_raw") in df.columns]
    df = df.drop(columns=drop_cal)
    rename_map = {}
    for col in df.columns:
        if col.endswith("_raw") and col.startswith("p_"):
            rename_map[col] = col[:-4]
    df = df.rename(columns=rename_map)
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        if f"p_{ev}" not in df.columns: continue
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    with open(lasso_pkl,"rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    missing = [f for f in feat if f not in df.columns]
    if missing:
        print(f"  WARN missing features in 2021 long: {missing}")
        return
    df["lasso_score"] = lasso.predict(sc.transform(df[feat].values))

    # walkforward long uses 'realized_after_snap_<E>' (or similar); detect
    real_col = None
    for cand in ("realized_after_snap_MLB_DEBUT","realized_MLB_DEBUT"):
        if cand in df.columns: real_col = cand; break
    if real_col is None:
        # try any "realized_*MLB_DEBUT" column
        for c in df.columns:
            if c.startswith("realized") and "MLB_DEBUT" in c:
                real_col = c; break
    if real_col is None:
        print("  WARN no realized_MLB_DEBUT col in 2021 long")
        return
    df["realized_MLB_DEBUT"] = df[real_col].astype(int)

    print(f"  scored 2021 cohort with same lasso (features match)")
    print(f"  realized MLB_DEBUT by snap (out of {df.player_id.nunique()} players):")
    for snap in sorted(df.snap_year.unique()):
        sub = df[df.snap_year == snap]
        print(f"    snap={int(snap)}  yip={int(sub.snap_offset.iloc[0])}  "
              f"n={len(sub)}  realized={int(sub.realized_MLB_DEBUT.sum())} "
              f"({sub.realized_MLB_DEBUT.mean():.1%})")

    # Per-snap slab analysis (snap=2022, 2023)
    rows = []
    for snap in (2022, 2023):
        sub = df[df.snap_year == snap]
        if len(sub) < 50: continue
        y = sub["realized_MLB_DEBUT"].values
        if y.sum() == 0:
            print(f"\n  snap={snap}: zero realized — no signal to evaluate yet")
            continue
        base = y.mean()
        print(f"\n--- 2021 draftees at snap={snap} (yip={int(sub.snap_offset.iloc[0])}) ---")
        print(f"  n={len(sub)}  base_rate={base:.1%}  realized={int(y.sum())}")
        print(f"  {'slab':>10} {'n':>4} {'TP':>4} {'rate':>7} {'lift':>5} {'score_lo':>9} {'score_hi':>9}")
        scores = sub["lasso_score"].values
        for lo, hi, label in SLABS:
            r = slab_metrics(scores, y, lo, hi)
            if r is None or r["n"] < 1: continue
            lift = r["rate"]/base if base>0 else np.nan
            print(f"  {label:>10} {r['n']:>4d} {r['tp']:>4d} {r['rate']*100:>6.1f}% {lift:>4.1f}x {r['score_lo']:>+9.3f} {r['score_hi']:>+9.3f}")
            rows.append({"snap":snap,"yip":int(sub.snap_offset.iloc[0]),"slab":label,
                         "n":r["n"],"tp":r["tp"],"rate":r["rate"],"lift":lift,
                         "score_lo":r["score_lo"],"score_hi":r["score_hi"]})
        # Top picks: list them
        top_k = max(1, int(len(sub) * 0.02))
        order = np.argsort(-scores)[:top_k]
        top = sub.iloc[order][["player_id","name","snap_offset","lasso_score","realized_MLB_DEBUT","p_MLB_DEBUT"]].copy()
        out_lb = f"val_{prefix}_2021lookback_snap{snap}_top2pct.csv"
        top.to_csv(out_lb, index=False)
        print(f"  saved top-2% picks → {out_lb}")

    if rows:
        pd.DataFrame(rows).to_csv(f"val_{prefix}_2021lookback_slabs.csv", index=False)


if __name__ == "__main__":
    main()
