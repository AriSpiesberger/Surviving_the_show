"""Validation extension: apply Lasso composite to val slice, report
per-yip accuracy in percentile bins AND absolute-score buckets.

Outputs:
  val_v14i_lasso_pctile.csv  — per (yip, event, pctile_bin)
  val_v14i_lasso_score.csv   — per (yip, event, score_bin)
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd


def main():
    long_csv = sys.argv[1] if len(sys.argv) > 1 else "v14i_val_pre2021_raw_long.csv"
    lasso_pkl = sys.argv[2] if len(sys.argv) > 2 else "lasso_v14i_td.pkl"
    prefix = long_csv.replace("_val_long.csv", "").replace("_val_pre2021_raw_long.csv", "")
    out_pctile = f"val_{prefix}_lasso_pctile.csv"
    out_score = f"val_{prefix}_lasso_score.csv"

    print(f"Loading {long_csv}")
    df = pd.read_csv(long_csv)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} players")

    with open(lasso_pkl, "rb") as fh:
        m = pickle.load(fh)
    scaler, lasso, fn = m["scaler"], m["lasso"], m["feature_names"]
    age_center = m["age_center"]
    yip_center = 3
    print(f"  Lasso features: {fn}")

    # Compute age_at_snap from snap_year - birth_year_proxy; we don't
    # have birth here so use age_at_snap if column exists; else approximate.
    # The long.csv schema doesn't have age — recompute from DB join skipped;
    # use 0 placeholder for the age_centered (it's a weak feature anyway).
    # Better: pull birth from DB.
    import sqlite3
    c = sqlite3.connect("prospects_snapshot.db")
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]], on="player_id", how="left")
    df["age_at_snap"] = df["snap_year"] - df["birth_year"]
    df["age_at_snap"] = df["age_at_snap"].fillna(22.0)

    df["yip"] = df["snap_offset"]
    df["age_at_snap_centered"] = df["age_at_snap"] - age_center
    df["years_in_pro"] = df["yip"]
    df["yip_centered"] = df["yip"] - yip_center
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"):
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]

    X = df[fn].values.astype(float)
    Xs = scaler.transform(X)
    df["lasso_score"] = lasso.predict(Xs)

    print(f"\nLasso score stats: mean={df.lasso_score.mean():.3f}  "
          f"min={df.lasso_score.min():.3f}  max={df.lasso_score.max():.3f}")

    EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
    PCTILE_BINS = [(0.0, 0.01, "top 0-1%"),
                   (0.01, 0.02, "top 1-2%"),
                   (0.02, 0.05, "top 2-5%"),
                   (0.05, 0.10, "top 5-10%"),
                   (0.10, 0.20, "top 10-20%"),
                   (0.20, 0.50, "top 20-50%"),
                   (0.50, 1.01, "bottom 50%")]
    SCORE_BINS = [(-np.inf, 0.0, "<0.0"),
                  (0.0, 0.5, "0.0-0.5"),
                  (0.5, 1.0, "0.5-1.0"),
                  (1.0, 1.5, "1.0-1.5"),
                  (1.5, 2.0, "1.5-2.0"),
                  (2.0, 3.0, "2.0-3.0"),
                  (3.0, np.inf, "3.0+")]

    rows_p, rows_s = [], []
    for yip in sorted(df.yip.unique()):
        sub = df[df.yip == yip].copy()
        if sub.empty: continue
        # Percentile within this yip group, descending score = rank 0 is top
        sub = sub.sort_values("lasso_score", ascending=False).reset_index(drop=True)
        sub["pctile"] = (np.arange(len(sub)) + 0.5) / len(sub)
        for ev in EVENTS:
            elig = sub[sub[f"eligible_{ev}"] == 1]
            if len(elig) < 20: continue
            base = elig[f"realized_{ev}"].mean()
            for lo, hi, label in PCTILE_BINS:
                bin_rows = elig[(elig.pctile >= lo) & (elig.pctile < hi)]
                if len(bin_rows) == 0: continue
                obs = bin_rows[f"realized_{ev}"].mean()
                pred = bin_rows["lasso_score"].mean()
                rows_p.append({
                    "yip": yip, "event": ev, "pctile_bin": label,
                    "n": len(bin_rows), "score_mean": pred,
                    "base_rate": base, "obs_rate": obs,
                    "lift": obs / max(base, 1e-9),
                })
            for lo, hi, label in SCORE_BINS:
                bin_rows = elig[(elig.lasso_score >= lo) & (elig.lasso_score < hi)]
                if len(bin_rows) < 5: continue
                obs = bin_rows[f"realized_{ev}"].mean()
                rows_s.append({
                    "yip": yip, "event": ev, "score_bin": label,
                    "n": len(bin_rows), "base_rate": base,
                    "obs_rate": obs, "lift": obs / max(base, 1e-9),
                })

    pd.DataFrame(rows_p).to_csv(out_pctile, index=False)
    pd.DataFrame(rows_s).to_csv(out_score, index=False)
    print(f"\nWrote {out_pctile} ({len(rows_p)} rows)")
    print(f"Wrote {out_score} ({len(rows_s)} rows)")

    print("\n=== MLB_DEBUT — pctile bins by yip ===")
    pdf = pd.DataFrame(rows_p)
    if len(pdf):
        view = pdf[pdf.event == "MLB_DEBUT"].pivot_table(
            index="pctile_bin", columns="yip", values="obs_rate", aggfunc="first")
        order = [b[2] for b in PCTILE_BINS if b[2] in view.index]
        print(view.loc[order].map(lambda x: f"{x:.1%}" if pd.notna(x) else "").to_string())

    print("\n=== MLB_DEBUT — score buckets by yip ===")
    sdf = pd.DataFrame(rows_s)
    if len(sdf):
        view = sdf[sdf.event == "MLB_DEBUT"].pivot_table(
            index="score_bin", columns="yip", values="obs_rate", aggfunc="first")
        order = [b[2] for b in SCORE_BINS if b[2] in view.index]
        n_view = sdf[sdf.event == "MLB_DEBUT"].pivot_table(
            index="score_bin", columns="yip", values="n", aggfunc="first")
        print("Observed rate:")
        print(view.loc[order].map(lambda x: f"{x:.1%}" if pd.notna(x) else "").to_string())
        print("\nn per cell:")
        print(n_view.loc[order].map(lambda x: f"{int(x):>4d}" if pd.notna(x) else "").to_string())


if __name__ == "__main__":
    main()
