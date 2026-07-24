"""v1.18b time-to-debut regression — adds landmark mean_t / sd_t features.

DIFF from fit_time_to_debut_v18.py:
  - The landmark inference produces explicit per-event timing moments
    (mean_t, sd_t) from sum_tp / sum_p across the horizon walk. These carry
    the model's "when?" signal directly, instead of forcing the regressor to
    back it out of the cumulative shape.

  - v1.18 (contemporaneous-LOCF) gets a free near-term timing signal from
    its LOCF bias — the cumulative spikes early because the model
    over-projects stalled-but-frozen profiles. v1.18b cleaned that up,
    and on cumulative-only features the timing MAE got worse (+0.10 on
    val, +0.50 specifically at actual_h=1). Adding mean_t restores the
    timing channel from the model itself.

  - mean_t is the model's E[T | event fires in horizon], horizon-truncated.
    Use mean_t for MLB_DEBUT and ESTABLISHED (these are the two events that
    bear on debut timing directly; STAR/ELITE timing is mostly noise for
    debut prediction). sd_t for MLB_DEBUT only (the uncertainty has
    information content; sd of the others is mostly correlated).

Same downstream interface as fit_time_to_debut_v18.py: writes a pkl with
{scaler, lasso, feature_names, ...} so build_v1.18_buylist.py / its v1.18b
sibling can apply it via the same `_score_timing` codepath.
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "STAR_PLUS_ELITE"]
# Base feature set inherits from v1.18 plus the v1.18b landmark-only timing
# moments. Adding mean_t for two events (debut and established) and sd_t for
# debut keeps the parameter count low and the signal targeted.
BASE_FEAT = (
    [f"p_{e}" for e in EVENTS]
    + ["age_at_snap_centered", "years_in_pro"]
    + [f"p_{e}_x_yip_centered" for e in EVENTS]
)
LANDMARK_TIMING_FEAT = [
    "mean_t_MLB_DEBUT",
    "sd_t_MLB_DEBUT",
    "mean_t_ESTABLISHED_MLB",
]
AGE_CENTER = 22
YIP_CENTER = 3


def add_feats(df: pd.DataFrame, db: str) -> pd.DataFrame:
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(
        birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]],
                  on="player_id", how="left")
    df["age_at_snap_centered"] = (
        (df["snap_year"] - df["birth_year"]).fillna(22.0) - AGE_CENTER
    )
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - YIP_CENTER
    for e in EVENTS:
        df[f"p_{e}_x_yip_centered"] = df[f"p_{e}"] * df["yip_centered"]
    return df


def _attach_p_debut(df, bundle_pkl):
    with open(bundle_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    art = bundle["per_event"]["MLB_DEBUT"]
    sc, lasso, feat = art["scaler"], art["lasso"], art["feature_names"]
    for col in feat:
        if col in df.columns:
            continue
        if col.endswith("_x_yip_centered"):
            base = col[:-len("_x_yip_centered")]
            if base in df.columns and "yip_centered" in df.columns:
                df[col] = df[base] * df["yip_centered"]
    df["p_debut_lasso"] = lasso.predict_proba(
        sc.transform(df[feat].values))[:, 1]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--bundle",
                    default="models/lasso_logits_v1.18b_prod.pkl")
    ap.add_argument("--include-p-debut", action="store_true")
    ap.add_argument("--max-h", type=int, default=12)
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--out", default="models/time_to_debut_v1.18b_prod.pkl")
    args = ap.parse_args()

    print(f"[v1.18b timing] loading {args.fit}")
    fit = pd.read_csv(args.fit)
    val = pd.read_csv(args.val)
    print(f"  fit: {len(fit):,} rows, val: {len(val):,} rows")

    # Verify the long CSV has the landmark timing columns. If not, fall
    # back to the v1.18 base feature set with a loud warning — keeps the
    # script usable on contemporaneous longs for ablation.
    missing_lm = [c for c in LANDMARK_TIMING_FEAT if c not in fit.columns]
    if missing_lm:
        print(f"  WARN landmark timing features missing from fit: "
              f"{missing_lm}. Falling back to BASE_FEAT only.", flush=True)
        landmark_feat = []
    else:
        landmark_feat = list(LANDMARK_TIMING_FEAT)

    fit = add_feats(fit, args.db)
    val = add_feats(val, args.db)
    feat = list(BASE_FEAT) + landmark_feat
    if args.include_p_debut:
        fit = _attach_p_debut(fit, args.bundle)
        val = _attach_p_debut(val, args.bundle)
        feat.append("p_debut_lasso")

    def _prep(df):
        d = df.dropna(subset=["mlb_debut_year"]).copy()
        d["mlb_debut_year"] = d["mlb_debut_year"].astype(int)
        d = d[d.snap_year < d.mlb_debut_year]
        d = d[d.entry_year <= args.max_entry]
        d["time_to_debut"] = (d["mlb_debut_year"] - d["snap_year"]).clip(
            upper=args.max_h)
        d = d.dropna(subset=feat + ["time_to_debut"])
        return d

    tr = _prep(fit); va = _prep(val)
    print(f"  train (debutee pre-debut snaps): {len(tr):,} rows, "
          f"{tr.player_id.nunique():,} players")
    print(f"  val   (debutee pre-debut snaps): {len(va):,} rows, "
          f"{va.player_id.nunique():,} players")
    print(f"  features ({len(feat)}): {feat}")

    X_tr = tr[feat].values.astype(float)
    y_tr = tr["time_to_debut"].values.astype(float)
    g_tr = tr["player_id"].values
    X_va = va[feat].values.astype(float)
    y_va = va["time_to_debut"].values.astype(float)

    scaler = StandardScaler().fit(X_tr)
    splits = list(GroupKFold(5).split(X_tr, y_tr, g_tr))
    lasso = LassoCV(
        cv=splits, alphas=np.logspace(-3, 1, 30),
        max_iter=20000, n_jobs=-1, random_state=42,
    ).fit(scaler.transform(X_tr), y_tr)
    print(f"\nLassoCV chose alpha={lasso.alpha_:.4g}")
    print(f"non-zero coefficients:")
    pairs = sorted(
        ((n, c) for n, c in zip(feat, lasso.coef_) if abs(c) > 1e-6),
        key=lambda kv: -abs(kv[1]),
    )
    for n, c in pairs:
        print(f"  {n:<42} {c:+.4f}")
    print(f"  {'intercept':<42} {lasso.intercept_:+.4f}")

    pred_tr = lasso.predict(scaler.transform(X_tr))
    mae_tr = mean_absolute_error(y_tr, pred_tr)
    sp_tr = spearmanr(pred_tr, y_tr).correlation
    print(f"\nin-train MAE={mae_tr:.2f}  Spearman={sp_tr:+.3f}")

    pred_va = lasso.predict(scaler.transform(X_va))
    mae_va = mean_absolute_error(y_va, pred_va)
    sp_va = spearmanr(pred_va, y_va).correlation
    pr_va = float(np.corrcoef(pred_va, y_va)[0, 1])
    print(f"\n===== HONEST VAL (v1.18b) =====")
    print(f"  MAE = {mae_va:.3f} years")
    print(f"  Spearman = {sp_va:+.4f}")
    print(f"  Pearson  = {pr_va:+.4f}")

    print(f"\n--- val: predicted vs actual time-to-debut ---")
    va["pred"] = pred_va
    print(f"{'actual_h':>9} {'n':>5} {'mean_pred':>10} "
          f"{'med_pred':>9} {'MAE':>6}")
    for h in range(1, args.max_h + 1):
        s = va[va.time_to_debut == h]
        if len(s) < 10: continue
        mae = (s["pred"] - h).abs().mean()
        print(f"{h:>9d} {len(s):>5d} {s['pred'].mean():>9.2f} "
              f"{s['pred'].median():>8.2f} {mae:>5.2f}")

    print(f"\n--- val: by snap_offset ---")
    print(f"{'snap_off':>8} {'n':>5} {'spearman':>10} "
          f"{'mean_actual':>12} {'mean_pred':>10} {'MAE':>6}")
    for so in sorted(va.snap_offset.unique()):
        s = va[va.snap_offset == so]
        if len(s) < 30: continue
        sp = spearmanr(s["pred"], s["time_to_debut"]).correlation
        mae_so = (s["pred"] - s["time_to_debut"]).abs().mean()
        print(f"{so:>8d} {len(s):>5d} {sp:>+9.3f} "
              f"{s.time_to_debut.mean():>11.2f} {s['pred'].mean():>9.2f} "
              f"{mae_so:>5.2f}")

    with open(args.out, "wb") as fh:
        pickle.dump({
            "scaler": scaler, "lasso": lasso,
            "feature_names": list(feat),
            "alpha": float(lasso.alpha_),
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "mae_train": float(mae_tr), "mae_val": float(mae_va),
            "spearman_val": float(sp_va), "pearson_val": float(pr_va),
            "include_p_debut": bool(args.include_p_debut),
            "max_h": args.max_h,
            "age_center": AGE_CENTER, "yip_center": YIP_CENTER,
            "kind": "time_to_debut_regression_v1.18b_landmark",
            "uses_landmark_timing": bool(landmark_feat),
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
