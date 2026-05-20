"""Model B: P(outcome | debut) trained on early-MLB performance labels.

Outcome label (nested, highest tier wins):
  breakout: any season (wOBA>=.350 & PA>=300) or (ERA<=3.50 & IP>=100)
  regular:  any season PA>=350 or IP>=80
  utility:  >=2 MLB seasons, max season PA<350 and IP<80
  cup:      career PA<100 and no MLB year > debut+1

Training set: fit + val slices combined (held out of hazard training).
Debut window: 2010-2024 (exclude 2025-2026 partials).
Features: raw hazards at earliest pre-debut snap + scouting covariates.
Model:    multinomial logistic, L2, GroupKFold(5) CV by player_id.
"""
from __future__ import annotations

import pickle
import sqlite3
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DB = "prospects_snapshot.db"
FIT = "v1.17_fit_long.csv"
VAL = "v1.17_val_long.csv"
DEBUT_MIN, DEBUT_MAX = 2010, 2024
OUT_MODEL = "models/model_b_outcomes_v1.17.pkl"


def load_mlb_stats():
    c = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT player_id, season_year, pa, ip, woba, era, primary_position "
        "FROM season_stats WHERE UPPER(level)='MLB'", c)
    c.close()
    df["pa"] = df["pa"].fillna(0)
    df["ip"] = df["ip"].fillna(0)
    return df


def label_player(rows: pd.DataFrame, debut_year: int) -> str:
    post = rows[rows.season_year >= debut_year].copy()
    if post.empty:
        return "cup"
    has_breakout = (
        ((post.woba >= 0.350) & (post.pa >= 300)).any()
        or ((post.era <= 3.50) & (post.ip >= 100)).any()
    )
    if has_breakout:
        return "breakout"
    has_regular = ((post.pa >= 350) | (post.ip >= 80)).any()
    if has_regular:
        return "regular"
    n_seasons = post.season_year.nunique()
    if n_seasons >= 2:
        return "utility"
    career_pa = post.pa.sum()
    came_back = (post.season_year > debut_year + 1).any()
    if career_pa >= 100 or came_back:
        return "utility"
    return "cup"


def main():
    print("Loading scored fit+val slices...")
    fit = pd.read_csv(FIT)
    val = pd.read_csv(VAL)
    df = pd.concat([fit, val], ignore_index=True)
    print(f"  combined: {len(df):,} rows, {df.player_id.nunique():,} players")

    df = df[df.mlb_debut_year.notna()].copy()
    df["mlb_debut_year"] = df["mlb_debut_year"].astype(int)
    df = df[(df.mlb_debut_year >= DEBUT_MIN) & (df.mlb_debut_year <= DEBUT_MAX)]
    print(f"  debutees {DEBUT_MIN}-{DEBUT_MAX}: {df.player_id.nunique():,}")

    df = df[df.snap_year < df.mlb_debut_year].copy()
    df = df.sort_values(["player_id", "snap_year"]).groupby("player_id").first().reset_index()
    print(f"  earliest pre-debut snap per player: {len(df):,} rows")

    print("\nLoading MLB stats for labels...")
    mlb = load_mlb_stats()
    print(f"  {len(mlb):,} MLB rows, {mlb.player_id.nunique():,} players")

    labels = {}
    for pid, debut in zip(df.player_id, df.mlb_debut_year):
        labels[pid] = label_player(mlb[mlb.player_id == pid], debut)
    df["outcome"] = df["player_id"].map(labels)

    print("\nClass distribution:")
    for c, n in Counter(df.outcome).most_common():
        print(f"  {c:<10s} {n:>4d}  {n/len(df):.1%}")

    pos_lk = pd.read_csv("models/player_position_from_stats.csv")
    df = df.merge(pos_lk, on="player_id", how="left")
    df["position"] = df["pos_seasonstats"].fillna("UNK")
    print(f"  position from season_stats for {df.pos_seasonstats.notna().sum()}/{len(df)} players")

    def pos_group(p):
        p = str(p).upper()
        if p == "C": return "C"
        if p in ("1B", "2B", "3B", "SS"): return "IF"
        if p in ("LF", "CF", "RF", "OF", "DH"): return "OF"
        return "OTH"
    df["pos_grp"] = df["position"].apply(pos_group)
    print("  pos_grp distribution:")
    for g, n in df.pos_grp.value_counts().items():
        print(f"    {g}: {n}")

    haz_cols = ["p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB", "p_STAR_PLUS_ELITE"]
    eps = 1e-6
    for c in haz_cols:
        df[f"logit_{c}"] = np.log((df[c].clip(eps, 1 - eps)) / (1 - df[c].clip(eps, 1 - eps)))

    bucket_dum = pd.get_dummies(df["bucket"], prefix="b", drop_first=True)
    pos_dum = pd.get_dummies(df["pos_grp"], prefix="pos", drop_first=True)

    feat = pd.concat([
        df[[f"logit_{c}" for c in haz_cols]].reset_index(drop=True),
        df[["snap_offset"]].reset_index(drop=True).rename(columns={"snap_offset": "yrs_pre_debut"}),
        bucket_dum.reset_index(drop=True),
        pos_dum.reset_index(drop=True),
    ], axis=1).astype(float)
    feature_names = list(feat.columns)
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")

    X = feat.values
    y_str = df["outcome"].values
    classes = ["cup", "utility", "regular", "breakout"]
    y = np.array([classes.index(c) for c in y_str])
    groups = df["player_id"].values

    print("\n--- GroupKFold(5) CV ---")
    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros((len(y), len(classes)))
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(
            penalty="l2", C=1.0,
            solver="lbfgs", max_iter=2000,
        ).fit(Xtr, y[tr])
        oof_proba[te] = clf.predict_proba(Xte)
        ll = -np.log(oof_proba[te][np.arange(len(te)), y[te]] + 1e-9).mean()
        print(f"  fold {fold}: n_tr={len(tr)} n_te={len(te)} log-loss={ll:.4f}")

    overall_ll = -np.log(oof_proba[np.arange(len(y)), y] + 1e-9).mean()
    print(f"\nOOF multinomial log-loss: {overall_ll:.4f}")

    base_rates = np.array([np.mean(y == k) for k in range(len(classes))])
    base_ll = -np.log(base_rates[y] + 1e-9).mean()
    print(f"Baseline (class priors) log-loss: {base_ll:.4f}")
    print(f"Improvement: {(base_ll - overall_ll):.4f} nats/sample")

    print("\nPer-class OOF calibration (decile bins):")
    for k, name in enumerate(classes):
        p = oof_proba[:, k]
        obs = (y == k).astype(int)
        order = np.argsort(p)
        n = len(p)
        print(f"\n  {name} (base rate {obs.mean():.1%}):")
        print(f"    {'decile':<8} {'n':>4} {'pred':>8} {'obs':>8}")
        for d in range(10):
            lo = d * n // 10
            hi = (d + 1) * n // 10
            idx = order[lo:hi]
            pred = p[idx].mean()
            o = obs[idx].mean()
            print(f"    {d:<8d} {len(idx):>4d} {pred:>8.1%} {o:>8.1%}")

    print("\nFitting final model on full debutee set...")
    sc_full = StandardScaler().fit(X)
    clf_full = LogisticRegression(
        penalty="l2", C=1.0,
        solver="lbfgs", max_iter=2000,
    ).fit(sc_full.transform(X), y)

    print("\nFinal model coefficients (each row = one outcome class):")
    print(f"  {'':22s}", " ".join(f"{c:>10s}" for c in classes))
    for j, fn in enumerate(feature_names):
        print(f"  {fn:22s}", " ".join(f"{clf_full.coef_[k][j]:>10.3f}" for k in range(len(classes))))

    with open(OUT_MODEL, "wb") as fh:
        pickle.dump({
            "model": clf_full,
            "scaler": sc_full,
            "feature_names": feature_names,
            "classes": classes,
            "oof_log_loss": overall_ll,
            "baseline_log_loss": base_ll,
        }, fh)
    print(f"\nSaved {OUT_MODEL}")


if __name__ == "__main__":
    main()
