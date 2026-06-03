"""v1.18: per-event L1-logistic ("lasso-logit") replacing the single
time-decayed lasso composite.

For each of the four reported events
  TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB, STAR_PLUS_ELITE
fit a separate L1-penalized LogisticRegression on its binary realized
target, using the same 10 features the v1.17 debut lasso used:

  p_TOP_100_PROSPECT, p_MLB_DEBUT, p_ESTABLISHED_MLB, p_STAR_PLUS_ELITE,
  age_at_snap_centered, years_in_pro,
  p_*_x_yip_centered  (4 interactions)

Each row is filtered to eligible_<event>=1 and entry_year <= 2020.
Group-aware CV (GroupKFold by player_id) picks regularization strength.

Output: a single pkl with {event: {scaler, lasso, feature_names, alpha,
n_train, base_rate, val_auc}} for all four events.

Usage:
    python -m scripts_v17.train.fit_lasso_logits_v18 \\
        --fit v1.17h_fit_long.csv \\
        --val v1.17h_val_long.csv \\
        --db prospects_snapshot.db \\
        --out models/lasso_logits_v1.18h.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EVENTS = [
    "TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
    "ELITE", "STAR", "STAR_PLUS_ELITE",
]
HAZARD_PROBS = [
    "p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB",
    "p_ELITE", "p_STAR", "p_STAR_PLUS_ELITE",
]
FEAT = (
    HAZARD_PROBS
    + ["age_at_snap_centered", "years_in_pro"]
    + [f"{p}_x_yip_centered" for p in HAZARD_PROBS]
)
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
    for ev in EVENTS:
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    return df


def fit_one(df_fit: pd.DataFrame, df_val: pd.DataFrame, event: str,
            max_entry: int = 2020, seed: int = 42) -> dict:
    """Fit L1-logistic for one event, return artifact dict."""
    cols_needed = (FEAT + [f"realized_{event}", f"eligible_{event}",
                           "entry_year", "player_id"])
    tr = df_fit[df_fit[f"eligible_{event}"] == 1].copy()
    tr = tr[tr.entry_year <= max_entry]
    tr = tr.dropna(subset=cols_needed)
    n_pos = int(tr[f"realized_{event}"].sum())
    n = len(tr)
    print(f"  [{event}]  train: n={n:,}  pos={n_pos:,} "
          f"({n_pos/max(n,1):.2%} base)")
    if n_pos < 20 or n_pos == n:
        print(f"    SKIP — too few positives or saturated")
        return None

    X = tr[FEAT].values.astype(float)
    y = tr[f"realized_{event}"].values.astype(int)
    g = tr["player_id"].values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(Xs, y, g))
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 30),
        cv=splits,
        penalty="l1",
        solver="saga",
        scoring="neg_log_loss",
        max_iter=5000,
        n_jobs=-1,
        refit=True,
        random_state=seed,
    ).fit(Xs, y)
    C_chosen = float(clf.C_[0])
    print(f"    chose C={C_chosen:.4g}  (alpha={1/C_chosen:.4g})")

    # Print non-zero coefficients
    print(f"    non-zero coefficients (logit):")
    for name, coef in zip(FEAT, clf.coef_.ravel()):
        if abs(coef) > 1e-6:
            print(f"      {name:<42} {coef:+.4f}")
    print(f"      {'intercept':<42} {clf.intercept_[0]:+.4f}")

    # Val AUC for sanity
    val_auc = float("nan")
    if df_val is not None:
        v = df_val[df_val[f"eligible_{event}"] == 1].copy()
        v = v[v.entry_year <= max_entry].dropna(subset=cols_needed)
        if len(v) > 50 and v[f"realized_{event}"].sum() >= 5:
            Xv = scaler.transform(v[FEAT].values.astype(float))
            yv = v[f"realized_{event}"].values.astype(int)
            try:
                val_auc = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
                print(f"    honest val AUC: {val_auc:.3f}  "
                      f"(n={len(v):,}, pos={int(yv.sum())})")
            except Exception:
                pass

    return {
        "scaler": scaler,
        "lasso": clf,
        "feature_names": list(FEAT),
        "C": C_chosen,
        "alpha": 1.0 / C_chosen,
        "n_train": n,
        "n_pos_train": n_pos,
        "base_rate_train": n_pos / n,
        "val_auc": val_auc,
        "age_center": AGE_CENTER,
        "yip_center": YIP_CENTER,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="v1.17h_fit_long.csv")
    ap.add_argument("--val", default="v1.17h_val_long.csv",
                    help="Used only for AUC sanity print, not for fitting.")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading fit slice {args.fit}")
    fit = pd.read_csv(args.fit)
    print(f"  fit: {len(fit):,} rows, {fit.player_id.nunique():,} players")
    if args.val:
        print(f"Loading val slice {args.val}")
        val = pd.read_csv(args.val)
        print(f"  val: {len(val):,} rows, {val.player_id.nunique():,} players")
    else:
        val = None

    fit = add_feats(fit, args.db)
    if val is not None:
        val = add_feats(val, args.db)

    artifacts: dict[str, dict] = {}
    for ev in EVENTS:
        print(f"\n[{ev}]")
        result = fit_one(fit, val, ev, max_entry=args.max_entry)
        if result is not None:
            artifacts[ev] = result

    bundle = {
        "events": EVENTS,
        "per_event": artifacts,
        "feature_names": list(FEAT),
        "age_center": AGE_CENTER,
        "yip_center": YIP_CENTER,
        "version": "v1.18",
        "kind": "lasso_logit_per_event",
        "note": ("Per-event L1-logistic on realized_<event> with "
                 "eligible_<event>=1 + entry_year<=max_entry. Score is "
                 "the model's logit (decision_function); calibrated prob "
                 "is predict_proba(X)[:, 1]."),
    }
    with open(args.out, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")
    print(f"  events: {list(artifacts.keys())}")
    for ev, a in artifacts.items():
        print(f"  {ev:<22} val_AUC={a['val_auc']:.3f}  "
              f"n_train={a['n_train']:,}  base={a['base_rate_train']:.2%}")


if __name__ == "__main__":
    main()
