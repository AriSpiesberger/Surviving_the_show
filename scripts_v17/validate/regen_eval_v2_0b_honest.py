"""Regenerate the v2.0b validation tables using HONEST inputs.

Inputs:
  val_long: results/training/v2.0b_oof_val_long.csv
            (val pids scored by hazards_full trained on the 90% universe
             excluding val — honest at hazard layer)
  XGB:      models/joint_xgb_v2.0b_oof.pkl
            (trained on OOF-stacked rows where each row came from a
             leave-one-out hazards model — honest at XGB layer)

For each event in {TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB,
STAR_PLUS_ELITE} we build:

  per_bucket_validation.csv      one row per (bucket × event)
  per_yip_validation.csv         one row per (snap_offset × event)
  per_level_validation.csv       one row per (current_level × event)
  walkforward.csv                long format, per (event, snap_offset)
  headline.json                  weighted-AP + per-event APs at threshold 0.5

All cells emit: n, pos, base_rate, auc, ap, ap_lift, plus threshold-0.5
precision/recall/F1/accuracy + tp/fp/tn/fn for the buy-list cutoff view.

Usage:
    python -m scripts_v17.validate.regen_eval_v2_0b_honest
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT",
          "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
EVENT_WEIGHTS = {"TOP_100_PROSPECT": 1.0, "MLB_DEBUT": 2.0,
                 "ESTABLISHED_MLB": 1.0, "STAR_PLUS_ELITE": 1.0}

VAL_LONG = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
XGB_PKL = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"
DB = REPO_ROOT / "prospects_snapshot.db"
OUT_DIR = REPO_ROOT / "evaluation" / "v2.0b_landmark"

AGE_CENTER, YIP_CENTER = 22, 3
HAZARD_PROBS = [
    "p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB",
    "p_ELITE", "p_STAR", "p_STAR_PLUS_ELITE",
]


def _prep_for_xgb(df: pd.DataFrame, db: str, max_entry: int):
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
    for p in HAZARD_PROBS:
        df[f"{p}_x_yip_centered"] = df[p] * df["yip_centered"]
    from prospects.features.scouting_grades import attach_scouting_summary
    df = attach_scouting_summary(df)  # point-in-time scouting summary cols
    df = df[df.entry_year <= max_entry].copy()
    return df


def _score_xgb(df: pd.DataFrame, xgb_pkl: Path) -> pd.DataFrame:
    with open(xgb_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    feat = bundle["feature_names"]
    scaler = bundle["scaler"]
    booster = bundle["model"]
    best_iter = bundle.get("best_iteration")
    calibrators = bundle.get("calibrators")  # optional per-event isotonic
    needed = list(feat)
    sub = df.dropna(subset=needed).copy()
    X = scaler.transform(sub[feat].values.astype(np.float32))
    d = xgb.DMatrix(X, feature_names=list(feat))
    if best_iter is not None:
        P = booster.predict(d, iteration_range=(0, best_iter + 1))
    else:
        P = booster.predict(d)
    for k, ev in enumerate(bundle["events"]):
        raw = P[:, k]
        if calibrators is not None and ev in calibrators:
            cal = np.clip(calibrators[ev].predict(raw), 0.0, 1.0)
            sub[f"xp_{ev}"] = cal
            sub[f"xp_raw_{ev}"] = raw
        else:
            sub[f"xp_{ev}"] = raw
    return sub


def _join_current_level(df: pd.DataFrame, db: str) -> pd.DataFrame:
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
    df["cur_level"] = df["cur_rank"].map(labels).fillna("NONE")
    return df.drop(columns=["season_year", "cur_rank"], errors="ignore")


def _metric_row(sub: pd.DataFrame, event: str, group_name: str,
                 group_val, threshold: float = 0.5) -> dict:
    p_col = f"xp_{event}"
    y_col = f"realized_{event}"
    elig_col = f"eligible_{event}"
    if p_col not in sub.columns or y_col not in sub.columns:
        return {}
    if elig_col in sub.columns:
        sub = sub[sub[elig_col] == 1]
    n = len(sub)
    if n == 0:
        return {}
    y = sub[y_col].astype(int).values
    p = sub[p_col].astype(float).values
    pos = int(y.sum())
    base = float(y.mean())
    auc = (float(roc_auc_score(y, p))
           if 0 < pos < n else float("nan"))
    ap = float(average_precision_score(y, p)) if pos > 0 else float("nan")
    ap_lift = (ap / base if base > 0 else float("nan")) if ap == ap \
               else float("nan")
    # Spearman rank correlation between scores and outcomes
    if 0 < pos < n and p.std() > 0:
        rho, rho_p = spearmanr(p, y)
        spearman_rho = float(rho)
        spearman_p = float(rho_p)
    else:
        spearman_rho = float("nan"); spearman_p = float("nan")
    # Threshold metrics at user-specified cutoff
    thr = float(threshold)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan")) \
         if precision == precision and recall == recall else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")
    return {
        "event": event,
        group_name: group_val,
        "n": n, "pos": pos, "base_rate": base,
        "auc": auc, "ap": ap, "ap_lift": ap_lift,
        "spearman_rho": spearman_rho, "spearman_p": spearman_p,
        "threshold": thr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "predicted_positives": int(pred.sum()),
    }


def _bucket_of(row) -> str:
    if int(row.get("is_international") or 0) == 1:
        return "IFA"
    rnd = row.get("draft_round")
    try:
        rnd = int(rnd)
    except Exception:
        return "UNKNOWN"
    if rnd == 1:
        return "R1"
    if 2 <= rnd <= 3:
        return "R2-R3"
    if 4 <= rnd <= 10:
        return "R4-R10"
    return "R10+"


def _write_table(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  wrote {path.name}: {len(df)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="Score cutoff for the confusion-matrix view. "
                         "0.60 is the production buy-list cutoff.")
    args = ap.parse_args()

    print(f"Loading {VAL_LONG.name}...")
    df = pd.read_csv(VAL_LONG)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} pids")

    df = _prep_for_xgb(df, str(DB), args.max_entry)
    print(f"  after entry<={args.max_entry}: {len(df):,} rows")

    print(f"Scoring with {XGB_PKL.name}...")
    df = _score_xgb(df, XGB_PKL)

    print(f"Joining current level...")
    df = _join_current_level(df, str(DB))

    # Add bucket
    df["bucket"] = df.apply(_bucket_of, axis=1)

    # ---- per-bucket ----
    print(f"\n=== per-bucket ===")
    bucket_rows = []
    for ev in EVENTS:
        for b in ["ALL", "R1", "R2-R3", "R4-R10", "R10+", "IFA"]:
            sub = df if b == "ALL" else df[df["bucket"] == b]
            row = _metric_row(sub, ev, "bucket", b, args.threshold)
            if row:
                bucket_rows.append(row)
    _write_table(bucket_rows, OUT_DIR / "per_bucket_validation.csv")

    # ---- per-yip / snap_offset ----
    print(f"\n=== per-yip ===")
    yip_rows = []
    for ev in EVENTS:
        for off in sorted(df["snap_offset"].unique()):
            sub = df[df["snap_offset"] == int(off)]
            row = _metric_row(sub, ev, "snap_offset", int(off),
                              args.threshold)
            if row:
                yip_rows.append(row)
    _write_table(yip_rows, OUT_DIR / "per_yip_validation.csv")

    # ---- per-level ----
    print(f"\n=== per-level ===")
    level_rows = []
    for ev in EVENTS:
        for lvl in ["ALL", "RK", "A-", "A", "A+", "AA", "AAA", "NONE"]:
            sub = df if lvl == "ALL" else df[df["cur_level"] == lvl]
            row = _metric_row(sub, ev, "cur_level", lvl, args.threshold)
            if row:
                level_rows.append(row)
    _write_table(level_rows, OUT_DIR / "per_level_validation.csv")

    # ---- walkforward.csv (event × snap_offset long) ----
    _write_table(yip_rows, OUT_DIR / "walkforward.csv")

    # ---- headline.json ----
    weighted_ap = 0.0
    total_w = 0.0
    overall = []
    for ev in EVENTS:
        r = next((b for b in bucket_rows
                   if b["event"] == ev and b["bucket"] == "ALL"), None)
        if r and r["ap"] == r["ap"]:
            w = EVENT_WEIGHTS[ev]
            weighted_ap += w * r["ap"]
            total_w += w
            overall.append({"event": ev, "ap": r["ap"],
                              "auc": r["auc"], "ap_lift": r["ap_lift"]})
    wap = weighted_ap / total_w if total_w else 0.0
    headline = {
        "model": str(XGB_PKL),
        "val_long": str(VAL_LONG),
        "weighted_ap": wap,
        "per_event": overall,
    }
    (OUT_DIR / "headline.json").write_text(
        json.dumps(headline, indent=2))
    print(f"\nWeighted-AP: {wap:.4f}")
    for r in overall:
        print(f"  {r['event']:<22} AP={r['ap']:.3f}  lift={r['ap_lift']:.1f}x  "
              f"AUC={r['auc']:.3f}")


if __name__ == "__main__":
    main()
