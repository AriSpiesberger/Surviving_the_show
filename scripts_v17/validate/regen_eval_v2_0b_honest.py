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

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.joint_cond import (  # noqa: E402
    EVENTS, H_MAX, PUBLISH_H, predict_trajectory, prep_base, realized_by_h,
)

EVENT_WEIGHTS = {"TOP_100_PROSPECT": 1.0, "MLB_DEBUT": 2.0,
                 "ESTABLISHED_MLB": 1.0, "STAR_PLUS_ELITE": 1.0}

VAL_LONG = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
XGB_PKL = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"
DB = REPO_ROOT / "prospects_snapshot.db"
OUT_DIR = REPO_ROOT / "evaluation" / "v2.0b_landmark"


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
                 group_val, threshold: float = 0.5, horizon: int = PUBLISH_H) -> dict:
    # Conditional model: score = refined cumulative P(event by snap+h); label =
    # realized_by_h (event fired within h years). The caller restricts `sub` to
    # rows resolved at this horizon (years_fwd >= h) so negatives are trustworthy.
    p_col = f"xp_{event}_h{horizon}"
    y_col = f"rby_{event}"
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
    # Calibration: Brier + mean-predicted vs observed (calib_ratio ~ 1.0 = well
    # calibrated; >1 over-predicts, <1 under-predicts).
    brier = float(brier_score_loss(y, p)) if n else float("nan")
    mean_pred = float(p.mean()) if n else float("nan")
    calib_ratio = (mean_pred / base) if base > 0 else float("nan")
    return {
        "event": event,
        group_name: group_val,
        "horizon": horizon,
        "n": n, "pos": pos, "base_rate": base,
        "auc": auc, "ap": ap, "ap_lift": ap_lift,
        "brier": brier, "mean_pred": mean_pred, "calib_ratio": calib_ratio,
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


def _add_rby(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Stamp the cumulative-by-horizon labels rby_<event> = realized within
    `horizon` years of the snap, for the resolved evaluation slice."""
    df = df.copy()
    for ev in EVENTS:
        df[f"rby_{ev}"] = realized_by_h(df, ev, horizon)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="Score cutoff for the confusion-matrix view. "
                         "0.60 is the production buy-list cutoff.")
    ap.add_argument("--val-long", default=str(VAL_LONG))
    ap.add_argument("--eval-horizon", type=int, default=PUBLISH_H,
                    help="Horizon h for the headline/stratified tables: score "
                         "xp_<event>_h{h} vs realized-within-h, on rows resolved "
                         "at h (years_fwd >= h). Default = publish horizon (6).")
    args = ap.parse_args()
    H = args.eval_horizon

    print(f"Loading {Path(args.val_long).name}...")
    df = pd.read_csv(args.val_long)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} pids")

    df = prep_base(df, str(DB), max_entry=args.max_entry)
    print(f"  after entry<={args.max_entry}: {len(df):,} rows")

    print(f"Scoring conditional trajectory with {XGB_PKL.name}...")
    df = predict_trajectory(pickle.load(open(XGB_PKL, "rb")), df)

    print(f"Joining current level...")
    df = _join_current_level(df, str(DB))
    df["bucket"] = df.apply(_bucket_of, axis=1)

    # Headline / stratified tables: evaluate at the publish horizon on the
    # RESOLVED slice (>= H forward years), so negatives are trustworthy.
    resolved = df[df["years_fwd"] >= H].copy()
    resolved = _add_rby(resolved, H)
    print(f"\nEval horizon h={H}: {len(resolved):,} resolved rows "
          f"(>= {H} fwd yrs) of {len(df):,}")

    # ---- per-bucket ----
    print(f"\n=== per-bucket (h={H}) ===")
    bucket_rows = []
    for ev in EVENTS:
        for b in ["ALL", "R1", "R2-R3", "R4-R10", "R10+", "IFA"]:
            sub = resolved if b == "ALL" else resolved[resolved["bucket"] == b]
            row = _metric_row(sub, ev, "bucket", b, args.threshold, H)
            if row:
                bucket_rows.append(row)
    _write_table(bucket_rows, OUT_DIR / "per_bucket_validation.csv")

    # ---- per-yip / snap_offset ----
    print(f"\n=== per-yip (h={H}) ===")
    yip_rows = []
    for ev in EVENTS:
        for off in sorted(resolved["snap_offset"].unique()):
            sub = resolved[resolved["snap_offset"] == int(off)]
            row = _metric_row(sub, ev, "snap_offset", int(off),
                              args.threshold, H)
            if row:
                yip_rows.append(row)
    _write_table(yip_rows, OUT_DIR / "per_yip_validation.csv")

    # ---- per-level ----
    print(f"\n=== per-level (h={H}) ===")
    level_rows = []
    for ev in EVENTS:
        for lvl in ["ALL", "RK", "A-", "A", "A+", "AA", "AAA", "NONE"]:
            sub = resolved if lvl == "ALL" else resolved[resolved["cur_level"] == lvl]
            row = _metric_row(sub, ev, "cur_level", lvl, args.threshold, H)
            if row:
                level_rows.append(row)
    _write_table(level_rows, OUT_DIR / "per_level_validation.csv")

    # ---- walkforward.csv (event × snap_offset long) ----
    _write_table(yip_rows, OUT_DIR / "walkforward.csv")

    # ---- per_horizon.csv: the trajectory-quality curve. For each h, evaluate
    # xp_<event>_h{h} vs realized-within-h on the rows resolved at that h. ----
    print(f"\n=== per-horizon trajectory (h=1..{H_MAX}) ===")
    horizon_rows = []
    for h in range(1, H_MAX + 1):
        sub_h = _add_rby(df[df["years_fwd"] >= h].copy(), h)
        for ev in EVENTS:
            row = _metric_row(sub_h, ev, "horizon_eval", h, args.threshold, h)
            if row:
                horizon_rows.append(row)
    _write_table(horizon_rows, OUT_DIR / "per_horizon.csv")

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
        "eval_horizon": H,
        "weighted_ap": wap,
        "per_event": overall,
    }
    (OUT_DIR / "headline.json").write_text(
        json.dumps(headline, indent=2))
    print(f"\nWeighted-AP @ h={H}: {wap:.4f}")
    for r in overall:
        print(f"  {r['event']:<22} AP={r['ap']:.3f}  lift={r['ap_lift']:.1f}x  "
              f"AUC={r['auc']:.3f}")


if __name__ == "__main__":
    main()
