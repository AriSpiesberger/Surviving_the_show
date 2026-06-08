"""Regenerate the FULL v2.0b validation packet honestly.

Recovers every table that was in the leaky v2.0b_landmark packet, but
produced from honest inputs:
  - val_long scored by hazards_full (val pids held out)
  - XGB applied = joint_xgb_v2.0b_oof_tuned.pkl

Tables produced (in evaluation/v2.0b_landmark/):
  bucket.csv                            per (event × bucket) @ snap_offset=2
  walkforward.csv                       per (event × snap_offset) detailed
  per_bucket_validation.csv             per (event × bucket) cutoff-view
  per_yip_validation.csv                per (event × snap_offset) cutoff-view
  per_level_validation.csv              per (event × cur_level) cutoff-view
  headline.json                         summary
  report.txt                            human-readable summary
  <EVENT>_walkforward.csv               per-event detailed walkforward
  <EVENT>_pct_slabs.csv                 per (event × yip × slab) slab analysis
  <EVENT>_cum_above_threshold.csv       per (event × yip × pctile_cut)
  MLB_DEBUT_per_current_level.csv       per current level breakdown
  MLB_DEBUT_thresholds_at_p60.csv       per yip threshold + precision/recall
  MLB_DEBUT_time_to_debut.csv           per (player, snap) timing actuals vs preds
  walkforward_2021entry_by_year/snap{YYYY}.csv   2021-entry cohort by snap year

Usage:
    python -m scripts_v17.validate.regen_full_eval_v2_0b
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
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts_v17.validate.regen_eval_v2_0b_honest import (
    EVENTS, EVENT_WEIGHTS, AGE_CENTER, YIP_CENTER, HAZARD_PROBS,
    _prep_for_xgb, _score_xgb, _join_current_level, _metric_row,
    _bucket_of,
)

VAL_LONG = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
XGB_PKL = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof_tuned.pkl"
TIMING_PKL = REPO_ROOT / "models" / "time_to_debut_v1.18_prod.pkl"
DB = REPO_ROOT / "prospects_snapshot.db"
OUT_DIR = REPO_ROOT / "evaluation" / "v2.0b_landmark"

# Confidence intervals on AUC via DeLong-ish bootstrap-ish — simpler version
def _auc_ci(y, p, n_boot=200, seed=42) -> tuple[float, float, float]:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]; ps = p[idx]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        aucs.append(roc_auc_score(ys, ps))
    a = float(roc_auc_score(y, p))
    return a, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _ece(y, p, n_bins=10) -> float:
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    binned = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = binned == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def _spiegelhalter(y, p) -> float:
    """Hosmer-Lemeshow-like p-value via z-score; returns p-value."""
    if len(y) == 0 or p.std() == 0:
        return float("nan")
    z_num = (y - p).sum()
    z_den = np.sqrt(((1 - 2 * p) ** 2 * (y * (1 - p) + (1 - y) * p)).sum())
    if z_den == 0:
        return float("nan")
    z = z_num / z_den
    from scipy.stats import norm
    return float(2 * (1 - norm.cdf(abs(z))))


def _lift_at_k(y, p, k_pct: float):
    """Lift among the top-k% by predicted prob."""
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), 0
    k = max(1, int(round(n * k_pct / 100.0)))
    order = np.argsort(-p)
    top = order[:k]
    base = float(y.mean()) if y.mean() > 0 else float("nan")
    if not (base == base) or base == 0:
        return float("nan"), float("nan"), int(k)
    rate = float(y[top].mean())
    recall = float(y[top].sum() / y.sum()) if y.sum() else float("nan")
    return rate / base, recall, int(k)


def _detailed_metrics(sub: pd.DataFrame, event: str,
                       label_group_col: str, group_val) -> dict:
    """Returns the rich metric row used in walkforward.csv (mean_fwd_years,
    pred_mean, auc + ci, ap, ap_lift, brier, brier_skill, ece,
    spiegelhalter_p, lift@K, recall@K, k@K for K in {1,5,10})."""
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
    auc, auc_lo, auc_hi = _auc_ci(y, p)
    ap = float(average_precision_score(y, p)) if pos else float("nan")
    ap_lift = ap / base if base > 0 and ap == ap else float("nan")
    brier = float(brier_score_loss(y, p)) if pos else float("nan")
    brier_skill = (
        1 - brier / (base * (1 - base))
        if base > 0 and brier == brier else float("nan")
    )
    ece = _ece(y, p)
    sph = _spiegelhalter(y, p)
    mean_fwd = float(sub.get("years_fwd", pd.Series([np.nan])).mean())
    pred_mean = float(p.mean())
    out = {
        "event": event, label_group_col: group_val,
        "mean_fwd_years": mean_fwd,
        "n": n, "pos": pos, "base_rate": base, "pred_mean": pred_mean,
        "auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "ap": ap, "ap_lift": ap_lift,
        "brier": brier, "brier_skill": brier_skill,
        "ece": ece, "spiegelhalter_p": sph,
    }
    for k_pct in (1, 5, 10):
        lift, recall, k = _lift_at_k(y, p, k_pct)
        out[f"lift@{k_pct}%"] = lift
        out[f"recall@{k_pct}%"] = recall
        out[f"k@{k_pct}%"] = k
    return out


def _build_pct_slabs(df: pd.DataFrame, event: str) -> pd.DataFrame:
    """Per (event, yip, percentile-slab) slabs."""
    p_col = f"xp_{event}"
    y_col = f"realized_{event}"
    elig_col = f"eligible_{event}"
    slabs = [
        ("0.0-0.5%", 0.0, 0.5),
        ("0.5-1.0%", 0.5, 1.0),
        ("1.0-1.5%", 1.0, 1.5),
        ("1.5-2.0%", 1.5, 2.0),
        ("2-3%", 2.0, 3.0),
        ("3-4%", 3.0, 4.0),
        ("4-5%", 4.0, 5.0),
        ("5-10%", 5.0, 10.0),
        ("10-20%", 10.0, 20.0),
        ("20-50%", 20.0, 50.0),
        ("bottom 50%", 50.0, 100.0),
    ]
    rows = []
    for yip in sorted(df["snap_offset"].unique()):
        sub = df[df["snap_offset"] == int(yip)]
        if elig_col in sub.columns:
            sub = sub[sub[elig_col] == 1]
        if len(sub) == 0:
            continue
        sub = sub.copy()
        p = sub[p_col].astype(float).values
        y = sub[y_col].astype(int).values
        base = y.mean() if len(y) else float("nan")
        if base == 0 or len(p) == 0:
            continue
        # Percentile is "top X% by predicted prob" (so lower pct = higher score)
        scores_sorted = np.sort(p)[::-1]
        n = len(p)
        for label, lo, hi in slabs:
            i_lo = max(0, int(np.floor(n * lo / 100.0)))
            i_hi = max(i_lo, int(np.ceil(n * hi / 100.0)))
            if i_lo >= n:
                continue
            i_hi = min(i_hi, n)
            score_hi = scores_sorted[i_lo] if i_lo < n else float("nan")
            score_lo = scores_sorted[i_hi - 1] if i_hi - 1 < n else float("nan")
            mask = (p >= score_lo) & (p <= score_hi)
            if mask.sum() == 0:
                continue
            rows.append({
                "event": event, "yip": int(yip),
                "slab": label, "slab_lo": lo, "slab_hi": hi,
                "n": int(mask.sum()),
                "tp": int(y[mask].sum()),
                "rate": float(y[mask].mean()),
                "base_rate": float(base),
                "lift": float(y[mask].mean() / base) if base > 0 else float("nan"),
                "score_lo": float(score_lo), "score_hi": float(score_hi),
                "score_mean": float(p[mask].mean()),
            })
    return pd.DataFrame(rows)


def _build_cum_above_threshold(df: pd.DataFrame, event: str) -> pd.DataFrame:
    """Per (event, yip, pctile_cut), cumulative rate ABOVE the cut."""
    p_col = f"xp_{event}"
    y_col = f"realized_{event}"
    elig_col = f"eligible_{event}"
    pctile_cuts = list(range(1, 21)) + [25, 30, 40, 50, 60, 75, 100]
    rows = []
    for yip in sorted(df["snap_offset"].unique()):
        sub = df[df["snap_offset"] == int(yip)]
        if elig_col in sub.columns:
            sub = sub[sub[elig_col] == 1]
        if len(sub) == 0:
            continue
        sub = sub.copy()
        p = sub[p_col].astype(float).values
        y = sub[y_col].astype(int).values
        n = len(p)
        base = float(y.mean()) if n else float("nan")
        if base == 0 or n == 0:
            continue
        order = np.argsort(-p)
        for cut in pctile_cuts:
            k = max(1, int(round(n * cut / 100.0)))
            top = order[:k]
            score_floor = float(p[top].min())
            rate_above = float(y[top].mean())
            lift = rate_above / base if base > 0 else float("nan")
            rows.append({
                "event": event, "yip": int(yip), "pctile_cut": cut,
                "score_floor": score_floor,
                "n_above": int(k), "rate_above": rate_above,
                "base_rate": base, "lift": lift,
                "tp_above": int(y[top].sum()),
            })
    return pd.DataFrame(rows)


def _build_thresholds_at_p60(df: pd.DataFrame) -> pd.DataFrame:
    """For each yip, find the min threshold yielding precision ≥ 0.60."""
    rows = []
    p_col = "xp_MLB_DEBUT"
    y_col = "realized_MLB_DEBUT"
    elig_col = "eligible_MLB_DEBUT"
    for yip in sorted(df["snap_offset"].unique()):
        sub = df[df["snap_offset"] == int(yip)]
        if elig_col in sub.columns:
            sub = sub[sub[elig_col] == 1]
        if len(sub) == 0:
            continue
        p = sub[p_col].astype(float).values
        y = sub[y_col].astype(int).values
        n_total = len(y)
        n_pos_total = int(y.sum())
        # Walk down thresholds, find min thr where precision >= 0.6
        order = np.argsort(-p)
        cumulative_tp = 0
        chosen_thr = None; chosen_n = 0; chosen_tp = 0
        for i, idx in enumerate(order, start=1):
            cumulative_tp += int(y[idx])
            precision = cumulative_tp / i
            if precision >= 0.60:
                chosen_thr = float(p[idx])
                chosen_n = i
                chosen_tp = cumulative_tp
            elif chosen_thr is not None:
                break
        if chosen_thr is None:
            continue
        recall = chosen_tp / n_pos_total if n_pos_total else float("nan")
        precision = chosen_tp / chosen_n if chosen_n else float("nan")
        rows.append({
            "yip": int(yip), "threshold": chosen_thr,
            "n_above": int(chosen_n), "tp_above": int(chosen_tp),
            "precision": precision, "recall": recall,
            "n_total": int(n_total), "n_pos_total": int(n_pos_total),
        })
    return pd.DataFrame(rows)


def _build_time_to_debut(df: pd.DataFrame) -> pd.DataFrame:
    """Per (player, snap, MLB_DEBUT triggered): actual_h vs t_pred (time
    to debut)."""
    if "trigger_MLB_DEBUT" not in df.columns:
        return pd.DataFrame()
    sub = df[df["trigger_MLB_DEBUT"].notna()].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    sub["mlb_debut_year"] = sub["trigger_MLB_DEBUT"].astype(int)
    sub["actual_h"] = sub["mlb_debut_year"] - sub["snap_year"]
    if "mean_t_MLB_DEBUT" not in sub.columns:
        return pd.DataFrame()
    sub["t_pred"] = sub["mean_t_MLB_DEBUT"]
    cols = ["player_id", "snap_year", "snap_offset",
            "mlb_debut_year", "actual_h", "t_pred"]
    return sub[cols].sort_values(["player_id", "snap_year"])


def _build_per_current_level(df: pd.DataFrame, event: str) -> pd.DataFrame:
    """Per (event × current_level) detailed metrics."""
    rows = []
    for lvl in ["ALL", "RK", "A-", "A", "A+", "AA", "AAA", "NONE"]:
        sub = df if lvl == "ALL" else df[df["cur_level"] == lvl]
        m = _detailed_metrics(sub, event, "cur_level", lvl)
        if m:
            rows.append(m)
    return pd.DataFrame(rows)


def _build_walkforward_2021_entry(df: pd.DataFrame, snap_year: int
                                    ) -> pd.DataFrame:
    """For 2021-entry players, per-event metrics at this specific snap year."""
    sub = df[(df["entry_year"] == 2021) & (df["snap_year"] == snap_year)]
    rows = []
    for ev in EVENTS:
        m = _detailed_metrics(sub, ev, "event", ev)
        if m:
            m_clean = dict(m); del m_clean["event"]
            m_clean["event"] = ev
            m_clean["snap_year"] = snap_year
            rows.append(m_clean)
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, path: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"  {path.relative_to(OUT_DIR)}: {len(df)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entry", type=int, default=2020)
    args = ap.parse_args()

    print(f"Loading {VAL_LONG.name}...")
    df = pd.read_csv(VAL_LONG)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} pids")
    df = _prep_for_xgb(df, str(DB), args.max_entry)
    print(f"Scoring with {XGB_PKL.name}...")
    df = _score_xgb(df, XGB_PKL)
    df = _join_current_level(df, str(DB))
    df["bucket"] = df.apply(_bucket_of, axis=1)
    print(f"  prepared: {len(df):,} rows\n")

    # --- bucket.csv: per (event × bucket) at snap_offset=2 ---
    print("=== bucket.csv (snap_offset=2) ===")
    sub2 = df[df["snap_offset"] == 2]
    rows = []
    for ev in EVENTS:
        for b in ["ALL", "R1", "R2-R3", "R4-R10", "R10+", "IFA"]:
            ss = sub2 if b == "ALL" else sub2[sub2["bucket"] == b]
            m = _detailed_metrics(ss, ev, "bucket", b)
            if m:
                rows.append(m)
    _write(pd.DataFrame(rows), OUT_DIR / "bucket.csv")

    # --- walkforward.csv: per (event × snap_offset) detailed ---
    print("\n=== walkforward.csv ===")
    rows = []
    for ev in EVENTS:
        for off in sorted(df["snap_offset"].unique()):
            ss = df[df["snap_offset"] == int(off)]
            m = _detailed_metrics(ss, ev, "snap_offset", int(off))
            if m:
                rows.append(m)
    _write(pd.DataFrame(rows), OUT_DIR / "walkforward.csv")

    # --- per-event walkforward + pct_slabs + cum_above_threshold ---
    print("\n=== per-event detailed tables ===")
    for ev in EVENTS:
        ev_wf = pd.DataFrame([
            _detailed_metrics(df[df["snap_offset"] == int(off)], ev,
                              "snap_offset", int(off))
            for off in sorted(df["snap_offset"].unique())
            if _detailed_metrics(df[df["snap_offset"] == int(off)], ev,
                                 "snap_offset", int(off))
        ])
        _write(ev_wf, OUT_DIR / f"{ev}_walkforward.csv")
        _write(_build_pct_slabs(df, ev), OUT_DIR / f"{ev}_pct_slabs.csv")
        _write(_build_cum_above_threshold(df, ev),
               OUT_DIR / f"{ev}_cum_above_threshold.csv")

    # --- MLB_DEBUT specific tables ---
    print("\n=== MLB_DEBUT specific ===")
    _write(_build_per_current_level(df, "MLB_DEBUT"),
           OUT_DIR / "MLB_DEBUT_per_current_level.csv")
    _write(_build_thresholds_at_p60(df),
           OUT_DIR / "MLB_DEBUT_thresholds_at_p60.csv")
    ttd = _build_time_to_debut(df)
    if len(ttd):
        _write(ttd, OUT_DIR / "MLB_DEBUT_time_to_debut.csv")

    # --- walkforward_2021entry_by_year/ ---
    print("\n=== walkforward_2021entry_by_year/ ===")
    for sy in range(2021, 2027):
        w = _build_walkforward_2021_entry(df, sy)
        if len(w):
            _write(w,
                   OUT_DIR / "walkforward_2021entry_by_year"
                   / f"snap{sy}.csv")

    # --- already-present compact tables (preserve) ---
    print("\n=== compact tables (already up-to-date) ===")
    print(f"  per_bucket_validation.csv: regenerated by "
          f"regen_eval_v2_0b_honest.py (already in repo)")
    print(f"  per_yip_validation.csv: ditto")
    print(f"  per_level_validation.csv: ditto")
    print(f"  headline.json: ditto")

    print("\nDONE.")


if __name__ == "__main__":
    main()
