"""Calibrate the tuned v2.0b prod stack and unify val + prod inference.

The Optuna-tuned hazards (max_depth=4, leaves=15, l2=4.2) produce sharper
ranking (higher AP) but more compressed probabilities — the tail of high-
confidence cases is squeezed inward. The fixed 0.60 buy threshold was
calibrated against the OLD distribution, so on the TUNED stack only 18
picks survive vs ~118 on the untuned.

Fix: fit isotonic regression per event on the TUNED-stack val outputs so
calibrated P(event) actually means "this fraction of similar players hit
that event". Then BOTH validation tables and the production buy list
use the same calibrated probabilities.

Steps:
  1. Score val with TUNED hazards (already have val_long_tuned_hazards.csv
     from eval_tuned_v2_0b.py) -> apply tuned XGB -> raw val probs.
  2. Per event: IsotonicRegression(raw_p, realized_y) -> calibrator.
  3. Save calibrators alongside the tuned XGB bundle ->
     models/joint_xgb_v2.0b_prod.pkl (the canonical production bundle).
  4. Apply the same XGB+calibrator stack to snap=2026 ->
     results/scored/snap2026_v2.0b_prod_long.csv (calibrated probs).
  5. Regenerate the eval tables off the calibrated val long.

After this, P(MLB_DEBUT) >= 0.60 in production means precision ~ 0.60 in
val terms — the threshold-at-p60 cells stay meaningful.
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]

VAL_LONG_TUNED = (REPO_ROOT / "scratch" / "v20b_oof"
                    / "val_long_tuned_hazards.csv")
SNAP_LONG_PROD = (REPO_ROOT / "results" / "scored"
                    / "snap2026_v2.0b_tuned_prod_long.csv")

XGB_PKL = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof_tuned.pkl"
PROD_BUNDLE_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_prod.pkl"

CALIB_VAL_LONG = (REPO_ROOT / "results" / "training"
                    / "v2.0b_calibrated_val_long.csv")
CALIB_SNAP_LONG = (REPO_ROOT / "results" / "scored"
                    / "snap2026_v2.0b_prod_long.csv")

DB = REPO_ROOT / "prospects_snapshot.db"

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT",
          "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
HAZARD_PROBS = [
    "p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB",
    "p_ELITE", "p_STAR", "p_STAR_PLUS_ELITE",
]
AGE_CENTER, YIP_CENTER = 22, 3


def _prep_for_xgb(df: pd.DataFrame, db: str, max_entry: int = 2020,
                   filter_entry: bool = True) -> pd.DataFrame:
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
    if filter_entry:
        df = df[df.entry_year <= max_entry].copy()
    return df


def _apply_xgb_raw(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    feat = bundle["feature_names"]
    scaler = bundle["scaler"]
    booster = bundle["model"]
    best_iter = bundle.get("best_iteration")
    X = scaler.transform(df[feat].values.astype(np.float32))
    d = xgb.DMatrix(X, feature_names=list(feat))
    if best_iter is not None:
        P = booster.predict(d, iteration_range=(0, best_iter + 1))
    else:
        P = booster.predict(d)
    return P  # shape (n, n_events)


def _apply_calibrators(P_raw: np.ndarray,
                        calibrators: dict, events: list) -> np.ndarray:
    P_cal = np.empty_like(P_raw)
    for k, ev in enumerate(events):
        cal = calibrators[ev]
        P_cal[:, k] = np.clip(cal.predict(P_raw[:, k]), 0.0, 1.0)
    return P_cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entry", type=int, default=2020)
    args = ap.parse_args()

    # ---- 1. Load tuned XGB bundle ----
    print(f"Loading tuned XGB bundle: {XGB_PKL.name}")
    with XGB_PKL.open("rb") as fh:
        bundle = pickle.load(fh)
    bundle_events = list(bundle["events"])
    print(f"  events: {bundle_events}")

    # ---- 2. Score val with tuned XGB raw ----
    print(f"\nLoading val long: {VAL_LONG_TUNED.name}")
    val = pd.read_csv(VAL_LONG_TUNED)
    val = _prep_for_xgb(val, str(DB), args.max_entry)
    print(f"  val: {len(val):,} rows, {val.player_id.nunique():,} pids")
    needed = bundle["feature_names"]
    val = val.dropna(subset=needed).copy()
    print(f"  after dropna: {len(val):,} rows")
    P_val_raw = _apply_xgb_raw(val, bundle)

    # ---- 3. Fit isotonic calibrator per event ----
    print(f"\nFitting isotonic calibrators per event...")
    calibrators = {}
    for k, ev in enumerate(bundle_events):
        y_col = f"realized_{ev}"
        elig_col = f"eligible_{ev}"
        sub_mask = np.ones(len(val), dtype=bool)
        if elig_col in val.columns:
            sub_mask &= val[elig_col].values == 1
        y = val[y_col].astype(int).values[sub_mask]
        p_raw = P_val_raw[sub_mask, k]
        if y.sum() < 10 or y.sum() == len(y):
            print(f"  {ev}: too few positives ({int(y.sum())}/{len(y)}) "
                  f"— identity calibrator")
            calibrators[ev] = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            )
            calibrators[ev].fit([0.0, 1.0], [0.0, 1.0])
            continue
        cal = IsotonicRegression(out_of_bounds="clip",
                                  y_min=0.0, y_max=1.0)
        cal.fit(p_raw, y)
        calibrators[ev] = cal
        # Diagnostic: show calibration curve points
        bins = np.linspace(0, 1, 11)
        binned = np.digitize(p_raw, bins[1:-1])
        diag_lines = []
        for b in range(10):
            m = binned == b
            if m.sum() == 0:
                continue
            raw_m = p_raw[m].mean()
            real_m = y[m].mean()
            cal_m = cal.predict(np.array([raw_m]))[0]
            diag_lines.append(
                f"    bin {bins[b]:.1f}-{bins[b+1]:.1f}: "
                f"n={int(m.sum()):>5d}  raw={raw_m:.3f}  "
                f"realized={real_m:.3f}  calibrated={cal_m:.3f}")
        print(f"  {ev}:")
        for ln in diag_lines:
            print(ln)

    # ---- 4. Save the calibrated PROD bundle ----
    calibrated_bundle = dict(bundle)
    calibrated_bundle["calibrators"] = calibrators
    calibrated_bundle["calibration_kind"] = "isotonic_per_event_on_val"
    calibrated_bundle["calibration_n_val"] = int(len(val))
    calibrated_bundle["calibration_source"] = str(VAL_LONG_TUNED)
    calibrated_bundle["version"] = "v2.0b_prod_calibrated"
    with PROD_BUNDLE_OUT.open("wb") as fh:
        pickle.dump(calibrated_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved calibrated prod bundle -> {PROD_BUNDLE_OUT}")

    # ---- 5. Write val long WITH calibrated XGB outputs ----
    P_val_cal = _apply_calibrators(P_val_raw, calibrators, bundle_events)
    out_val = val.copy()
    for k, ev in enumerate(bundle_events):
        out_val[f"xp_raw_{ev}"] = P_val_raw[:, k]
        out_val[f"xp_{ev}"] = P_val_cal[:, k]
    out_val.to_csv(CALIB_VAL_LONG, index=False)
    print(f"Wrote calibrated val long -> {CALIB_VAL_LONG.name} "
          f"({len(out_val):,} rows)")

    # ---- 6. Write snap=2026 long with calibrated XGB outputs ----
    # Load original snap long (NOT pre-processed) so downstream buy-list
    # builder can do its own _add_feats merge cleanly.
    print(f"\nLoading prod snap long: {SNAP_LONG_PROD.name}")
    snap_orig = pd.read_csv(SNAP_LONG_PROD)
    snap = _prep_for_xgb(snap_orig.copy(), str(DB),
                          args.max_entry, filter_entry=False)
    snap = snap.dropna(subset=needed).copy()
    P_snap_raw = _apply_xgb_raw(snap, bundle)
    P_snap_cal = _apply_calibrators(P_snap_raw, calibrators, bundle_events)
    # Build output: original snap_long columns + raw/calibrated XGB probs.
    # Overwrite p_<event> with calibrated values so build_v2.0_buylist.py
    # picks them up without code changes.
    snap_keyed = snap[["player_id", "snap_year"]].copy()
    for k, ev in enumerate(bundle_events):
        snap_keyed[f"xp_raw_{ev}"] = P_snap_raw[:, k]
        snap_keyed[f"xp_{ev}"] = P_snap_cal[:, k]
        snap_keyed[f"p_{ev}_cal"] = P_snap_cal[:, k]
    out_snap = snap_orig.merge(snap_keyed,
                                 on=["player_id", "snap_year"], how="left")
    for ev in bundle_events:
        out_snap[f"p_{ev}"] = out_snap[f"p_{ev}_cal"].fillna(
            out_snap[f"p_{ev}"])
        out_snap = out_snap.drop(columns=[f"p_{ev}_cal"])
    out_snap.to_csv(CALIB_SNAP_LONG, index=False)
    print(f"Wrote calibrated snap long -> {CALIB_SNAP_LONG.name} "
          f"({len(out_snap):,} rows, cols={len(out_snap.columns)})")

    # ---- 7. Compare distributions before/after calibration ----
    print(f"\n=== Calibration impact on MLB_DEBUT P distribution ===")
    raw_col = "xp_raw_MLB_DEBUT"; cal_col = "xp_MLB_DEBUT"
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        n_raw_val = int((out_val[raw_col] >= thr).sum())
        n_cal_val = int((out_val[cal_col] >= thr).sum())
        n_raw_snap = int((out_snap[raw_col] >= thr).sum())
        n_cal_snap = int((out_snap[cal_col] >= thr).sum())
        print(f"  P(MLB)>={thr:.2f}  VAL raw={n_raw_val:>5d} "
              f"cal={n_cal_val:>5d}   SNAP raw={n_raw_snap:>5d} "
              f"cal={n_cal_snap:>5d}")
    print(f"\nDONE.")


if __name__ == "__main__":
    main()
