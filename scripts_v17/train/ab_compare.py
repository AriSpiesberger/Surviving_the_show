"""A/B: scouting-grades ON vs OFF, measured at BOTH stages (hazard + joint XGB)
and BOTH slices (full val + entry>=2013), per event and weighted.

Reads:
  grades ON  : *_gradeson.csv  + joint_xgb_v2.0b_oof_gradeson.pkl
  grades OFF : v2.0b_oof_val_long.csv + joint_xgb_v2.0b_oof.pkl  (default paths)

Usage:  python -m scripts_v17.train.ab_compare
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from scripts_v17.train.tune_joint_xgb_v2_oof import EVENTS, EVENT_WEIGHTS, FEAT, _prep

REPO = Path(__file__).resolve().parents[2]
DB = str(REPO / "prospects_snapshot.db")
TR = REPO / "results" / "training"
MD = REPO / "models"
VARIANTS = {
    "grades_OFF": (TR / "v2.0b_oof_val_long.csv", MD / "joint_xgb_v2.0b_oof.pkl"),
    "grades_ON":  (TR / "v2.0b_oof_val_long_gradeson.csv",
                   MD / "joint_xgb_v2.0b_oof_gradeson.pkl"),
}


def _wap(df, pcol):
    """per-event AP + weighted-AP using p column formatter pcol(ev)."""
    out, wap, wt = {}, 0.0, 0.0
    for ev in EVENTS:
        sub = df[df.get(f"eligible_{ev}", 0) == 1]
        if sub.empty:
            continue
        y = sub[f"realized_{ev}"].astype(int).values
        p = sub[pcol(ev)].astype(float).values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        out[ev] = float(average_precision_score(y, p))
        wap += EVENT_WEIGHTS[ev] * out[ev]
        wt += EVENT_WEIGHTS[ev]
    out["WEIGHTED"] = wap / wt if wt else float("nan")
    return out


def hazard_scored(val_csv):
    return pd.read_csv(val_csv)  # p_<ev> already = hazard outputs


def xgb_scored(val_csv, pkl):
    art = pickle.load(open(pkl, "rb"))
    df = _prep(pd.read_csv(val_csv), DB, 2020).reset_index(drop=True)
    X = art["scaler"].transform(df[FEAT].values.astype(np.float32))
    P = art["model"].predict(xgb.DMatrix(X, feature_names=FEAT),
                             iteration_range=(0, art["best_iteration"] + 1))
    for k, ev in enumerate(EVENTS):
        df[f"xp_{ev}"] = P[:, k]
    return df


def main():
    results = {}  # (variant, stage, slice) -> dict
    for v, (val_csv, pkl) in VARIANTS.items():
        if not val_csv.exists() or not pkl.exists():
            print(f"[skip] {v}: missing {val_csv.name} or {pkl.name}")
            continue
        hz = hazard_scored(val_csv)
        hz = hz[hz.entry_year <= 2020]
        xg = xgb_scored(val_csv, pkl)
        for slc, mask_hz, mask_xg in [
            ("full", hz, xg),
            ("2013+", hz[hz.entry_year >= 2013], xg[xg.entry_year >= 2013])]:
            results[(v, "hazard", slc)] = _wap(mask_hz, lambda e: f"p_{e}")
            results[(v, "xgb", slc)] = _wap(mask_xg, lambda e: f"xp_{e}")

    cols = EVENTS + ["WEIGHTED"]
    for stage in ("hazard", "xgb"):
        for slc in ("full", "2013+"):
            print(f"\n=== {stage.upper()} stage | {slc} val ===")
            print(f"{'event':<20}{'OFF':>9}{'ON':>9}{'Δ':>9}")
            off = results.get(("grades_OFF", stage, slc), {})
            on = results.get(("grades_ON", stage, slc), {})
            for ev in cols:
                a, b = off.get(ev), on.get(ev)
                if a is None and b is None:
                    continue
                d = (b - a) if (a is not None and b is not None) else float("nan")
                tag = "  <<<" if ev == "WEIGHTED" else ""
                print(f"{ev:<20}{(a or float('nan')):>9.3f}{(b or float('nan')):>9.3f}"
                      f"{d:>+9.3f}{tag}")


if __name__ == "__main__":
    main()
