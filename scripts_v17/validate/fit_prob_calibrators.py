"""Fit per-(event, horizon) isotonic probability calibrators on the held-out
OOF val, so the buy list can emit calibrated probabilities instead of raw
conditional-XGB scores.

For each event and horizon h in 1..H_MAX we fit IsotonicRegression(
xp_<event>_h{h} -> realized-within-h) on the RESOLVED + eligible val slice
(years_fwd >= h), the same slice regen_eval scores. The result is a monotone
map raw_score -> calibrated P(event by snap+h). Events/horizons with too few
positives fall back to identity (no calibrator).

Output: models/prob_calibrators_v2.0b.pkl
  {"calibrators": {(event, h): IsotonicRegression}, "events": [...],
   "h_max": int, "fit_val": str}

Usage:
    python -m scripts_v17.validate.fit_prob_calibrators
    python -m scripts_v17.validate.fit_prob_calibrators \
        --val-long results/training/v2.0b_partial_oof_val_long.csv \
        --xgb models/joint_xgb_v2.0b_partial_oof.pkl \
        --out models/prob_calibrators_v2.0b_partial.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.prob_calibration import LogitCalibrator  # noqa: E402

from prospects.classifier.joint_cond import (  # noqa: E402
    EVENTS, H_MAX, predict_trajectory, prep_base, realized_by_h,
)

VAL_LONG = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
XGB_PKL = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"
DB = str(REPO_ROOT / "prospects_snapshot.db")
OUT = REPO_ROOT / "models" / "prob_calibrators_v2.0b.pkl"
MIN_POS = 25  # below this, calibration is noise -> fall back to identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-long", default=str(VAL_LONG))
    ap.add_argument("--xgb", default=str(XGB_PKL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--method", choices=["logistic", "isotonic"],
                    default="logistic",
                    help="logistic = smooth Platt-on-logit (default; preserves "
                         "ranking resolution). isotonic = step function (collapses "
                         "scores onto plateaus — bad for thresholding/ranking).")
    args = ap.parse_args()

    print(f"Loading {Path(args.val_long).name} + {Path(args.xgb).name}")
    val = prep_base(pd.read_csv(args.val_long), DB, max_entry=args.max_entry)
    val = predict_trajectory(pickle.load(open(args.xgb, "rb")), val)

    cals: dict = {}
    print(f"{'event':<22}{'h':>3}{'n':>8}{'pos':>7}  calibrator")
    for ev in EVENTS:
        for h in range(1, H_MAX + 1):
            col = f"xp_{ev}_h{h}"
            if col not in val.columns:
                continue
            d = val[(val["years_fwd"] >= h)
                    & (val.get(f"eligible_{ev}", 1) == 1)].copy()
            d["y"] = realized_by_h(d, ev, h).astype(float)
            d = d.dropna(subset=[col])
            n = len(d)
            pos = int(d["y"].sum())
            if pos < MIN_POS or pos == n:
                continue  # identity fallback (no entry)
            x = d[col].astype(float).values
            if args.method == "logistic":
                c = LogitCalibrator().fit(x, d["y"].values)
            else:
                c = IsotonicRegression(out_of_bounds="clip").fit(x, d["y"].values)
            cals[(ev, h)] = c
            print(f"{ev:<22}{h:>3}{n:>8,}{pos:>7,}  fit ({args.method})")

    bundle = {"calibrators": cals, "events": list(EVENTS),
              "h_max": H_MAX, "fit_val": str(args.val_long),
              "method": args.method}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\nwrote {args.out}: {len(cals)} (event,h) calibrators")


if __name__ == "__main__":
    main()
