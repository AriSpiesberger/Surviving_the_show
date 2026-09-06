"""Walk-forward A/B: recent-cohort resolved short-h labels in joint training.

Production trains the joint layer on entry<=2020 only, discarding 2021-23
entries whose h=1..3 outcomes ARE resolved — recent-era signal at exactly the
buy-list horizons. The random-split val cannot measure the benefit (it holds
only <=2020 entries); the walk-forward can: at origin Y, does ALSO training
on entry (Y, Y+6) rows resolved within Y+6 improve scoring of the snap-Y+6
cohort? (Those players' own earlier snaps enter training labeled only through
Y+6 — exactly what a production refresh would have.)

Reuses exp_walkforward2's cached hazards + longs per origin; protocol
otherwise identical (G recipe, NROUNDS fixed, 2-fold crossfit calibrator).
Baseline = exp_walkforward2's result.json.

    python -m prospects.model.train.exp_walkforward3
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.joint_xgb import _assemble
from prospects.model.train.exp_cdf_timing2 import FEAT2, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows, train_one
from prospects.model.train.exp_walkforward2 import (
    DB, G3_SLOW, GAP, EVAL_H, NROUNDS, _metrics, bucket_rows,
)

WF2 = REPO_ROOT / "runs" / "exp_walkforward2"
OUT_DIR = REPO_ROOT / "runs" / "exp_walkforward3"


def run_origin(Y, keep_raw, feats, t0):
    snap = Y + GAP
    odir = WF2 / f"Y{Y}"
    print(f"\n===== origin Y={Y} AUGMENTED [{(time.time()-t0)/60:.0f}m] =====")

    fit_base = prep_base(pd.read_csv(odir / "fit_long.csv"), DB, max_entry=Y)
    aug_base = prep_base(pd.read_csv(odir / "eval_long.csv"), DB)
    # augmentation: the recent cohorts' snaps BEFORE the scoring date; their
    # (row,h) pairs are kept by _assemble only where years_fwd >= h, i.e.
    # resolved within Y+GAP. The snap==Y+GAP rows (years_fwd=0) contribute
    # nothing to training and are the eval set.
    aug_train = aug_base[aug_base.snap_year < snap].copy()
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
        col = f"eligible_{ev}"
        for df_ in (fit_base, aug_train):
            if col in df_.columns:
                df_.drop(df_[df_[col] != 1].index, inplace=True)
    both = pd.concat([fit_base, aug_train], ignore_index=True)
    both = attach_raw_features(both, DB, keep_raw, verbose=False)
    fit_long, Y_fit = _assemble(both, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    n_aug = int((fit_long["entry_year"] > Y).sum())
    print(f"  joint fit rows: {len(fit_long):,} "
          f"({n_aug:,} from recent cohorts) [{(time.time()-t0)/60:.0f}m]")
    X = fit_long[feats].values.astype(np.float32)

    fpids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    snap_yr = fit_long["snap_year"].to_numpy()
    uniq = np.unique(fpids)
    rng = np.random.default_rng(7)
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    hm = np.isin(fpids, list(half))
    k = EVENTS.index("MLB_DEBUT")

    oofp = np.full((len(fit_long), len(EVENTS)), np.nan)
    for name, tr, ho in (("A", ~hm, hm), ("B", hm, ~hm)):
        b = train_one(X[tr], Y_fit[tr], feats, G3_SLOW, NROUNDS, 0, 42)
        oofp[ho] = predict_rows(b, X[ho], feats)
        del b
        print(f"  cross-fit {name} done [{(time.time()-t0)/60:.0f}m]")
    ok = np.isfinite(oofp[:, k]) & (snap_yr >= 2008)
    cal = HYip2Calibrator().fit(oofp[ok, k], h_arr[ok], yip_arr[ok],
                                Y_fit[ok, k].astype(int))
    bst = train_one(X, Y_fit, feats, G3_SLOW, NROUNDS, 0, 42)
    del X

    ev_base = prep_base(pd.read_csv(odir / "eval_long.csv"), DB)
    ev_base = ev_base[(ev_base.snap_year == snap)
                      & (ev_base.get("eligible_MLB_DEBUT", 1) == 1)].copy()
    ev_base = attach_raw_features(ev_base, DB, keep_raw, verbose=False)
    P_h = {}
    for h in (1, 2, 3):
        sub = stamp_extra_cols(add_cond_cols(ev_base, h))
        P_h[h] = predict_rows(bst, sub[feats].values.astype(np.float32),
                              feats)[:, k]
    del bst
    raw3 = np.maximum.accumulate(
        np.column_stack([P_h[1], P_h[2], P_h[3]]), axis=1)[:, 2]
    cal3 = cal.predict(raw3, np.full(len(ev_base), 3),
                       ev_base["snap_offset"].to_numpy())

    trig = pd.to_numeric(ev_base["trigger_MLB_DEBUT"], errors="coerce")
    y = ((trig > snap) & (trig <= snap + EVAL_H)).fillna(False)
    y = y.to_numpy().astype(int)
    print(f"  eval: n={len(y):,} pos={int(y.sum()):,} base={y.mean():.1%}")
    rows = [_metrics(raw3, y, "AUG joint raw"),
            _metrics(cal3, y, "AUG joint calibrated")]
    base_json = json.loads((odir / "result.json").read_text())
    base_cal = next(s for s in base_json["scorers"]
                    if s["scorer"] == "joint calibrated")
    print(f"  {'baseline joint cal':<22} AP={base_cal['ap']:.4f}  "
          f"AUC={base_cal['auc']:.4f}  calib={base_cal['calib']:.2f}")
    return {"Y": Y, "n": int(len(y)), "base": float(y.mean()),
            "aug": rows, "baseline_cal": base_cal,
            "n_aug_rows": n_aug, "buckets_cal": bucket_rows(cal3, y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", nargs="*", type=int, default=[2016, 2014])
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / "runs" / "current" / "models"
              / "joint_xgb_v2.3.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    feats = list(FEAT2) + keep_raw

    results = [run_origin(Y, keep_raw, feats, t0) for Y in args.origins]

    print(f"\n===== recent-cohort augmentation A/B (debut<= {EVAL_H}y, "
          f"walk-forward) =====")
    print(f"{'Y':>5} | {'baseline cal AP':>16} | {'AUG cal AP':>11} | "
          f"{'delta':>7} | {'calib b->a':>11}")
    for r in results:
        b = r["baseline_cal"]
        a = next(s for s in r["aug"] if "calibrated" in s["scorer"])
        print(f"{r['Y']:>5} | {b['ap']:>16.4f} | {a['ap']:>11.4f} | "
              f"{a['ap']-b['ap']:>+7.4f} | "
              f"{b['calib']:>5.2f}->{a['calib']:.2f}")
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"results": results,
         "elapsed_min": round((time.time() - t0) / 60, 1)}, indent=2,
        default=float))
    print(f"\nwrote {OUT_DIR}  ({(time.time()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
