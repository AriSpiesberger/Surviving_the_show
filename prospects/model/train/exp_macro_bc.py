"""Macro bake-off B and C on the clean walk-forward (2026-09-19).

Question (the user's): is the two-stage GBM stack the right macro at all?
Every prior experiment stayed inside it. Two cheap tests of its two biggest
assumptions, measured OUT OF ERA on exp_walkforward2's per-origin caches
(rebuilt with today's leak-free features), production-like (recent-cohort
augmentation as in exp_walkforward3):

  A   control: the production recipe (hazard stack -> joint GBM), same seeds.
  B   single stage: the joint GBM with NO hazard-derived inputs — every raw
      landmark feature (325) plus age / years-in-pro / scouting summary /
      acquisition tier. If B ties A, the hazard stack is dead weight.
  C1  era-adaptive fit: A's features, training rows recency-weighted
      (half-life --half-life years on snap_year).
  C2  era-adaptive level: A's calibrated output plus an intercept re-estimated
      from the most recent snaps whose 3y outcome had resolved by the scoring
      date (snaps Y+1..Y+3, cross-fit predictions) — an online base-rate fix.
  C3  C1 + C2.

Per origin Y in {2016, 2014, 2012}: fit on entry<=Y plus resolved recent rows,
2-fold player cross-fit for the calibrator, refit on 100%, score the
entry-(Y, Y+6] cohort at snap Y+6, label = debut within 3 more years.
Primary metric: AP at h=3 (the sheet's thesis), then AUC and calibration ratio.

    python -m prospects.model.train.exp_macro_bc
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar

from prospects.config import REPO_ROOT
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.exp_cdf_timing import _logit
from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, FEAT2, _mono_string, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows
from prospects.model.train.exp_walkforward2 import (
    DB, EVAL_H, G3_SLOW, GAP, NROUNDS, _metrics, bucket_rows,
)
from prospects.model.train.joint_xgb import _assemble

WF2 = REPO_ROOT / "runs" / "exp_walkforward2"
OUT_DIR = REPO_ROOT / "runs" / "exp_macro_bc"
HAZARD_TOKENS = ("p_", "hk", "haz_cum", "mean_t", "sd_t", "h_minus_mean_t", "z_h_debut",
                 "_x_yip_centered")
K = EVENTS.index("MLB_DEBUT")


def train_w(X, Y, feats, rounds, seed, weight=None):
    params = dict(BASE_PARAMS)
    params.update(G3_SLOW)
    params["seed"] = seed
    params["monotone_constraints"] = _mono_string(feats)
    d = xgb.QuantileDMatrix(X, label=Y, weight=weight, feature_names=feats)
    return xgb.train(params, d, num_boost_round=rounds, verbose_eval=False)


def feature_sets(keep_raw):
    a = list(FEAT2) + keep_raw
    b = [f for f in FEAT2
         if not (any(f.startswith(t) for t in HAZARD_TOKENS) or f.endswith("_x_yip_centered"))]
    b = b + [f"rw_{n}" for n in FEATURE_NAMES]
    return {"A": a, "B": b}


def load_origin(Y, keep_raw_all):
    """Production-like training frame (fit + resolved recent rows) and the eval frame."""
    snap = Y + GAP
    odir = WF2 / f"Y{Y}"
    fit_base = prep_base(pd.read_csv(odir / "fit_long.csv"), DB, max_entry=Y)
    aug_base = prep_base(pd.read_csv(odir / "eval_long.csv"), DB)
    aug_train = aug_base[aug_base.snap_year < snap].copy()
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
        col = f"eligible_{ev}"
        for df_ in (fit_base, aug_train):
            if col in df_.columns:
                df_.drop(df_[df_[col] != 1].index, inplace=True)
    both = pd.concat([fit_base, aug_train], ignore_index=True)
    both = attach_raw_features(both, DB, keep_raw_all, verbose=False)
    fit_long, Y_fit = _assemble(both, H_MAX)
    fit_long = stamp_extra_cols(fit_long)

    ev_base = prep_base(pd.read_csv(odir / "eval_long.csv"), DB)
    ev_base = ev_base[(ev_base.snap_year == snap)
                      & (ev_base.get("eligible_MLB_DEBUT", 1) == 1)].copy()
    ev_base = attach_raw_features(ev_base, DB, keep_raw_all, verbose=False)
    trig = pd.to_numeric(ev_base["trigger_MLB_DEBUT"], errors="coerce")
    y = ((trig > snap) & (trig <= snap + EVAL_H)).fillna(False).to_numpy().astype(int)
    return fit_long, Y_fit, ev_base, y


def run_variant(name, feats, fit_long, Y_fit, ev_base, y, Y, weight, era_fix, t0, log):
    snap = Y + GAP
    X = fit_long[feats].values.astype(np.float32)
    fpids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    snap_yr = fit_long["snap_year"].to_numpy()
    uniq = np.unique(fpids)
    rng = np.random.default_rng(7)
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    hm = np.isin(fpids, list(half))

    oofp = np.full(len(fit_long), np.nan)
    for tr, ho in ((~hm, hm), (hm, ~hm)):
        b = train_w(X[tr], Y_fit[tr], feats, NROUNDS, 42, None if weight is None else weight[tr])
        oofp[ho] = predict_rows(b, X[ho], feats)[:, K]
        del b
    ok = np.isfinite(oofp) & (snap_yr >= 2008)
    cal = HYip2Calibrator().fit(oofp[ok], h_arr[ok], yip_arr[ok], Y_fit[ok, K].astype(int))

    # era intercept: recent resolved snaps (Y+1..Y+3 at h=3), calibrated cross-fit preds
    offset = 0.0
    if era_fix:
        m = ok & (snap_yr >= Y + 1) & (snap_yr <= snap - EVAL_H) & (h_arr == EVAL_H)
        if m.sum() >= 200 and 0 < Y_fit[m, K].sum() < m.sum():
            pc = cal.predict(oofp[m], h_arr[m], yip_arr[m])
            lp, yy = _logit(pc), Y_fit[m, K].astype(float)
            # intercept-only shift: c minimising log loss of sigmoid(logit(p) + c)
            nll = lambda c: -np.mean(yy * np.log(1 / (1 + np.exp(-(lp + c))) + 1e-12)
                                     + (1 - yy) * np.log(1 - 1 / (1 + np.exp(-(lp + c))) + 1e-12))
            offset = float(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)
            log(f"    era offset from {int(m.sum()):,} recent resolved rows: {offset:+.3f} logit")

    bst = train_w(X, Y_fit, feats, NROUNDS, 42, weight)
    del X
    P_h = {}
    for h in (1, 2, 3):
        sub = stamp_extra_cols(add_cond_cols(ev_base, h))
        P_h[h] = predict_rows(bst, sub[feats].values.astype(np.float32), feats)[:, K]
    del bst
    raw3 = np.maximum.accumulate(np.column_stack([P_h[1], P_h[2], P_h[3]]), axis=1)[:, 2]
    cal3 = cal.predict(raw3, np.full(len(ev_base), 3), ev_base["snap_offset"].to_numpy())
    if era_fix:
        cal3 = 1.0 / (1.0 + np.exp(-(_logit(cal3) + offset)))
    log(f"  [{name}] done [{(time.time() - t0) / 60:.0f}m]")
    np.savez_compressed(OUT_DIR / f"preds_Y{Y}_{name}.npz", cal3=cal3, raw3=raw3, y=y,
                        pid=ev_base["player_id"].to_numpy(),
                        yip=ev_base["snap_offset"].to_numpy())
    return {"variant": name, **_metrics(cal3, y, f"{name} calibrated"),
            "raw": _metrics(raw3, y, f"{name} raw"), "buckets": bucket_rows(cal3, y),
            "era_offset": offset, "n_feats": len(feats)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origins", nargs="*", type=int, default=[2016, 2014, 2012])
    ap.add_argument("--half-life", type=float, default=4.0)
    ap.add_argument("--variants", nargs="*", default=["A", "B", "C1", "C2", "C3"])
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)

    with open(REPO_ROOT / "runs" / "current" / "models" / "joint_xgb_v2.3.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in keep_raw if c in live]
    fs = feature_sets(keep_raw)
    keep_raw_all = sorted(set(keep_raw) | set(fs["B"]) & live)
    log(f"features: A={len(fs['A'])}  B={len(fs['B'])}  (raw attached: {len(keep_raw_all)})")

    results = []
    for Y in args.origins:
        log(f"\n===== origin Y={Y}  (score snap {Y + GAP}, label debut <= {Y + GAP + EVAL_H}) =====")
        fit_long, Y_fit, ev_base, y = load_origin(Y, keep_raw_all)
        log(f"  fit rows {len(fit_long):,} | eval n={len(y):,} pos={int(y.sum()):,} base={y.mean():.1%} "
            f"[{(time.time() - t0) / 60:.0f}m]")
        w = (0.5 ** ((Y - fit_long["snap_year"].to_numpy()) / args.half_life)).astype(np.float32)
        spec = {"A": (fs["A"], None, False), "B": (fs["B"], None, False),
                "B1": (fs["B"], w, False),      # single stage + recency weights
                "C1": (fs["A"], w, False), "C2": (fs["A"], None, True), "C3": (fs["A"], w, True)}
        for v in args.variants:
            feats, weight, era = spec[v]
            r = run_variant(v, feats, fit_long, Y_fit, ev_base, y, Y, weight, era, t0, log)
            r.update({"Y": Y, "n": int(len(y)), "base": float(y.mean())})
            results.append(r)
            (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
        for a, b in (("A", "B"), ("A", "B1"), ("C1", "B1")):
            fa, fb = OUT_DIR / f"preds_Y{Y}_{a}.npz", OUT_DIR / f"preds_Y{Y}_{b}.npz"
            if fa.exists() and fb.exists():
                pa, pb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
                for wt in (0.5, 0.3):            # weight on the first (stack) member
                    z = wt * _logit(pa["cal3"]) + (1 - wt) * _logit(pb["cal3"])
                    pr = 1.0 / (1.0 + np.exp(-z))
                    nm = f"blend {a}{int(wt*100)}/{b}{int((1-wt)*100)}"
                    r = {"variant": nm, **_metrics(pr, y, nm), "raw": {"ap": float("nan"), "calib": float("nan")},
                         "buckets": bucket_rows(pr, y), "era_offset": 0.0, "n_feats": 0,
                         "Y": Y, "n": int(len(y)), "base": float(y.mean())}
                    results.append(r)
            (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
        del fit_long, Y_fit, ev_base

    df = pd.DataFrame([{k: r[k] for k in ("Y", "variant", "ap", "auc", "calib", "n_feats")} for r in results])
    piv = df.pivot(index="variant", columns="Y", values="ap")
    piv["mean_AP"] = piv.mean(axis=1)
    log("\n===== OUT-OF-ERA, debut <= 3y, AP by origin (calibrated) =====")
    log(piv.round(4).to_string())
    log("\nAUC:")
    log(df.pivot(index="variant", columns="Y", values="auc").round(4).to_string())
    log("\ncalibration ratio (pred/actual):")
    log(df.pivot(index="variant", columns="Y", values="calib").round(3).to_string())
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    log(f"\ndone [{(time.time() - t0) / 60:.0f}m]")


if __name__ == "__main__":
    main()
