"""Walk-forward A/B: FULL-FRESH stack (hazards + joint trained on all
person-time observed by the scoring date).

Production currently freezes the hazard panel at entry<=2020 and lets only
the joint layer see recent cohorts (v2.4 augmentation, +0.04..+0.07
out-of-era debut@3 per exp_walkforward3). The user's proposal: the DEPLOYED
stack should train on literally everything observed — recent entrants'
resolved person-years in the HAZARDS too, labels right-censored at the
observation date (the discrete-time framework handles this natively; no
label leakage because scored futures are unresolved).

Deployment analog at origin Y (score snap = Y+6, labels observed <= Y+6):
  BASELINE  (wf2): hazards entry<=Y,   joint entry<=Y
  AUG       (wf3): hazards entry<=Y,   joint + recent rows
  FULL-FRESH(wf4): hazards entry<=Y+5, joint on those longs (all cohorts)

Eval identical to wf2/wf3: entry (Y, Y+6] players at snap Y+6, debut<=3y
against today's outcomes. Note the eval players' PRE-snap person-time is in
training here — that is the point (production refreshes weekly with the
scoring cohort's own history), and their eval labels (post-snap) are
censored out of training by construction.

Reuses the wf2 panel-independent machinery; hazards refit per origin with
the lifted entry cap (panel holds entry<=2020, so Y=2016 gets 2017-20 —
4 of 5 fresh cohorts; Y=2014 gets all 5).

    python -m prospects.model.train.exp_walkforward4
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
from prospects.model.hazards import landmark as lm
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.pipelines import oof as oof_mod
from prospects.model.pipelines.oof import _entry_year, stage_panel
from prospects.model.train.joint_xgb import _assemble
from prospects.model.train.exp_cdf_timing2 import FEAT2, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows, train_one
from prospects.model.train.exp_walkforward2 import (
    DB, G3_SLOW, GAP, EVAL_H, NROUNDS, _metrics, bucket_rows,
)

WF3 = REPO_ROOT / "runs" / "exp_walkforward3"
OUT_DIR = REPO_ROOT / "runs" / "exp_walkforward4"


def run_origin(Y, X_lm, pids, S_yrs, joined, stats_by_pid, entry_by_pid,
               prospects_all, keep_raw, feats, t0):
    snap = Y + GAP
    odir = OUT_DIR / f"Y{Y}"
    odir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== origin Y={Y} FULL-FRESH (hazards entry<={Y+5}, "
          f"labels<= {snap}) [{(time.time()-t0)/60:.0f}m] =====")

    hz_pkl = odir / "hazards_fresh.pkl"
    if hz_pkl.exists():
        hazards = pickle.load(open(hz_pkl, "rb"))
        print(f"  hazards: cached")
    else:
        train_mask = np.array(
            [entry_by_pid.get(p) is not None and entry_by_pid[p] <= Y + 5
             for p in pids], dtype=bool)
        print(f"  hazard landmarks: {int(train_mask.sum()):,}")
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid, train_mask=train_mask,
            seed=42, max_obs_year=snap, verbose=False)
        pickle.dump(hazards, open(hz_pkl, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  hazards fit [{(time.time()-t0)/60:.0f}m]")

    # one long for everyone observed by Y+6 (train cohorts + eval cohort —
    # rows are snaps <= Y+6; training uses snaps < Y+6, eval snap == Y+6)
    all_pid_set = {p for p, e in entry_by_pid.items()
                   if e is not None and e <= snap}
    long_csv = odir / "all_long.csv"
    if not long_csv.exists():
        n = oof_mod._score_checkpointed(
            hazards, prospects_all, stats_by_pid, all_pid_set, long_csv,
            odir / "all_partial", max_entry_year=snap, observe_through=snap,
            max_offset=10, horizon=15)
        print(f"  long: {n:,} rows [{(time.time()-t0)/60:.0f}m]")
    del hazards

    base = prep_base(pd.read_csv(long_csv), DB)
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
        c = f"eligible_{ev}"
        if c in base.columns:
            base = base[base[c] == 1]
    train_base = base[base.snap_year < snap].copy()
    eval_base = base[(base.snap_year == snap)
                     & (base.get("eligible_MLB_DEBUT", 1) == 1)].copy()
    eval_base = eval_base[eval_base["player_id"].map(
        lambda p: entry_by_pid.get(p) is not None
        and Y < entry_by_pid[p] <= snap)]
    train_base = attach_raw_features(train_base, DB, keep_raw, verbose=False)
    eval_base = attach_raw_features(eval_base, DB, keep_raw, verbose=False)

    fit_long, Y_fit = _assemble(train_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    X = fit_long[feats].values.astype(np.float32)
    print(f"  joint fit rows: {len(fit_long):,} "
          f"[{(time.time()-t0)/60:.0f}m]")

    fpids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    era = fit_long["snap_year"].to_numpy() >= 2008
    uniq = np.unique(fpids)
    rng = np.random.default_rng(7)
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    hm = np.isin(fpids, list(half))
    k = EVENTS.index("MLB_DEBUT")

    oofp = np.full((len(fit_long), len(EVENTS)), np.nan)
    for tr, ho in ((~hm, hm), (hm, ~hm)):
        b = train_one(X[tr], Y_fit[tr], feats, G3_SLOW, NROUNDS, 0, 42)
        oofp[ho] = predict_rows(b, X[ho], feats)
        del b
        print(f"  cross-fit fold done [{(time.time()-t0)/60:.0f}m]")
    ok = np.isfinite(oofp[:, k]) & era
    cal = HYip2Calibrator().fit(oofp[ok, k], h_arr[ok], yip_arr[ok],
                                Y_fit[ok, k].astype(int))
    bst = train_one(X, Y_fit, feats, G3_SLOW, NROUNDS, 0, 42)
    del X

    P_h = {}
    for h in (1, 2, 3):
        sub = stamp_extra_cols(add_cond_cols(eval_base, h))
        P_h[h] = predict_rows(bst, sub[feats].values.astype(np.float32),
                              feats)[:, k]
    del bst
    raw3 = np.maximum.accumulate(
        np.column_stack([P_h[1], P_h[2], P_h[3]]), axis=1)[:, 2]
    cal3 = cal.predict(raw3, np.full(len(eval_base), 3),
                       eval_base["snap_offset"].to_numpy())

    trig = pd.to_numeric(eval_base["trigger_MLB_DEBUT"], errors="coerce")
    y = ((trig > snap) & (trig <= snap + EVAL_H)).fillna(False)
    y = y.to_numpy().astype(int)
    print(f"  eval: n={len(y):,} pos={int(y.sum()):,} base={y.mean():.1%}")
    rows = [_metrics(raw3, y, "FRESH joint raw"),
            _metrics(cal3, y, "FRESH joint calibrated")]

    ref = {}
    wf3_json = WF3 / "summary.json"
    if wf3_json.exists():
        for r in json.loads(wf3_json.read_text())["results"]:
            if r["Y"] == Y:
                a = next(s for s in r["aug"] if "calibrated" in s["scorer"])
                ref = {"aug_ap": a["ap"], "aug_calib": a["calib"],
                       "base_ap": r["baseline_cal"]["ap"],
                       "base_calib": r["baseline_cal"]["calib"]}
    if ref:
        print(f"  {'wf3 AUG ref':<22} AP={ref['aug_ap']:.4f}  "
              f"calib={ref['aug_calib']:.2f}")
        print(f"  {'wf2 baseline ref':<22} AP={ref['base_ap']:.4f}  "
              f"calib={ref['base_calib']:.2f}")
    res = {"Y": Y, "n": int(len(y)), "base": float(y.mean()),
           "fresh": rows, "refs": ref, "buckets": bucket_rows(cal3, y)}
    (odir / "result.json").write_text(json.dumps(res, indent=2,
                                                 default=float))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", nargs="*", type=int, default=[2016, 2014])
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / "runs" / "current" / "models"
              / "joint_xgb_v2.4.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    feats = list(FEAT2) + keep_raw

    X_lm, pids, S_yrs, joined, stats_by_pid = stage_panel(DB, 2020)
    entry_by_pid: dict = {}
    prospects_all, seen = [], set()
    for p in joined:
        pid = p["player_id"]
        if pid not in seen:
            seen.add(pid)
            prospects_all.append(p)
            entry_by_pid[pid] = _entry_year(p, stats_by_pid)

    results = [run_origin(Y, X_lm, pids, S_yrs, joined, stats_by_pid,
                          entry_by_pid, prospects_all, keep_raw, feats, t0)
               for Y in args.origins]

    print(f"\n===== FULL-FRESH vs AUG vs baseline (debut<= {EVAL_H}y) =====")
    for r in results:
        f = next(s for s in r["fresh"] if "calibrated" in s["scorer"])
        rr = r["refs"]
        print(f"  Y={r['Y']}: baseline {rr.get('base_ap', float('nan')):.4f}"
              f" -> AUG {rr.get('aug_ap', float('nan')):.4f}"
              f" -> FRESH {f['ap']:.4f}   "
              f"(calib {rr.get('base_calib', float('nan')):.2f} -> "
              f"{rr.get('aug_calib', float('nan')):.2f} -> {f['calib']:.2f})")
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"results": results,
         "elapsed_min": round((time.time() - t0) / 60, 1)}, indent=2,
        default=float))
    print(f"\nwrote {OUT_DIR}  ({(time.time()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
