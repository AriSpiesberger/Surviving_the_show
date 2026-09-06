"""Full-stack temporal walk-forward: hazards + joint + calibrators per origin.

exp_walkforward showed the hazard layer RANKS well out-of-era but its raw
probabilities drift 1.4-2.9x. This measures how much of that drift the
production stack's upper layers absorb, per origin Y in {2012, 2014, 2016}:

  1. hazards: entry <= Y, labels observed through Y+6 (default HP).
  2. training longs: the pipeline's own scorer (oof._score_checkpointed) on
     the entry<=Y cohort, observe_through=Y+6 — same schema as production.
  3. joint layer: G recipe (FEAT2 + 160 raw, monotone, g3_slow HP), (row,h)
     resolved within Y+6. 2-fold player cross-fit -> HYip2 calibrators
     (2008+ snaps, matching production); final model refit on 100%.
     Also a RECENCY-WEIGHTED calibrator (3y half-life on snap_year).
  4. eval: entry (Y, Y+6] players at snap=Y+6 (the 2026-cohort analog),
     labels = realized debut <= Y+9 from today's outcomes. Report
     hazard-raw vs joint-raw vs joint-cal vs joint-cal-recency:
     AP / AUC / calib + buckets.

Everything caches under runs/exp_walkforward2/Y<origin>/ — reruns resume.

    python -m prospects.model.train.exp_walkforward2
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.hazards import landmark as lm
from prospects.model.joint import EVENTS, H_MAX, prep_base
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.pipelines import oof as oof_mod
from prospects.model.pipelines.oof import _entry_year, stage_panel
from prospects.model.train.joint_xgb import _assemble
from prospects.model.train.exp_cdf_timing2 import FEAT2, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows, train_one

OUT_DIR = REPO_ROOT / "runs" / "exp_walkforward2"
DB = str(REPO_ROOT / "prospects_snapshot.db")
GAP, EVAL_H = 6, 3
G3_SLOW = {"max_depth": 8, "min_child_weight": 100,
           "colsample_bytree": 0.6, "learning_rate": 0.03}
NROUNDS = 340   # v2.3's chosen depth of boosting (fixed; no per-origin ES)


def _metrics(p, y, label):
    pos = int(y.sum())
    ap = float(average_precision_score(y, p)) if pos else float("nan")
    auc = float(roc_auc_score(y, p)) if 0 < pos < len(y) else float("nan")
    calib = float(p.mean() / y.mean()) if pos else float("nan")
    print(f"  {label:<22} AP={ap:.4f}  AUC={auc:.4f}  calib={calib:.2f}")
    return {"scorer": label, "ap": ap, "auc": auc, "calib": calib}


def bucket_rows(p, y):
    edges = [0, .05, .10, .20, .30, .40, .60, .80, 1.001]
    lab = ["0-5", "5-10", "10-20", "20-30", "30-40", "40-60", "60-80",
           "80-100"]
    b = pd.cut(pd.Series(p), edges, labels=lab)
    out = []
    for L in lab:
        m = (b == L).to_numpy()
        if m.sum() < 10:
            continue
        out.append({"bucket": L, "n": int(m.sum()),
                    "pred": float(p[m].mean()), "actual": float(y[m].mean())})
    return out


def run_origin(Y, X_lm, pids, S_yrs, joined, stats_by_pid, entry_by_pid,
               prospects_all, t0):
    snap = Y + GAP
    odir = OUT_DIR / f"Y{Y}"
    odir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== origin Y={Y} (snap {snap}) "
          f"[{(time.time()-t0)/60:.0f}m] =====")

    # -- 1. hazards (cached) ---------------------------------------------
    hz_pkl = odir / "hazards.pkl"
    if hz_pkl.exists():
        hazards = pickle.load(open(hz_pkl, "rb"))
        print(f"  hazards: cached")
    else:
        train_mask = np.array(
            [entry_by_pid.get(p) is not None and entry_by_pid[p] <= Y
             for p in pids], dtype=bool)
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid, train_mask=train_mask,
            seed=42, max_obs_year=snap, verbose=False)
        pickle.dump(hazards, open(hz_pkl, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  hazards: fit [{(time.time()-t0)/60:.0f}m]")

    # -- 2. longs (cached via _score_checkpointed's own partials) --------
    train_pid_set = {p for p, e in entry_by_pid.items()
                     if e is not None and e <= Y}
    eval_pid_set = {p for p, e in entry_by_pid.items()
                    if e is not None and Y < e <= snap}
    fit_csv, ev_csv = odir / "fit_long.csv", odir / "eval_long.csv"
    if not fit_csv.exists():
        n = oof_mod._score_checkpointed(
            hazards, prospects_all, stats_by_pid, train_pid_set, fit_csv,
            odir / "fit_partial", max_entry_year=Y, observe_through=snap,
            max_offset=10, horizon=15)
        print(f"  fit long: {n:,} rows [{(time.time()-t0)/60:.0f}m]")
    if not ev_csv.exists():
        n = oof_mod._score_checkpointed(
            hazards, prospects_all, stats_by_pid, eval_pid_set, ev_csv,
            odir / "eval_partial", max_entry_year=snap, observe_through=snap,
            max_offset=10, horizon=15)
        print(f"  eval long: {n:,} rows [{(time.time()-t0)/60:.0f}m]")
    del hazards

    # -- 3. joint layer ---------------------------------------------------
    with open(REPO_ROOT / "runs" / "current" / "models"
              / "joint_xgb_v2.3.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    feats = list(FEAT2) + keep_raw

    fit_base = prep_base(pd.read_csv(fit_csv), DB, max_entry=Y)
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
        col = f"eligible_{ev}"
        if col in fit_base.columns:
            fit_base = fit_base[fit_base[col] == 1]
    fit_base = attach_raw_features(fit_base, DB, keep_raw, verbose=False)
    fit_long, Y_fit = _assemble(fit_base.copy(), H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    X = fit_long[feats].values.astype(np.float32)
    print(f"  joint fit rows: {len(fit_long):,} "
          f"[{(time.time()-t0)/60:.0f}m]")

    fpids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    snap_yr = fit_long["snap_year"].to_numpy()
    uniq = np.unique(fpids)
    rng = np.random.default_rng(7)
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    hm = np.isin(fpids, list(half))

    oofp = np.full((len(fit_long), len(EVENTS)), np.nan)
    for name, tr, ho in (("A", ~hm, hm), ("B", hm, ~hm)):
        b = train_one(X[tr], Y_fit[tr], feats, G3_SLOW, NROUNDS, 0, 42)
        oofp[ho] = predict_rows(b, X[ho], feats)
        del b
        print(f"  cross-fit {name} done [{(time.time()-t0)/60:.0f}m]")

    k = EVENTS.index("MLB_DEBUT")
    ok = np.isfinite(oofp[:, k]) & (snap_yr >= 2008)
    yb = Y_fit[ok, k].astype(int)
    cal = HYip2Calibrator().fit(oofp[ok, k], h_arr[ok], yip_arr[ok], yb)
    w = 0.5 ** ((Y - snap_yr[ok]) / 3.0)          # 3y half-life recency
    cal_rec = HYip2Calibrator()
    cal_rec.lr.fit(cal_rec._feats(oofp[ok, k], h_arr[ok], yip_arr[ok]),
                   yb, sample_weight=w)

    bst = train_one(X, Y_fit, feats, G3_SLOW, NROUNDS, 0, 42)
    del X

    # -- 4. eval at snap=Y+GAP -------------------------------------------
    ev_base = prep_base(pd.read_csv(ev_csv), DB)
    ev_base = ev_base[(ev_base.snap_year == snap)
                      & (ev_base.get("eligible_MLB_DEBUT", 1) == 1)].copy()
    ev_base = attach_raw_features(ev_base, DB, keep_raw, verbose=False)
    from prospects.model.joint import add_cond_cols
    P_h = {}
    for h in (1, 2, 3):
        sub = stamp_extra_cols(add_cond_cols(ev_base, h))
        P_h[h] = predict_rows(bst, sub[feats].values.astype(np.float32),
                              feats)[:, k]
    del bst
    raw3 = np.maximum.accumulate(
        np.column_stack([P_h[1], P_h[2], P_h[3]]), axis=1)[:, 2]
    yipv = ev_base["snap_offset"].to_numpy()
    h3 = np.full(len(ev_base), 3)
    cal3 = cal.predict(raw3, h3, yipv)
    rec3 = cal_rec.predict(raw3, h3, yipv)

    trig = pd.to_numeric(ev_base["trigger_MLB_DEBUT"], errors="coerce")
    y = ((trig > snap) & (trig <= snap + EVAL_H)).fillna(False)
    y = y.to_numpy().astype(int)
    hk_cols = [f"hk{j}_MLB_DEBUT" for j in (1, 2, 3)]
    hz3 = 1.0 - np.prod(
        1.0 - ev_base[hk_cols].clip(0, 1).fillna(0).to_numpy(), axis=1)

    print(f"  eval: n={len(y):,} pos={int(y.sum()):,} "
          f"base={y.mean():.1%}")
    rows = [_metrics(hz3, y, "hazard raw"),
            _metrics(raw3, y, "joint raw"),
            _metrics(cal3, y, "joint calibrated"),
            _metrics(rec3, y, "joint cal RECENCY")]
    res = {"Y": Y, "snap": snap, "n": int(len(y)), "pos": int(y.sum()),
           "base": float(y.mean()), "scorers": rows,
           "buckets_cal": bucket_rows(cal3, y),
           "buckets_rec": bucket_rows(rec3, y)}
    (odir / "result.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", nargs="*", type=int,
                    default=[2016, 2014, 2012])
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
                          entry_by_pid, prospects_all, t0)
               for Y in args.origins]

    print(f"\n===== FULL-STACK walk-forward summary (debut<= {EVAL_H}y) =====")
    print(f"{'Y':>5}{'base%':>7} | " + " | ".join(
        f"{s:>26}" for s in ("hazard raw", "joint raw", "joint cal",
                             "cal RECENCY")))
    for r in results:
        cells = " | ".join(
            f"AP {s['ap']:.3f} calib {s['calib']:>5.2f}"
            for s in r["scorers"])
        print(f"{r['Y']:>5}{r['base']*100:>6.1f}% | {cells}")
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"results": results,
                    "elapsed_min": round((time.time() - t0) / 60, 1)},
                   indent=2))
    print(f"\nwrote {OUT_DIR}  ({(time.time()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
