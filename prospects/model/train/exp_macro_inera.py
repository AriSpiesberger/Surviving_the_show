"""In-era check of the macro bake-off winner (2026-09-19).

Out of era (exp_macro_bc, 3 origins) a 50/50 logit blend of
  C1 = the production stack features, recency-weighted fit, and
  B1 = a single-stage GBM with NO hazard inputs (all 325 raw features), recency-weighted
beat the production stack by +0.043 mean AP and at every origin. Before it is allowed
near the sheet it has to not give that back IN era: same fit players, same held-out val
players, same recent-cohort augmentation and calibration protocol as the deployed v2.4
(exp_cdf_timing5), which is the control and is loaded from disk, not refit.

Reports, on held-out val, for A (deployed v2.4), C1, B1 and the blends:
AP / AUC / calibration ratio at h = 1, 3, 6; ECE and the >=0.70 region at h=3; and the
basket test — precision of each model's top-N at the same N the deployed sheet rule selects
per years-in-pro.

    python -m prospects.model.train.exp_macro_inera
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
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import EVENTS, H_MAX, prep_base, realized_by_h
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.exp_cdf_timing import _logit
from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows, sweep_val
from prospects.model.train.exp_cal_topend import reliability
from prospects.model.train.exp_macro_bc import feature_sets, train_w
from prospects.model.train.joint_xgb import _assemble, _prep_train

_RUN = config.run()
DB = str(config.model_db())
EV = "MLB_DEBUT"
K = EVENTS.index(EV)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--cal", default=str(_RUN.scratch / "v24_build" / "calibrators_hyip2.pkl"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--cal-min-snap-year", type=int, default=2008)
    ap.add_argument("--half-life", type=float, default=4.0)
    ap.add_argument("--bag-seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "exp_macro_inera"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[inera] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bagA = pickle.load(fh)
    with open(args.cal, "rb") as fh:
        calA = pickle.load(fh)["calibrators"][EV]
    rounds, keep_raw = int(bagA["num_rounds"]), list(bagA["keep_raw"])
    fs = feature_sets(keep_raw)
    assert fs["A"] == list(bagA["feature_names"]), "control feature set drifted from the deployed bag"
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    raw_all = sorted(set(keep_raw) | (set(fs["B"]) & live))

    fit_base = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    fit_base = attach_raw_features(fit_base, DB, raw_all, verbose=False)
    val_base = attach_raw_features(val_base, DB, raw_all, verbose=False)
    fit_long, Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    snap_yr = fit_long["snap_year"].to_numpy()
    w = (0.5 ** ((args.max_entry - snap_yr) / args.half_life)).astype(np.float32)
    era_ok = snap_yr >= args.cal_min_snap_year
    elig = (fit_long[f"eligible_{EV}"] == 1).to_numpy() if f"eligible_{EV}" in fit_long.columns \
        else np.ones(len(fit_long), bool)
    uniq = np.unique(pids)
    rng = np.random.default_rng(7)
    rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False)      # same stream as exp5
    fold_of = {p: i % args.folds for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in pids])
    tick(f"fit long {len(fit_long):,} rows; rounds {rounds}; A {len(fs['A'])} feats, B {len(fs['B'])} feats")

    # ---- held-out val trajectories --------------------------------------------------
    def val_traj(models, feats, cal):
        sv = sweep_val(models, feats, val_base)
        cols = [f"xp_{EV}_h{h}" for h in range(1, H_MAX + 1)]
        raw = np.maximum.accumulate(sv[cols].to_numpy(dtype=np.float64), axis=1)
        yip = sv["snap_offset"].to_numpy()
        P = np.column_stack([cal.predict(raw[:, h - 1], np.full(len(sv), h), yip)
                             for h in range(1, H_MAX + 1)])
        return sv, np.maximum.accumulate(P, axis=1)

    sv, P_A = val_traj(bagA["models"], fs["A"], calA)
    preds = {"A (deployed v2.4)": P_A}
    tick("control scored")

    for name, feats in (("C1", fs["A"]), ("B1", fs["B"])):
        X = fit_long[feats].values.astype(np.float32)
        oof = np.full(len(fit_long), np.nan)
        for f in range(args.folds):
            tr, ho = fold_idx != f, fold_idx == f
            b = train_w(X[tr], Y[tr], feats, rounds, 42 + 200 + f, w[tr])
            oof[ho] = predict_rows(b, X[ho], feats)[:, K]
            del b
        ok = elig & era_ok & np.isfinite(oof)
        cal = HYip2Calibrator().fit(oof[ok], h_arr[ok], yip_arr[ok], Y[ok, K].astype(int))
        bag = [train_w(X, Y, feats, rounds, 42 + 100 + s, w) for s in range(args.bag_seeds)]
        del X
        _, preds[name] = val_traj(bag, feats, cal)
        with open(out / f"bag_{name}.pkl", "wb") as fh:
            pickle.dump({"models": bag, "feature_names": feats, "calibrator": cal,
                         "num_rounds": rounds, "half_life": args.half_life}, fh)
        del bag
        tick(f"{name} fit, calibrated, scored")

    blend = lambda a, b, wa: 1 / (1 + np.exp(-(wa * _logit(preds[a]) + (1 - wa) * _logit(preds[b]))))
    preds["blend C1 50 / B1 50"] = blend("C1", "B1", 0.5)
    preds["blend C1 30 / B1 70"] = blend("C1", "B1", 0.3)
    preds["blend A 50 / B1 50"] = blend("A (deployed v2.4)", "B1", 0.5)
    np.savez_compressed(out / "val_preds.npz", pid=sv["player_id"].to_numpy(),
                        snap_year=sv["snap_year"].to_numpy(), yip=sv["snap_offset"].to_numpy(),
                        **{k.replace(" ", "_").replace("/", "-"): v for k, v in preds.items()})

    # ---- metrics ---------------------------------------------------------------------
    yf = sv["years_fwd"].to_numpy()
    el = (sv[f"eligible_{EV}"] == 1).to_numpy() if f"eligible_{EV}" in sv.columns else np.ones(len(sv), bool)
    yip_v = sv["snap_offset"].to_numpy()
    rows = []
    for name, P in preds.items():
        for h in (1, 3, 6):
            m = (yf >= h) & el
            p, y = P[m, h - 1], np.asarray(realized_by_h(sv[m], EV, h), dtype=float)
            top = p >= 0.70
            rows.append({"model": name, "h": h, "n": int(m.sum()),
                         "ap": average_precision_score(y, p), "auc": roc_auc_score(y, p),
                         "calib": p.mean() / y.mean(), "ece": reliability(p, y),
                         "brier": float(np.mean((p - y) ** 2)),
                         "top_n": int(top.sum()), "top_pred": float(p[top].mean()),
                         "top_actual": float(y[top].mean())})
    res = pd.DataFrame(rows)

    # basket test at h=3: every model picks the same NUMBER per yip that the deployed rule picks
    thr = {int(k): float(v) for k, v in json.load(open(_RUN.yip_thresholds(60))).items()}
    m = (yf >= 3) & el & (yip_v <= 3)
    y3 = np.asarray(realized_by_h(sv[m], EV, 3), dtype=float)
    yv = yip_v[m]
    pa = preds["A (deployed v2.4)"][m, 2]
    basket = []
    for name, P in preds.items():
        p = P[m, 2]
        sel = np.zeros(len(p), bool)
        for yy in range(0, 4):
            idx = np.where(yv == yy)[0]
            n_take = int((pa[idx] >= thr.get(yy, 9)).sum())
            sel[idx[np.argsort(-p[idx])[:n_take]]] = True
        basket.append({"model": name, "n_selected": int(sel.sum()), "precision": float(y3[sel].mean()),
                       "debuts_captured": int(y3[sel].sum()), "of_total": int(y3.sum())})
    bk = pd.DataFrame(basket)
    res.to_csv(out / "metrics.csv", index=False)
    bk.to_csv(out / "basket.csv", index=False)

    pd.set_option("display.width", 220)
    for h in (3, 6, 1):
        print(f"\n===== held-out val, P(debut <= {h}y) =====")
        print(res[res.h == h].drop(columns="h").round(4).to_string(index=False))
    print("\n===== basket test, h=3, yip<=3: same N per yip as the deployed rule selects =====")
    print(bk.round(4).to_string(index=False))
    tick("done")


if __name__ == "__main__":
    main()
