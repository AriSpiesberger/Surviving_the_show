"""EXPERIMENT 7: calibrators fit on cross-fitted BAGS (v2.3b candidate).

The v2.3 calibrators are fit on 3-fold cross-fitted SINGLE-model predictions
but applied to the deployed 5-seed bag trained on 100% of fit. More data +
bag averaging shifts the raw-score distribution, so the map is misaligned
with the scores it calibrates — the stable ~0.95 deployment-era calib is the
signature. Fix: per fold, train a small BAG on the other folds and predict
the held fold; fit the HYip2 map (2008+ snaps) on those — calibration data
now comes from the deployed model class.

Uses the promoted v2.3 recipe verbatim (hz3 longs, g3_slow HP, 342 rounds,
FEAT2 + 160 raw full-coverage). Writes runs/exp_cal_bagfit/ and prints the
before/after val calib per h and per era against the CURRENT v2.3 bundle —
promotion (overwriting calibrators_v2.3.pkl) is a separate explicit step.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, prep_base, realized_by_h
from prospects.model.joint2 import (
    HYip2Calibrator, apply_calibrators_frame, attach_raw_features,
    load_calibrators, score_trajectory,
)
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing2 import FEAT2, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows, train_one
from prospects.model.joint2 import _logit


class HingeCalibrator:
    """HYip2 + monotone hinge terms in logit(p) — lets the map's slope
    differ by probability region (the S-shape fix) while staying (near-)
    monotone. Knots at p≈0.15 and p=0.5."""
    KNOTS = (-1.7346, 0.0)

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=5000)

    @classmethod
    def _feats(cls, p, h, yip):
        lp = _logit(p)
        hc = np.asarray(h, dtype=np.float64) - 5.0
        yc = np.clip(np.asarray(yip, dtype=np.float64), 0, 10) - 3.0
        cols = [lp] + [np.maximum(lp - k, 0.0) for k in cls.KNOTS]
        cols += [hc, yc, lp * hc, lp * yc, hc * hc, yc * yc]
        return np.column_stack(cols)

    def fit(self, p, h, yip, y):
        self.lr.fit(self._feats(p, h, yip), np.asarray(y, dtype=int))
        return self

    def predict(self, p, h, yip):
        return self.lr.predict_proba(self._feats(p, h, yip))[:, 1]


def bucket_table(p, y, label):
    edges = [0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.001]
    lab = ["0-5", "5-10", "10-20", "20-30", "30-40", "40-50",
           "50-60", "60-70", "70-80", "80-90", "90-100"]
    b = pd.cut(pd.Series(p), edges, labels=lab)
    print(f"  --- {label} ---")
    print(f"  {'bucket':<8}{'n':>7}{'pred':>7}{'actual':>8}{'diff':>8}")
    for L in lab:
        m = (b == L).to_numpy()
        if m.sum() < 15:
            continue
        print(f"  {L:<8}{int(m.sum()):>7,}{p[m].mean():>6.1%}"
              f"{y[m].mean():>8.1%}{y[m].mean()-p[m].mean():>+8.1%}")

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_cal_bagfit"
G3_SLOW = {"max_depth": 8, "min_child_weight": 100,
           "colsample_bytree": 0.6, "learning_rate": 0.03}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--db", default=DB)
    ap.add_argument("--xgb", default=str(_RUN.models / "joint_xgb_v2.3.pkl"))
    ap.add_argument("--old-cals",
                    default=str(_RUN.models / "calibrators_v2.3.pkl"))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold-bag-seeds", type=int, default=2)
    ap.add_argument("--cal-min-snap-year", type=int, default=2008)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(args.xgb, "rb") as fh:
        bundle = pickle.load(fh)
    keep_raw = list(bundle["keep_raw"])
    feats = list(bundle["feature_names"])
    nrounds = int(bundle["recipe"]["num_rounds"])
    print(f"[cal-bagfit] recipe: {nrounds} rounds, {len(feats)} feats, "
          f"{args.folds} folds x {args.fold_bag_seeds}-seed bags")

    fit_base = _prep_train(pd.read_csv(args.fit), args.db, 2020)
    fit_base = attach_raw_features(fit_base, args.db, keep_raw)
    fit_long, Y_fit = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    X = fit_long[feats].values.astype(np.float32)
    print(f"  fit long: {len(fit_long):,} rows [{(time.time()-t0)/60:.1f}m]")

    fit_pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    uniq = np.unique(fit_pids)
    rng = np.random.default_rng(7)
    rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False)  # burn ES draw
    fold_of = {p: i % args.folds for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in fit_pids])

    oof_npz = OUT_DIR / (f"oof_bag_preds_f{args.folds}"
                         f"s{args.fold_bag_seeds}.npz")
    if oof_npz.exists():
        print(f"  reusing cached OOF bag predictions {oof_npz.name}")
        oof = np.load(oof_npz)["oof"]
        del X
    else:
        oof = np.full((len(fit_long), len(EVENTS)), np.nan)
        for f in range(args.folds):
            tr, ho = fold_idx != f, fold_idx == f
            preds = []
            for s in range(args.fold_bag_seeds):
                b = train_one(X[tr], Y_fit[tr], feats, G3_SLOW, nrounds, 0,
                              args.seed + 300 + f * 10 + s)
                preds.append(predict_rows(b, X[ho], feats))
                del b
                print(f"  fold {f} seed {s} done [{(time.time()-t0)/60:.1f}m]")
            oof[ho] = np.mean(preds, axis=0)
        del X
        np.savez_compressed(oof_npz, oof=oof)
        print(f"  cached OOF bag predictions -> {oof_npz.name}")

    cals: dict = {}
    hinge_cals: dict = {}
    era_ok = fit_long["snap_year"].to_numpy() >= args.cal_min_snap_year
    for k, ev in enumerate(EVENTS):
        elig = (fit_long[f"eligible_{ev}"] == 1).to_numpy() \
            if f"eligible_{ev}" in fit_long.columns \
            else np.ones(len(fit_long), bool)
        ok = elig & era_ok & np.isfinite(oof[:, k])
        y = Y_fit[ok, k].astype(int)
        if y.sum() < 25 or y.sum() == len(y):
            continue
        cals[ev] = HYip2Calibrator().fit(oof[ok, k], h_arr[ok],
                                         yip_arr[ok], y)
        hinge_cals[ev] = HingeCalibrator().fit(oof[ok, k], h_arr[ok],
                                               yip_arr[ok], y)

    # ---- DISCRIMINATOR: does the S-shape exist on the honest FIT-OOF
    # buckets? If yes -> the linear-logit map is too rigid (hinge fixes it
    # honestly). If fit-OOF is flat but val is S-shaped -> population
    # difference, no honest calibrator fix. Debut @ h=3, 2008+ rows.
    k_deb = EVENTS.index("MLB_DEBUT")
    elig = (fit_long["eligible_MLB_DEBUT"] == 1).to_numpy()
    m = elig & era_ok & np.isfinite(oof[:, k_deb]) & (h_arr == 3)
    yb = Y_fit[m, k_deb].astype(int)
    print(f"\n[cal-bagfit] FIT-OOF buckets (debut h=3, 2008+, "
          f"n={int(m.sum()):,}):")
    p_lin = cals["MLB_DEBUT"].predict(oof[m, k_deb], h_arr[m], yip_arr[m])
    bucket_table(p_lin, yb, "fit-OOF, linear-logit (HYip2)")
    p_hng = hinge_cals["MLB_DEBUT"].predict(oof[m, k_deb], h_arr[m],
                                            yip_arr[m])
    bucket_table(p_hng, yb, "fit-OOF, hinge")
    out_cal = OUT_DIR / "calibrators_bagfit.pkl"
    with open(out_cal, "wb") as fh:
        pickle.dump({"calibrators": cals, "events": list(EVENTS),
                     "h_max": H_MAX, "version": "v2.3b",
                     "kind": "hyip2_logistic_oof",
                     "fit_on": f"{args.folds}x{args.fold_bag_seeds}-seed "
                               f"cross-fitted bags, snaps>="
                               f"{args.cal_min_snap_year}"}, fh)
    print(f"  wrote {out_cal}")

    # ---- val before/after ----------------------------------------------
    print(f"\n[cal-bagfit] val comparison (deployed v2.3 bag)")
    val = prep_base(pd.read_csv(args.val), args.db, max_entry=2020)
    val, _ = score_trajectory(args.xgb, val, args.db)
    old = apply_calibrators_frame(val, load_calibrators(args.old_cals))
    new = apply_calibrators_frame(val, {"calibrators": cals,
                                        "kind": "hyip2_logistic_oof"})
    hng = apply_calibrators_frame(val, {"calibrators": hinge_cals,
                                        "kind": "hyip2_logistic_oof"})
    with open(OUT_DIR / "calibrators_hinge.pkl", "wb") as fh:
        pickle.dump({"calibrators": hinge_cals, "events": list(EVENTS),
                     "h_max": H_MAX, "version": "v2.3b-hinge",
                     "kind": "hyip2_logistic_oof"}, fh)

    print(f"\n[cal-bagfit] VAL buckets (debut h=3, 2008+):")
    mv = (val.years_fwd >= 3) & (val.eligible_MLB_DEBUT == 1) \
        & (val.snap_year >= 2008)
    yv = realized_by_h(val[mv], "MLB_DEBUT", 3).astype(int)
    for frame, lbl in ((old, "val, OLD v2.3 cals"),
                       (new, "val, bagfit linear"),
                       (hng, "val, bagfit hinge")):
        bucket_table(frame.loc[mv, "xp_MLB_DEBUT_h3"].to_numpy(), yv, lbl)

    rows = []
    print(f"{'slice':<26}{'h':>3}{'old_calib':>10}{'new_calib':>10}")
    for name, era in (("all", 0), ("2008+", 2008), ("2016+", 2016)):
        for h in (1, 3, 6):
            m = (val.years_fwd >= h) & (val.eligible_MLB_DEBUT == 1) \
                & (val.snap_year >= era)
            y = realized_by_h(val[m], "MLB_DEBUT", h)
            oc = float(old.loc[m, f"xp_MLB_DEBUT_h{h}"].mean() / y.mean())
            nc = float(new.loc[m, f"xp_MLB_DEBUT_h{h}"].mean() / y.mean())
            rows.append({"slice": name, "h": h, "old": oc, "new": nc})
            print(f"{'debut ' + name:<26}{h:>3}{oc:>10.3f}{nc:>10.3f}")
    for ev in ("TOP_100_PROSPECT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"):
        for h in (3, 6):
            m = (val.years_fwd >= h) & (val.get(f"eligible_{ev}", 1) == 1) \
                & (val.snap_year >= 2008)
            y = realized_by_h(val[m], ev, h)
            if y.sum() < 10:
                continue
            oc = float(old.loc[m, f"xp_{ev}_h{h}"].mean() / y.mean())
            nc = float(new.loc[m, f"xp_{ev}_h{h}"].mean() / y.mean())
            rows.append({"slice": f"{ev} 2008+", "h": h, "old": oc, "new": nc})
            print(f"{ev[:22] + ' 2008+':<26}{h:>3}{oc:>10.3f}{nc:>10.3f}")

    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rows": rows,
                   "elapsed_min": round((time.time() - t0) / 60, 1)},
                  fh, indent=2)
    print(f"\n[cal-bagfit] done ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
