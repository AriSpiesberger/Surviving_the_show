"""Calibration top-end experiment (2026-09-17).

The deployed v2.4 stack is well calibrated below ~0.70 and runs 5-8 points hot
above it in every held-out slice (printed 0.85 -> realised ~0.78). Two separate
hypotheses, tested separately:

  SHAPE     HYip2Calibrator is one logistic curve in logit(p); a single sigmoid
            cannot bend the top independently of the middle.
  MISMATCH  the calibrator is fit on cross-fit predictions from SINGLE models
            trained on 2/3 of the fit players, then applied to a 5-seed BAG
            trained on 100%. More data and averaging change the score
            distribution the calibrator is asked to map.

Everything is fit on cross-fitted predictions of fit players only (snaps >=
--cal-min-snap-year) and judged on the held-out val players, exactly as the
deployed calibrators are. Reproduces exp_cdf_timing5's data prep, fold
assignment and seeds, so variant "single/hyip2" IS the deployed calibrator.

The cross-fit predictions are saved to <out-dir>/oof.npz so later calibration
work does not have to pay for the refits again.

    python -m prospects.model.train.exp_cal_topend
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, prep_base, realized_by_h
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.exp_cdf_timing import _logit
from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows, sweep_val, train_one
from prospects.model.train.exp_cdf_timing5 import G3_SLOW
from prospects.model.train.joint_xgb import _assemble, _prep_train

_RUN = config.run()
DB = str(config.model_db())
KNOTS = (0.50, 0.70, 0.85)


class HYip2Spline(HYip2Calibrator):
    """HYip2 plus hinge terms in logit(p), so the top can bend on its own."""

    @staticmethod
    def _feats(p, h, yip):
        base = HYip2Calibrator._feats(p, h, yip)
        lp = _logit(p)
        hinges = [np.maximum(0.0, lp - _logit(k)) for k in KNOTS]
        return np.column_stack([base] + hinges)

    def __init__(self):
        self.lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=6000)


class HYip2Iso:
    """HYip2, then a monotone non-parametric correction of its output."""

    def __init__(self):
        self.base = HYip2Calibrator()
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p, h, yip, y):
        self.base.fit(p, h, yip, y)
        self.iso.fit(self.base.predict(p, h, yip), np.asarray(y, dtype=float))
        return self

    def predict(self, p, h, yip):
        return self.iso.predict(self.base.predict(p, h, yip))


VARIANTS = {"hyip2": HYip2Calibrator, "spline": HYip2Spline, "iso": HYip2Iso}


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return ece


def _crossfit(args, feats, keep_raw, rounds, out, tick):
    """exp_cdf_timing5's data prep and cross-fit, returning what the calibrators need."""
    # ---- data prep: identical to exp_cdf_timing5 ---------------------------
    fit_base = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if args.aug_long and Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    fit_base = attach_raw_features(fit_base, DB, keep_raw, verbose=False)
    fit_long, Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    X = fit_long[feats].values.astype(np.float32)
    pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    tick(f"fit long {len(fit_long):,} rows, {len(feats)} feats, {rounds} rounds")

    # same RNG stream as exp5: ES players are drawn first, then the fold permutation
    uniq = np.unique(pids)
    rng = np.random.default_rng(7)
    rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False)
    fold_of = {p: i % args.folds for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in pids])

    oof_single = np.full((len(fit_long), len(EVENTS)), np.nan)
    oof_bag = np.zeros((len(fit_long), len(EVENTS)))
    for f in range(args.folds):
        tr, ho = fold_idx != f, fold_idx == f
        acc = np.zeros((int(ho.sum()), len(EVENTS)))
        for s in range(args.fold_seeds):
            seed = args.seed + 200 + f if s == 0 else args.seed + 300 + 10 * f + s
            b = train_one(X[tr], Y[tr], feats, G3_SLOW, rounds, 0, seed)
            pr = predict_rows(b, X[ho], feats)
            if s == 0:
                oof_single[ho] = pr          # exactly what the deployed calibrator saw
            acc += pr
            del b
            tick(f"fold {f} seed {s} done")
        oof_bag[ho] = acc / args.fold_seeds
    era_ok = fit_long["snap_year"].to_numpy() >= args.cal_min_snap_year
    elig = np.column_stack([
        (fit_long[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in fit_long.columns
        else np.ones(len(fit_long), bool) for ev in EVENTS])
    np.savez_compressed(out / "oof.npz", oof_single=oof_single, oof_bag=oof_bag, y=Y,
                        h=h_arr, yip=yip_arr, snap_year=fit_long["snap_year"].to_numpy(),
                        elig=elig, pids=pids, events=np.array(EVENTS))
    tick("saved oof.npz")

    return oof_single, oof_bag, Y, h_arr, yip_arr, elig, era_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--cal-min-snap-year", type=int, default=2008)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold-seeds", type=int, default=3,
                    help="models averaged per cross-fit fold for the bag-matched OOF")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "exp_cal_topend"))
    ap.add_argument("--from-oof", action="store_true",
                    help="skip the cross-fit; load <out-dir>/oof.npz from a previous run")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def tick(msg):
        print(f"[cal] {msg}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bag = pickle.load(fh)
    feats, keep_raw, rounds = list(bag["feature_names"]), list(bag["keep_raw"]), int(bag["num_rounds"])

    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    val_base = attach_raw_features(val_base, DB, keep_raw, verbose=False)
    if args.from_oof:
        z = np.load(out / "oof.npz", allow_pickle=True)
        oof_single, oof_bag, Y = z["oof_single"], z["oof_bag"], z["y"]
        h_arr, yip_arr, elig = z["h"], z["yip"], z["elig"]
        era_ok = z["snap_year"] >= args.cal_min_snap_year
        tick("loaded oof.npz")
    else:
        oof_single, oof_bag, Y, h_arr, yip_arr, elig, era_ok = _crossfit(args, feats, keep_raw, rounds, out, tick)

    # ---- held-out val, scored by the DEPLOYED bag --------------------------
    sv = sweep_val(bag["models"], feats, val_base)
    for ev in EVENTS:
        cols = [f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]
        sv[cols] = np.maximum.accumulate(sv[cols].to_numpy(dtype=np.float64), axis=1)
    yip_val = sv["snap_offset"].to_numpy()
    tick("val scored with the deployed bag")

    rows, cals_out = [], {}
    for src, oof in (("single", oof_single), ("bagmatched", oof_bag)):
        for vname, cls in VARIANTS.items():
            cals = {}
            for k, ev in enumerate(EVENTS):
                ok = elig[:, k] & era_ok & np.isfinite(oof[:, k])
                y = Y[ok, k].astype(int)
                if y.sum() < 25 or y.sum() == len(y):
                    continue
                cals[ev] = cls().fit(oof[ok, k], h_arr[ok], yip_arr[ok], y)
            cals_out[f"{src}/{vname}"] = cals
            for ev, cal in cals.items():
                P = np.column_stack([cal.predict(sv[f"xp_{ev}_h{h}"].to_numpy(dtype=np.float64),
                                                 np.full(len(sv), h), yip_val)
                                     for h in range(1, H_MAX + 1)])
                P = np.maximum.accumulate(P, axis=1)
                for h in (1, 3, 6):
                    m = (sv["years_fwd"].to_numpy() >= h)
                    if f"eligible_{ev}" in sv.columns:
                        m &= (sv[f"eligible_{ev}"] == 1).to_numpy()
                    p = P[m, h - 1]
                    y = np.asarray(realized_by_h(sv[m], ev, h), dtype=float)
                    for slab, sm in (("all", np.ones(len(p), bool)),
                                     ("yip<=3", sv["snap_offset"].to_numpy()[m] <= 3)):
                        pp, yy = p[sm], y[sm]
                        top = pp >= 0.70
                        rows.append({
                            "oof": src, "cal": vname, "event": ev, "h": h, "slice": slab,
                            "n": len(pp), "ece": reliability(pp, yy),
                            "brier": float(np.mean((pp - yy) ** 2)),
                            "logloss": float(-np.mean(yy * np.log(np.clip(pp, 1e-9, 1)) +
                                                      (1 - yy) * np.log(np.clip(1 - pp, 1e-9, 1)))),
                            "ratio": float(pp.mean() / max(yy.mean(), 1e-12)),
                            "top_n": int(top.sum()),
                            "top_pred": float(pp[top].mean()) if top.any() else np.nan,
                            "top_actual": float(yy[top].mean()) if top.any() else np.nan,
                        })
    res = pd.DataFrame(rows)
    res["top_gap"] = res.top_actual - res.top_pred
    res.to_csv(out / "results.csv", index=False)
    with open(out / "calibrators_all_variants.pkl", "wb") as fh:
        pickle.dump(cals_out, fh)

    pd.set_option("display.width", 200)
    for ev, h in (("MLB_DEBUT", 3), ("MLB_DEBUT", 6), ("TOP_100_PROSPECT", 6), ("ESTABLISHED_MLB", 6)):
        v = res[(res.event == ev) & (res.h == h) & (res.slice == "yip<=3")]
        print(f"\n===== {ev}  h={h}  held-out, years-in-pro <= 3 =====")
        print(v[["oof", "cal", "n", "ece", "brier", "logloss", "ratio", "top_n",
                 "top_pred", "top_actual", "top_gap"]].round(4).to_string(index=False))
    tick("done")


if __name__ == "__main__":
    main()
