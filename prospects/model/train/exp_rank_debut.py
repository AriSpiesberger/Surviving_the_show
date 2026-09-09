"""EXPERIMENT: ranking-loss debut head (LambdaMART pairwise) + post-hoc recal.

The H-form A/B showed the cumulative objective wins AP partly because its
event double-counting acts as implicit positive re-weighting — a ranking
emphasis smuggled into a probability loss. This tests the explicit version:
a debut-only head trained with `rank:pairwise` (queries = horizon h, so
pairs are compared within-h exactly as AP@h evaluates), same features /
augmentation / monotone-h as v2.4, then recalibrated post-hoc on
cross-fitted margins (the architecture treats calibration as a separate
layer, so a non-probabilistic loss costs nothing).

Readout vs v2.4's debut column on the clean val:
  AP@1/3/6 (rank-invariant — the honest test of the ranking loss) and the
  calibrated bucket-quality + CDF timing (does post-hoc calibration fully
  recover a sheet-grade probability from margins?).

    python -m prospects.model.train.exp_rank_debut          # full
    python -m prospects.model.train.exp_rank_debut --quick  # smoke
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base, realized_by_h
from prospects.model.joint2 import attach_raw_features
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import (
    cdf_timing, score_lasso_timing, timing_report,
)
from prospects.model.train.exp_cdf_timing2 import FEAT2, MONO_UP, stamp_extra_cols

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_rank_debut"
# g3_slow minus min_child_weight: pairwise-rank hessians are tiny, so the
# logloss-tuned mcw=100 blocks every split (constant margins, AUC 0.5).
G3 = {"max_depth": 8, "min_child_weight": 1,
      "colsample_bytree": 0.6, "learning_rate": 0.03}
V24_REF = {1: 0.4421, 3: 0.6120, 6: 0.6683}
K_DEB = EVENTS.index("MLB_DEBUT")


class MarginCalibrator:
    """HYip2's form with a z-scored ranking margin in place of logit(p)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=4000)
        self.mu = 0.0
        self.sd = 1.0

    def _feats(self, m, h, yip):
        z = (np.asarray(m, dtype=np.float64) - self.mu) / self.sd
        hc = np.asarray(h, dtype=np.float64) - 5.0
        yc = np.clip(np.asarray(yip, dtype=np.float64), 0, 10) - 3.0
        return np.column_stack([z, hc, yc, z * hc, z * yc, hc * hc, yc * yc])

    def fit(self, m, h, yip, y):
        m = np.asarray(m, dtype=np.float64)
        self.mu, self.sd = float(m.mean()), float(m.std() + 1e-9)
        self.lr.fit(self._feats(m, h, yip), np.asarray(y, dtype=int))
        return self

    def predict(self, m, h, yip):
        return self.lr.predict_proba(self._feats(m, h, yip))[:, 1]


def _mono(feats):
    return "(" + ",".join("1" if f in MONO_UP else "0" for f in feats) + ")"


def _grouped_dm(X, y, h, feats):
    """DMatrix sorted by h with query groups = horizon (order + index map)."""
    order = np.argsort(h, kind="stable")
    d = xgb.DMatrix(X[order], label=y[order], feature_names=list(feats))
    _, counts = np.unique(h[order], return_counts=True)
    d.set_group(counts)
    return d, order


def train_rank(X_tr, y_tr, h_tr, feats, rounds, estop, seed,
               es=None):
    # NOTE: no monotone constraints — they interact badly with the pairwise
    # objective (degenerate constant margins in testing); cross-h coherence
    # is restored post-hoc by the calibrator + cummax. ES on AUC (map over
    # one giant group per h can sit flat and kill early stopping).
    params = dict(tree_method="hist", objective="rank:pairwise",
                  eval_metric="auc", seed=seed, verbosity=0, **G3)
    dtr, _ = _grouped_dm(X_tr, y_tr, h_tr, feats)
    evals, kw = [(dtr, "t")], {}
    if es is not None:
        des, _ = _grouped_dm(*es, feats)
        evals.append((des, "es"))
        kw = dict(early_stopping_rounds=estop)
    return xgb.train(params, dtr, num_boost_round=rounds, evals=evals,
                     verbose_eval=False, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long",
                    default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--db", default=DB)
    ap.add_argument("--bag-seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(_RUN.models / "joint_xgb_v2.4.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    feats = list(FEAT2) + keep_raw

    print(f"[rank] loading longs (+augmentation, v2.4 parity)")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, 2020)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long), args.db)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            c = f"eligible_{ev}"
            if c in aug.columns:
                aug = aug[aug[c] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val), args.db, max_entry=2020)

    rounds_cap, estop = 2000, 50
    if args.quick:
        rng = np.random.default_rng(0)
        pids = fit_base["player_id"].unique()
        keep = set(rng.choice(pids, size=max(200, len(pids) // 7),
                              replace=False))
        fit_base = fit_base[fit_base["player_id"].isin(keep)]
        rounds_cap, estop, args.bag_seeds = 100, 20, 1
        print(f"  [quick] {fit_base.player_id.nunique():,} players")

    fit_base = attach_raw_features(fit_base, args.db, keep_raw, verbose=False)
    fit_long, Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    X = fit_long[feats].values.astype(np.float32)
    y = Y[:, K_DEB].astype(np.float32)
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    era = fit_long["snap_year"].to_numpy() >= 2008
    fpids = fit_long["player_id"].to_numpy()
    print(f"  rows: {len(fit_long):,}, pos {int(y.sum()):,} "
          f"[{(time.time()-t0)/60:.1f}m]")

    uniq = np.unique(fpids)
    rng = np.random.default_rng(7)
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_m = np.isin(fpids, list(es_players))
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    hm = np.isin(fpids, list(half))

    print(f"[rank] ES training")
    b = train_rank(X[~es_m], y[~es_m], h_arr[~es_m], feats, rounds_cap,
                   estop, args.seed, es=(X[es_m], y[es_m], h_arr[es_m]))
    nrounds = int(b.best_iteration) + 1
    print(f"  rounds = {nrounds} (best es-auc "
          f"{b.best_score:.4f}) [{(time.time()-t0)/60:.1f}m]")
    del b

    print(f"[rank] 2-fold cross-fit + margin calibrator")
    marg = np.full(len(fit_long), np.nan)
    for tr_m, ho_m in ((~hm, hm), (hm, ~hm)):
        b = train_rank(X[tr_m], y[tr_m], h_arr[tr_m], feats, nrounds, 0,
                       args.seed)
        marg[ho_m] = b.predict(
            xgb.DMatrix(X[ho_m], feature_names=list(feats)))
        del b
        print(f"  fold done [{(time.time()-t0)/60:.1f}m]")
    ok = era & np.isfinite(marg)
    cal = MarginCalibrator().fit(marg[ok], h_arr[ok], yip_arr[ok],
                                 y[ok].astype(int))

    bag = [train_rank(X, y, h_arr, feats, nrounds, 0, args.seed + 100 + s)
           for s in range(args.bag_seeds)]
    print(f"  bag done [{(time.time()-t0)/60:.1f}m]")
    del X
    with open(out_dir / "rank_debut_bag.pkl", "wb") as fh:
        pickle.dump({"bags": bag, "feature_names": feats,
                     "keep_raw": keep_raw, "nrounds": nrounds,
                     "calibrator": cal, "kind": "rank_pairwise_debut"}, fh)

    # ---- val -------------------------------------------------------------
    print(f"\n[rank] scoring val")
    val_base = attach_raw_features(val_base, args.db, keep_raw, verbose=False)
    yipv = val_base["snap_offset"].to_numpy()
    M_by_h = {}
    for h in range(1, H_MAX + 1):
        sub = stamp_extra_cols(add_cond_cols(val_base, h))
        Xv = sub[feats].values.astype(np.float32)
        d = xgb.DMatrix(Xv, feature_names=list(feats))
        M_by_h[h] = np.mean([bb.predict(d) for bb in bag], axis=0)
    sv = val_base.copy()
    C = np.column_stack([
        cal.predict(M_by_h[h], np.full(len(sv), h), yipv)
        for h in range(1, H_MAX + 1)])
    C = np.maximum.accumulate(C, axis=1)
    new = {}
    for hi, h in enumerate(range(1, H_MAX + 1)):
        new[f"m_MLB_DEBUT_h{h}"] = M_by_h[h]
        new[f"cal_xp_MLB_DEBUT_h{h}"] = C[:, hi]
    sv = pd.concat([sv, pd.DataFrame(new, index=sv.index)], axis=1)

    print(f"\n===== debut vs v2.4 (clean val) =====")
    print(f"{'h':>3}{'R_AP(margin)':>13}{'R_AP(cal)':>10}{'v2.4':>8}"
          f"{'AUC':>8}{'cal calib':>10}")
    res = []
    for h in (1, 2, 3, 6):
        m = (sv.years_fwd >= h) & (sv.eligible_MLB_DEBUT == 1)
        yv = realized_by_h(sv[m], "MLB_DEBUT", h).astype(int)
        pm = sv.loc[m, f"m_MLB_DEBUT_h{h}"].to_numpy()
        pc = sv.loc[m, f"cal_xp_MLB_DEBUT_h{h}"].to_numpy()
        apm = average_precision_score(yv, pm)
        apc = average_precision_score(yv, pc)
        auc = roc_auc_score(yv, pm)
        calib = float(pc.mean() / yv.mean())
        ref = V24_REF.get(h, float("nan"))
        print(f"{h:>3}{apm:>13.4f}{apc:>10.4f}{ref:>8.4f}{auc:>8.4f}"
              f"{calib:>10.2f}")
        res.append({"h": h, "ap_margin": float(apm), "ap_cal": float(apc),
                    "v24": ref, "auc": float(auc), "calib": calib})

    tim = cdf_timing(sv, "MLB_DEBUT", "cal_")
    trep = timing_report(val_base, {
        "R_cdf_median(cal)": tim["t_med"].to_numpy(),
        "lasso_timing.pkl": score_lasso_timing(val_base, _RUN.timing,
                                               args.db),
    }, {"R_cdf_median(cal)": (tim["t_q25"].to_numpy(),
                              tim["t_q75"].to_numpy())})
    print(f"\n[rank] timing (v2.4 median MAE ref 1.034):")
    print(trep.to_string(index=False))

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "quick": bool(args.quick), "nrounds": nrounds,
                   "results": res,
                   "timing": trep.to_dict(orient="records"),
                   "elapsed_min": round((time.time() - t0) / 60, 1)},
                  fh, indent=2, default=float)
    print(f"\n[rank] wrote {out_dir} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
