"""EXPERIMENT 5b: retrain the exp4 champion with FULL-coverage raw features.

Why: exp3/exp4 attached raw features from the landmark panel cache, which
only covers landmarks 2007-2024 for the training universe — pre-2007 snaps
(mostly IFA history) and post-panel snaps trained and evaluated as NaN. The
promoted v2.2 pipeline builds raw features for EVERY row (deployment must:
the 2026 cohort is entirely outside the panel), and feeding real values to a
model trained to expect NaN on those rows is out-of-distribution — val AUC
drops hard on the affected slice. Fix: train with the same full-coverage
attachment inference uses (prospects.model.joint2.attach_raw_features —
panel fast path where covered, DB build elsewhere; values verified identical
where both exist).

Recipe frozen from exp4's winner (g3_slow): FEAT2 + exp4's 160 raw features,
depth 8 / mcw 100 / colsample 0.6 / lr 0.03, monotone h + margins, ES on the
internal seed-7 player split, 5-seed bag on 100% fit, 3-fold cross-fit ->
HYip2 h/yip calibrators.

Writes runs/exp_cdf_timing5/. Touches nothing under runs/current/ except
reading the panel cache.
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
from prospects.model.joint import EVENTS, H_MAX, PUBLISH_H, prep_base
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import (
    per_h_metrics, score_lasso_timing, timing_report, weighted_ap_at,
)
from prospects.model.train.exp_cdf_timing2 import FEAT2, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import (
    apply_cal_trajectory, predict_rows, sweep_val, train_one,
)

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_cdf_timing5"
EXP4_DIR = REPO_ROOT / "runs" / "exp_cdf_timing4"

G3_SLOW = {"max_depth": 8, "min_child_weight": 100,
           "colsample_bytree": 0.6, "learning_rate": 0.03}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--db", default=DB)
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--bag-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--aug-long", default=None,
                    help="Optional recent-cohort augmentation long (from "
                         "score_recent_cohorts). Its snaps are appended to "
                         "the fit longs; only (row,h) pairs resolved by the "
                         "observation cutoff survive the years_fwd>=h mask. "
                         "Walk-forward-proven (+0.04..+0.07 out-of-era "
                         "debut@3, exp_walkforward3).")
    ap.add_argument("--cal-min-snap-year", type=int, default=0,
                    help="Fit calibrators only on snaps >= this year. The "
                         "era-drift analysis (2026-09-05) showed pre-2008 "
                         "snaps are a different data regime (calib 0.79 vs "
                         "0.91-1.09 for 2008+) — pass 2008 so the calibration "
                         "map describes the deployment-relevant eras.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Raw-feature list: prefer the promoted bundle (stable prod dependency);
    # fall back to the exp4 experiment artifact it originally came from.
    _kr_src = _RUN.models / "joint_xgb_v2.3.pkl"
    if not _kr_src.exists():
        _kr_src = EXP4_DIR / "joint_xgb_exp4_bag.pkl"
    with open(_kr_src, "rb") as fh:
        b4 = pickle.load(fh)
    keep_raw = list(b4["keep_raw"])
    print(f"[exp5] keep_raw ({len(keep_raw)}) from {_kr_src.name}")
    feats = list(FEAT2) + keep_raw
    del b4

    print(f"[exp5] loading longs")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, args.max_entry)
    if args.aug_long and Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long), args.db)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            col = f"eligible_{ev}"
            if col in aug.columns:
                aug = aug[aug[col] == 1]
        print(f"  augmentation: {len(aug):,} recent-cohort snaps "
              f"({Path(args.aug_long).name})")
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    elif args.aug_long:
        print(f"  WARN --aug-long {args.aug_long} missing — skipped")
    val_base = prep_base(pd.read_csv(args.val), args.db,
                         max_entry=args.max_entry)

    folds, bag_seeds, cap, estop = args.folds, args.bag_seeds, 2000, 50
    if args.quick:
        rng = np.random.default_rng(0)
        pids = fit_base["player_id"].unique()
        keep = set(rng.choice(pids, size=max(200, len(pids) // 7),
                              replace=False))
        fit_base = fit_base[fit_base["player_id"].isin(keep)]
        folds, bag_seeds, cap, estop = 2, 2, 100, 20
        print(f"  [quick] {fit_base.player_id.nunique():,} players")

    # FULL-coverage raw attachment on the snap frames (one build per
    # (player,snap) key), carried through the (row,h) expansion.
    print(f"[exp5] attaching full-coverage raw features "
          f"(fit {len(fit_base):,} snaps, val {len(val_base):,} snaps)")
    fit_base = attach_raw_features(fit_base, args.db, keep_raw)
    val_base = attach_raw_features(val_base, args.db, keep_raw)
    print(f"  [{(time.time()-t0)/60:,.1f}m]")

    fit_long, Y_fit = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    print(f"  fit long: {len(fit_long):,} rows "
          f"[{(time.time()-t0)/60:,.1f}m]")
    X = fit_long[feats].values.astype(np.float32)

    fit_pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    uniq = np.unique(fit_pids)
    rng = np.random.default_rng(7)
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_mask = np.isin(fit_pids, list(es_players))

    print(f"[exp5] ES training (g3_slow recipe)")
    bst = train_one(X[~es_mask], Y_fit[~es_mask], feats, G3_SLOW, cap, estop,
                    args.seed, X[es_mask], Y_fit[es_mask])
    bi = int(bst.best_iteration)
    print(f"  best_iteration = {bi}  [{(time.time()-t0)/60:,.1f}m]")
    del bst

    print(f"[exp5] bagging x{bag_seeds} on 100% fit")
    bag = []
    for s in range(bag_seeds):
        bag.append(train_one(X, Y_fit, feats, G3_SLOW, bi + 1, 0,
                             args.seed + 100 + s))
        print(f"  seed {args.seed + 100 + s} done "
              f"[{(time.time()-t0)/60:,.1f}m]")
    with open(out_dir / "joint_xgb_exp5_bag.pkl", "wb") as fh:
        pickle.dump({"models": bag, "feature_names": feats,
                     "keep_raw": keep_raw, "events": list(EVENTS),
                     "h_max": H_MAX, "publish_h": PUBLISH_H,
                     "overrides": G3_SLOW, "num_rounds": bi + 1,
                     "raw_coverage": "full (joint2.attach_raw_features)",
                     "kind": "exp5_fullraw_bag"}, fh)

    print(f"[exp5] {folds}-fold cross-fit for calibration")
    fold_of = {p: i % folds for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in fit_pids])
    oof = np.full((len(fit_long), len(EVENTS)), np.nan)
    for f in range(folds):
        tr, ho = fold_idx != f, fold_idx == f
        b = train_one(X[tr], Y_fit[tr], feats, G3_SLOW, bi + 1, 0,
                      args.seed + 200 + f)
        oof[ho] = predict_rows(b, X[ho], feats)
        print(f"  fold {f} done [{(time.time()-t0)/60:,.1f}m]")
        del b

    cals: dict = {}
    era_ok = (fit_long["snap_year"].to_numpy() >= args.cal_min_snap_year)
    if args.cal_min_snap_year:
        print(f"  calibrators fit on snaps >= {args.cal_min_snap_year}: "
              f"{int(era_ok.sum()):,}/{len(fit_long):,} rows")
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
    with open(out_dir / "calibrators_hyip2.pkl", "wb") as fh:
        pickle.dump({"calibrators": cals, "events": list(EVENTS),
                     "h_max": H_MAX, "kind": "hyip2_logistic_oof"}, fh)
    del X, oof

    # ---- val eval (full-coverage features — matches deployment) ---------
    print(f"\n[exp5] scoring val")
    sv = sweep_val(bag, feats, val_base)
    for ev in EVENTS:
        cols = [f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]
        sv[cols] = np.maximum.accumulate(
            sv[cols].to_numpy(dtype=np.float64), axis=1)
    sv = apply_cal_trajectory(sv, cals)

    rows = []
    rows += per_h_metrics(sv, "", "G_raw(fullraw bag)")
    rows += per_h_metrics(sv, "cal_", "G_cal(OOF,honest)")
    met = pd.DataFrame(rows)
    f_csv = EXP4_DIR / "per_event_h_metrics.csv"
    if f_csv.exists():
        f_met = pd.read_csv(f_csv)
        f_met = f_met[f_met.scorer.isin(
            ["A_raw(v2.1c)", "F_raw(exp4 bag)", "F_cal(OOF,honest)"])]
        met = pd.concat([met, f_met], ignore_index=True)
    met.to_csv(out_dir / "per_event_h_metrics.csv", index=False)

    print(f"\n===== MLB_DEBUT (PRIMARY: h=3) =====")
    print(f"{'scorer':<24}{'h':>3}{'AP':>8}{'AUC':>8}{'calib':>7}")
    for scorer in met["scorer"].unique():
        for h in (1, 3, 6):
            r = met[(met.scorer == scorer) & (met.event == "MLB_DEBUT")
                    & (met.h == h)]
            if len(r):
                r = r.iloc[0]
                print(f"{scorer:<24}{h:>3}{r['ap']:>8.4f}{r['auc']:>8.4f}"
                      f"{r['calib']:>7.2f}")
    print(f"\n===== weighted AP @ h=6 =====")
    for scorer in met["scorer"].unique():
        sub_ = met[(met.scorer == scorer) & (met.h == PUBLISH_H)]
        print(f"  {scorer:<24} "
              f"{weighted_ap_at(sub_.to_dict(orient='records'), PUBLISH_H):.4f}")

    from prospects.model.train.exp_cdf_timing import cdf_timing as _cdt
    tim = _cdt(sv, "MLB_DEBUT", "cal_")
    est = {"G_cdf_mean(cal,OOF)": tim["t_mean"].to_numpy(),
           "G_cdf_median(cal,OOF)": tim["t_med"].to_numpy(),
           "lasso_timing.pkl": score_lasso_timing(val_base, _RUN.timing,
                                                  args.db)}
    windows = {"G_cdf_median(cal,OOF)": (tim["t_q25"].to_numpy(),
                                         tim["t_q75"].to_numpy())}
    trep = timing_report(val_base, est, windows)
    trep.to_csv(out_dir / "timing_metrics.csv", index=False)
    print(f"\n[exp5] timing (MLB_DEBUT)")
    print(trep.to_string(index=False))

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "quick": bool(args.quick), "best_iter": bi,
                   "timing": trep.to_dict(orient="records"),
                   "elapsed_min": round((time.time() - t0) / 60, 1)},
                  fh, indent=2, default=str)
    print(f"\n[exp5] wrote {out_dir}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
