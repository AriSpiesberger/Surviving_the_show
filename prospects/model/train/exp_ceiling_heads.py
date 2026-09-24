"""Dedicated per-event models vs the joint model, in era (2026-09-23).

The user's goal is AP on every target — top-100, debut, established, star+ — not debut alone.
Production fits ONE XGBoost to all four events (4 outputs x 10 horizons) with a single
round count and early stopping chosen for that blend. A debut-only model beat it out of era,
plausibly because the rare events get the least of a shared budget. That argument is stronger
for established (base ~2%) and star+ (~0.4%), which are also the sheet's ceiling columns.

Arms, on the same fit / held-out val players, augmentation and HYip2 calibration protocol as
the deployed v2.4 (the control, loaded from disk):
  joint      deployed v2.4 (control)
  dedicated  one GBM per event on the SAME 251 production features, rounds chosen per event by
             early stopping (aucpr) on a 10% inner player holdout, then a 5-seed bag on 100%;
             per-event HYip2 calibrator on 3-fold cross-fit predictions.

Reported per event at h = 3 and 6 on held-out val: AP / AUC / calibration / ECE, plus a paired
player-bootstrap on AP vs the joint model.

    python -m prospects.model.train.exp_ceiling_heads --quick   # screen: 25% of fit players, 1 seed
    python -m prospects.model.train.exp_ceiling_heads           # full confirmation run

--quick subsamples TRAINING players with a fixed seed; the held-out val set is always used in
full (star+ has ~0.4% positives, so a smaller test set would make AP noise). The control is
the deployed full-data joint model, so a quick-mode win is conservative (dedicated has 1/4 the data).
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base, realized_by_h
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.exp_cal_topend import reliability
from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, _mono_string, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import sweep_val
from prospects.model.train.joint_xgb import _assemble, _prep_train

_RUN = config.run()
DB = str(config.model_db())
PARAMS = {"max_depth": 8, "min_child_weight": 100, "colsample_bytree": 0.6, "learning_rate": 0.03}


def fit(X, y, feats, rounds, seed, Xes=None, yes=None):
    p = dict(BASE_PARAMS)
    p.update(PARAMS)
    p["seed"] = seed
    p["monotone_constraints"] = _mono_string(feats)
    d = xgb.QuantileDMatrix(X, label=y, feature_names=feats)
    if Xes is None:
        return xgb.train(p, d, num_boost_round=rounds, verbose_eval=False)
    p["eval_metric"] = "logloss"  # aucpr on a small holdout stops rare events at ~15 rounds
    de = xgb.QuantileDMatrix(Xes, label=yes, feature_names=feats, ref=d)
    return xgb.train(p, d, num_boost_round=rounds, evals=[(de, "es")], early_stopping_rounds=80,
                     verbose_eval=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--cal", default=str(_RUN.scratch / "v24_build" / "calibrators_hyip2.pkl"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--cal-min-snap-year", type=int, default=2008)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--events", nargs="*", default=list(EVENTS))
    ap.add_argument("--quick", action="store_true", help="25%% of fit players, 1 seed, 200 bootstraps")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "exp_ceiling_heads"))
    args = ap.parse_args()
    if args.quick:
        args.sample_frac = args.sample_frac or 0.25
        args.seeds = 1
        args.out_dir = args.out_dir + "_quick"
    n_boot = 200 if args.quick else 300
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[heads] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bag = pickle.load(fh)
    with open(args.cal, "rb") as fh:
        cals = pickle.load(fh)["calibrators"]
    feats, keep_raw = list(bag["feature_names"]), list(bag["keep_raw"])

    fit_raw = pd.read_csv(args.fit, low_memory=False)
    if args.sample_frac:
        u = np.sort(fit_raw["player_id"].unique())
        keep = np.random.default_rng(args.sample_seed).choice(u, int(len(u) * args.sample_frac), replace=False)
        fit_raw = fit_raw[fit_raw["player_id"].isin(keep)]
        print(f"[heads] sample: {len(keep):,}/{len(u):,} fit players (seed {args.sample_seed})", flush=True)
    fit_base = _prep_train(fit_raw, DB, args.max_entry)
    del fit_raw
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        if args.sample_frac:
            ua = np.sort(aug["player_id"].unique())
            ka = np.random.default_rng(args.sample_seed).choice(ua, int(len(ua) * args.sample_frac), replace=False)
            aug = aug[aug["player_id"].isin(ka)]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    fit_base = attach_raw_features(fit_base, DB, keep_raw, verbose=False)
    val = attach_raw_features(val, DB, keep_raw, verbose=False)
    fit_long, Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    del fit_base
    X = fit_long[feats].values.astype(np.float32)
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    era_ok = fit_long["snap_year"].to_numpy() >= args.cal_min_snap_year
    pids = fit_long["player_id"].to_numpy()
    elig = {ev: ((fit_long[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in fit_long.columns
                 else np.ones(len(fit_long), bool)) for ev in EVENTS}
    del fit_long
    uniq = np.unique(pids)
    rng = np.random.default_rng(7)
    hold = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
    es = np.isin(pids, list(hold))
    fold_of = {p: i % 3 for i, p in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[p] for p in pids])
    tick(f"fit rows {len(X):,}, {len(feats)} feats; val {val.player_id.nunique():,} players")

    # control: deployed joint model, all events, one sweep
    sv = sweep_val(bag["models"], feats, val)
    yip_v = sv["snap_offset"].to_numpy()

    def traj(raw, cal, ev):
        raw = np.maximum.accumulate(raw, axis=1)
        P = np.column_stack([cal.predict(raw[:, h - 1], np.full(len(raw), h), yip_v)
                             for h in range(1, H_MAX + 1)])
        return np.maximum.accumulate(P, axis=1)

    preds = {}
    for ev in args.events:
        raw = sv[[f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]].to_numpy(dtype=np.float64)
        preds[("joint", ev)] = traj(raw, cals[ev], ev) if ev in cals else np.maximum.accumulate(raw, axis=1)
    tick("control scored")

    subs = [stamp_extra_cols(add_cond_cols(val, h))[feats].values.astype(np.float32)
            for h in range(1, H_MAX + 1)]
    rounds_used = {}
    for ev in args.events:
        k = EVENTS.index(ev)
        y = Y[:, k].astype(np.float32)
        m = elig[ev]
        if y[m].sum() < 50:
            tick(f"{ev}: too few positives, skipped")
            continue
        es_m = m & es & (h_arr == 6)
        b = fit(X[m & ~es], y[m & ~es], feats, 2000, 42, X[es_m], y[es_m])
        nr = int(b.best_iteration) + 1
        rounds_used[ev] = nr
        del b
        oof = np.full(len(y), np.nan)
        for f in range(3):
            tr, ho = m & (fold != f), m & (fold == f)
            b = fit(X[tr], y[tr], feats, nr, 300 + f)
            oof[ho] = b.predict(xgb.DMatrix(X[ho], feature_names=feats))
            del b
        ok = m & era_ok & np.isfinite(oof)
        cal = HYip2Calibrator().fit(oof[ok], h_arr[ok], yip_arr[ok], y[ok].astype(int))
        models = [fit(X[m], y[m], feats, nr, 1000 + s) for s in range(args.seeds)]
        raw = np.column_stack([np.mean([mm.predict(xgb.DMatrix(s_, feature_names=feats)) for mm in models], axis=0)
                               for s_ in subs])
        preds[("dedicated", ev)] = traj(raw, cal, ev)
        with open(out / f"head_{ev}.pkl", "wb") as fh:
            pickle.dump({"models": models, "calibrator": cal, "rounds": nr, "feature_names": feats}, fh)
        del models
        tick(f"{ev}: rounds {nr} (joint uses {bag.get('num_rounds')}), bagged + calibrated + scored")

    # ---- metrics + paired player bootstrap -------------------------------------------
    yf = sv["years_fwd"].to_numpy()
    rows, boot = [], {}
    rng = np.random.default_rng(0)
    for ev in args.events:
        if ("dedicated", ev) not in preds:
            continue
        el = (sv[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in sv.columns else np.ones(len(sv), bool)
        for h in (3, 6):
            m = (yf >= h) & el
            yy = np.asarray(realized_by_h(sv[m], ev, h), dtype=float)
            if yy.sum() < 5:
                continue
            for arm in ("joint", "dedicated"):
                p = preds[(arm, ev)][m, h - 1]
                rows.append({"event": ev, "h": h, "arm": arm, "n": int(m.sum()), "pos": int(yy.sum()),
                             "ap": average_precision_score(yy, p), "auc": roc_auc_score(yy, p),
                             "calib": p.mean() / yy.mean(), "ece": reliability(p, yy)})
            pl = sv.loc[m, "player_id"].to_numpy()
            upl = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in upl}
            pj, pd_ = preds[("joint", ev)][m, h - 1], preds[("dedicated", ev)][m, h - 1]
            d = []
            for _ in range(n_boot):
                ix = np.concatenate([pos[q] for q in rng.choice(upl, len(upl), replace=True)])
                if yy[ix].sum():
                    d.append(average_precision_score(yy[ix], pd_[ix]) - average_precision_score(yy[ix], pj[ix]))
            d = np.array(d)
            boot[(ev, h)] = (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n===== held-out val: joint (deployed) vs dedicated per-event models =====")
    print(res.round(4).to_string(index=False))
    print(f"\n===== AP(dedicated) - AP(joint), paired player-bootstrap ({n_boot}) =====")
    for (ev, h), (m_, lo, hi) in boot.items():
        print(f"  {ev:18s} h={h}  {m_:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\nrounds per event:", rounds_used, "| joint:", bag.get("num_rounds"))
    tick("done")


if __name__ == "__main__":
    main()
