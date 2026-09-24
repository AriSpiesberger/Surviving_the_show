"""Quick screen of ceiling-event model variants (2026-09-23).

Goal: AP on ESTABLISHED_MLB and STAR_PLUS_ELITE (and debut), not debut alone. One data load,
a seeded subsample of TRAINING players (default 25%, seed 7), the FULL held-out val set.

Arms (one dedicated GBM per event unless noted; learning rate 0.05 for speed, same for all):
  mcw{100,20,5,1}  min_child_weight sweep. XGBoost measures it in hessian units, p(1-p) per row,
                   so for an event at base rate ~0.4% a floor of 100 means ~25,000 rows per leaf:
                   production may barely be able to split on the players who become stars.
  spw10            mcw 5 + scale_pos_weight 10 (up-weight rare positives)
  cond             P(event by h) = P(debut by h) [deployed joint, calibrated]
                                   x P(event by h | debut by h) [GBM trained only on debut rows]
  cond_d5 / blend  combos of the round-1 winners (see ALL_ARMS comment)
  rank             rank:pairwise, one query group per (snap_year, h); order-only objective

Baseline = the production joint recipe retrained on the SAME sampled players (joint_small);
the deployed model (4x the data) is printed for reference only. cond uses joint_small's debut.
Every arm AND the baseline are put through the same calibration: an HYip2
calibrator 2-fold cross-fitted on val by player (fit on one half, applied to the other), so no
arm is favoured by its calibration. AP / AUC / calibration at h = 3 and 6, paired player
bootstrap vs the control.

    python -m prospects.model.train.exp_ceiling_grid
    python -m prospects.model.train.exp_ceiling_grid --events STAR_PLUS_ELITE --arms mcw5 cond
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
from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, _mono_string, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import sweep_val
from prospects.model.train.joint_xgb import _assemble, _prep_train

_RUN = config.run()
DB = str(config.model_db())
BASE = {"max_depth": 8, "colsample_bytree": 0.6, "learning_rate": 0.05}
ALL_ARMS = ["mcw100", "mcw20", "mcw5", "mcw1", "spw10", "cond", "rank"]
# combos (2026-09-23, after round 1: mcw5 and cond both beat the same-size joint on established)
#   cond_d5  cond, but P(debut) from a dedicated mcw5 debut model instead of joint_small
#   blend    mean of mcw5 and cond probabilities (needs both in --arms, listed before it)
#   blendjNN NN% cond + (100-NN)% same-size joint (after the full run: joint is strong at 100%)
DEBUT_ARMS = ["mcw100", "mcw20", "mcw5", "mcw1"]
EVAL_H = (3, 6)


# min_child_weight is an ABSOLUTE hessian sum, so on a fraction f of the players a floor of m
# is as strict as m/f on the full data. Round 1 of the quick screen (2026-09-23) used unscaled
# floors and overstated the mcw5 gain 4x: +0.035 AP on established at 25% vanished at 100%.
# Every arm's (and the baseline's) floor is therefore quoted at FULL-DATA scale and multiplied
# by the sample fraction here.
MCW_SCALE = 1.0


def train(X, y, feats, over, seed, Xes, yes, qid=None, qid_es=None):
    p = dict(BASE_PARAMS)
    p.update(BASE)
    p.update(over)
    if "min_child_weight" in p:
        p["min_child_weight"] = float(p["min_child_weight"]) * MCW_SCALE
    p["seed"] = seed
    p["monotone_constraints"] = _mono_string(feats)
    if qid is not None:
        o = np.argsort(qid, kind="stable")
        oe = np.argsort(qid_es, kind="stable")
        d = xgb.DMatrix(X[o], label=y[o], qid=qid[o], feature_names=feats)
        de = xgb.DMatrix(Xes[oe], label=yes[oe], qid=qid_es[oe], feature_names=feats)
        p.update({"objective": "rank:pairwise", "eval_metric": "map",
                  "lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 8})
    else:
        d = xgb.QuantileDMatrix(X, label=y, feature_names=feats)
        de = xgb.QuantileDMatrix(Xes, label=yes, feature_names=feats, ref=d)
        # logloss, not aucpr: on a 10% holdout with a few dozen positives aucpr is so noisy it
        # peaks by chance within ~15 rounds (smoke test 2026-09-23: 10-16 rounds for star+).
        p["eval_metric"] = "logloss"
        return xgb.train(p, d, num_boost_round=3000, evals=[(de, "es")], early_stopping_rounds=100,
                         verbose_eval=False)
    return xgb.train(p, d, num_boost_round=800, verbose_eval=False)  # rank: fixed rounds


def _n(b):
    return b.best_iteration + 1 if "best_iteration" in b.attributes() else b.num_boosted_rounds()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--cal", default=str(_RUN.scratch / "v24_build" / "calibrators_hyip2.pkl"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--sample-frac", type=float, default=0.25)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--events", nargs="*", default=["ESTABLISHED_MLB", "STAR_PLUS_ELITE", "MLB_DEBUT"])
    ap.add_argument("--arms", nargs="*", default=ALL_ARMS)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "exp_ceiling_grid"))
    args = ap.parse_args()
    global MCW_SCALE
    MCW_SCALE = args.sample_frac if args.sample_frac and args.sample_frac < 1 else 1.0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[grid] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bag = pickle.load(fh)
    with open(args.cal, "rb") as fh:
        cals = pickle.load(fh)["calibrators"]
    feats, keep_raw = list(bag["feature_names"]), list(bag["keep_raw"])

    def sample(df):
        if not args.sample_frac or args.sample_frac >= 1:
            return df
        u = np.sort(df["player_id"].unique())
        k = np.random.default_rng(args.sample_seed).choice(u, int(len(u) * args.sample_frac), replace=False)
        return df[df["player_id"].isin(k)]

    fit_base = _prep_train(sample(pd.read_csv(args.fit, low_memory=False)), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, sample(aug)], ignore_index=True)
    val = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    fit_base = attach_raw_features(fit_base, DB, keep_raw, verbose=False)
    val = attach_raw_features(val, DB, keep_raw, verbose=False)
    fl, Y = _assemble(fit_base, H_MAX)
    fl = stamp_extra_cols(fl)
    del fit_base
    X = fl[feats].values.astype(np.float32)
    h_arr = fl["h"].astype(int).to_numpy()
    qid_all = (fl["snap_year"].astype(int) * 100 + h_arr).to_numpy()
    pids = fl["player_id"].to_numpy()
    elig = {ev: ((fl[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in fl.columns
                 else np.ones(len(fl), bool)) for ev in EVENTS}
    del fl
    uniq = np.unique(pids)
    rng = np.random.default_rng(args.sample_seed)
    es = np.isin(pids, rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
    tick(f"fit rows {len(X):,} ({len(uniq):,} players), {len(feats)} feats; "
         f"val {val.player_id.nunique():,} players")

    # ---- val: control + features at the eval horizons ------------------------------------
    sv = sweep_val(bag["models"], feats, val)
    yip_v = sv["snap_offset"].to_numpy()
    yf = sv["years_fwd"].to_numpy()
    vpid = sv["player_id"].to_numpy()
    half = np.isin(vpid, np.random.default_rng(11).choice(np.unique(vpid), np.unique(vpid).size // 2,
                                                          replace=False))
    Xv = {h: stamp_extra_cols(add_cond_cols(val, h))[feats].values.astype(np.float32) for h in EVAL_H}

    def ctrl(ev):  # deployed joint, production calibration, monotone in h
        raw = np.maximum.accumulate(sv[[f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]]
                                    .to_numpy(dtype=np.float64), axis=1)
        if ev not in cals:
            return {h: raw[:, h - 1] for h in EVAL_H}
        P = np.column_stack([cals[ev].predict(raw[:, h - 1], np.full(len(raw), h), yip_v)
                             for h in range(1, H_MAX + 1)])
        P = np.maximum.accumulate(P, axis=1)
        return {h: P[:, h - 1] for h in EVAL_H}

    tick("deployed model scored (reference only: trained on 4x the data)")

    # Same-size baseline (user, 2026-09-23: "your baseline has to be the smaller baseline too,
    # otherwise more data will win on its own"). The production joint recipe — one multi-output
    # model over all 4 events, G3 params, logloss early stopping — on the SAME sampled players,
    # at the same learning rate as every arm. Every arm is judged against THIS model.
    pj = dict(BASE_PARAMS)
    pj.update(BASE)
    pj.update({"min_child_weight": 100.0 * MCW_SCALE, "seed": 42, "monotone_constraints": _mono_string(feats)})
    bj = xgb.train(pj, xgb.DMatrix(X[~es], label=Y[~es], feature_names=feats), num_boost_round=3000,
                   evals=[(xgb.DMatrix(X[es], label=Y[es], feature_names=feats), "es")],
                   early_stopping_rounds=100, verbose_eval=False)
    PJ = {h: bj.predict(xgb.DMatrix(Xv[h], feature_names=feats), iteration_range=(0, _n(bj))) for h in EVAL_H}
    tick(f"same-size joint baseline: {_n(bj)} rounds")
    del bj
    small_debut = {h: PJ[h][:, EVENTS.index("MLB_DEBUT")] for h in EVAL_H}
    _d5: dict = {}

    def debut5():  # dedicated mcw5 debut model, trained once on demand
        if not _d5:
            kd = EVENTS.index("MLB_DEBUT")
            md = elig["MLB_DEBUT"]
            yd = Y[:, kd].astype(np.float32)
            b5 = train(X[md & ~es], yd[md & ~es], feats, {"min_child_weight": 5.0}, 42, X[md & es], yd[md & es])
            _d5.update({h: b5.predict(xgb.DMatrix(Xv[h], feature_names=feats), iteration_range=(0, _n(b5)))
                        for h in EVAL_H})
            tick(f"dedicated mcw5 debut model: {_n(b5)} rounds")
        return _d5

    def evalset(ev, h):
        el = (sv[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in sv.columns else np.ones(len(sv), bool)
        m = (yf >= h) & el
        return m, np.asarray(realized_by_h(sv[m], ev, h), dtype=float)

    def xcal(ev, score):
        """2-fold player cross-fit HYip2 on val over both eval horizons -> calibrated dict."""
        rows = []
        for h in EVAL_H:
            m, yy = evalset(ev, h)
            idx = np.where(m)[0]
            rows.append((h, idx, yy))
        outp = {h: np.full(len(sv), np.nan) for h in EVAL_H}
        for side in (True, False):
            P, H, YP, YY = [], [], [], []
            for h, idx, yy in rows:
                sel = half[idx] == side
                P.append(score[h][idx][sel]); H.append(np.full(sel.sum(), h))
                YP.append(yip_v[idx][sel]); YY.append(yy[sel])
            c = HYip2Calibrator().fit(np.clip(np.concatenate(P), 1e-6, 1 - 1e-6), np.concatenate(H),
                                      np.concatenate(YP), np.concatenate(YY).astype(int))
            for h, idx, yy in rows:
                sel = half[idx] != side
                ii = idx[sel]
                outp[h][ii] = c.predict(np.clip(score[h][ii], 1e-6, 1 - 1e-6), np.full(len(ii), h), yip_v[ii])
        return outp

    def squash(s):  # rank scores -> (0,1) so the logit-based calibrator can take them
        return {h: 1 / (1 + np.exp(-np.asarray(v, dtype=np.float64))) for h, v in s.items()}

    rows, boots = [], []
    for ev in args.events:
        k = EVENTS.index(ev)
        y = Y[:, k].astype(np.float32)
        m = elig[ev]
        ctl = ctrl(ev)
        base_raw = {h: PJ[h][:, k] for h in EVAL_H}
        arms = {"deployed_4x": ctl, "joint_small": base_raw, "joint_small_xcal": xcal(ev, base_raw)}
        want = [a for a in args.arms if ev != "MLB_DEBUT" or a in DEBUT_ARMS]
        for arm in want:
            ta = time.time()
            if arm.startswith("mcw"):
                over = {"min_child_weight": float(arm[3:])}
                b = train(X[m & ~es], y[m & ~es], feats, over, 42, X[m & es], y[m & es])
                raw = {h: b.predict(xgb.DMatrix(Xv[h], feature_names=feats), iteration_range=(0, _n(b))) for h in EVAL_H}
            elif arm == "spw10":
                over = {"min_child_weight": 5.0, "scale_pos_weight": 10.0}
                b = train(X[m & ~es], y[m & ~es], feats, over, 42, X[m & es], y[m & es])
                raw = {h: b.predict(xgb.DMatrix(Xv[h], feature_names=feats), iteration_range=(0, _n(b))) for h in EVAL_H}
            elif arm == "cond":
                dm = m & (Y[:, EVENTS.index("MLB_DEBUT")] == 1)
                b = train(X[dm & ~es], y[dm & ~es], feats, {"min_child_weight": 5.0}, 42, X[dm & es], y[dm & es])
                raw = {h: small_debut[h] * b.predict(xgb.DMatrix(Xv[h], feature_names=feats),
                                                    iteration_range=(0, _n(b))) for h in EVAL_H}
            elif arm == "cond_d5":
                dm = m & (Y[:, EVENTS.index("MLB_DEBUT")] == 1)
                b = train(X[dm & ~es], y[dm & ~es], feats, {"min_child_weight": 5.0}, 42, X[dm & es], y[dm & es])
                d5 = debut5()
                raw = {h: d5[h] * b.predict(xgb.DMatrix(Xv[h], feature_names=feats),
                                            iteration_range=(0, _n(b))) for h in EVAL_H}
            elif arm == "blend":
                if "mcw5" not in arms or "cond" not in arms:
                    continue
                arms["blend"] = {h: 0.5 * (arms["mcw5"][h] + arms["cond"][h]) for h in EVAL_H}
                arms["blend_xcal"] = xcal(ev, arms["blend"])
                tick(f"{ev} blend")
                continue
            elif arm.startswith("blendj"):  # blendj / blendj30: w*cond + (1-w)*joint_small
                if "cond" not in arms:
                    continue
                w = float(arm[6:]) / 100 if len(arm) > 6 else 0.5
                arms[arm] = {h: w * arms["cond"][h] + (1 - w) * arms["joint_small"][h] for h in EVAL_H}
                arms[arm + "_xcal"] = xcal(ev, arms[arm])
                tick(f"{ev} {arm}")
                continue
            elif arm == "rank":
                b = train(X[m & ~es], y[m & ~es], feats, {"min_child_weight": 1.0}, 42, X[m & es], y[m & es],
                          qid=qid_all[m & ~es], qid_es=qid_all[m & es])
                raw = squash({h: b.predict(xgb.DMatrix(Xv[h], feature_names=feats), iteration_range=(0, _n(b))) for h in EVAL_H})
            else:
                continue
            arms[arm] = raw
            arms[arm + "_xcal"] = xcal(ev, raw)
            tick(f"{ev} {arm}: {_n(b)} rounds, {(time.time() - ta) / 60:.1f}m")
            del b

        for h in EVAL_H:
            mm, yy = evalset(ev, h)
            pl = vpid[mm]
            upl = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in upl}
            brng = np.random.default_rng(0)
            draws = [np.concatenate([pos[q] for q in brng.choice(upl, len(upl), replace=True)])
                     for _ in range(args.n_boot)]
            draws = [ix for ix in draws if yy[ix].sum()]
            b_raw, b_cal = arms["joint_small"][h][mm], arms["joint_small_xcal"][h][mm]
            for name, sc in arms.items():
                p = sc[h][mm]
                rows.append({"event": ev, "h": h, "arm": name, "n": int(mm.sum()), "pos": int(yy.sum()),
                             "ap": average_precision_score(yy, p), "auc": roc_auc_score(yy, p),
                             "calib": p.mean() / yy.mean()})
                if name not in ("joint_small", "joint_small_xcal"):
                    base = b_cal if name.endswith("_xcal") else b_raw
                    d = np.array([average_precision_score(yy[ix], p[ix]) - average_precision_score(yy[ix], base[ix])
                                  for ix in draws])
                    boots.append({"event": ev, "h": h, "arm": name, "d_ap": d.mean(),
                                  "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
        res = pd.DataFrame(rows)
        res.to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(boots).to_csv(out / "bootstrap.csv", index=False)
        tick(f"{ev} evaluated")

    pd.set_option("display.width", 220)
    res = pd.DataFrame(rows)
    bt = pd.DataFrame(boots)
    print("\n===== quick grid: held-out val (full), training on "
          f"{args.sample_frac:.0%} of players, seed {args.sample_seed} =====")
    print(res.round(4).to_string(index=False))
    print(f"\n===== AP(arm) - AP(same-size joint; raw vs raw, xcal vs xcal), paired player bootstrap ({args.n_boot}) =====")
    print(bt.round(4).to_string(index=False))
    tick("done")


if __name__ == "__main__":
    main()
