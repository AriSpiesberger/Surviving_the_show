"""v3 recipe for every event, in era (2026-09-23).

v3 (debut) = single-stage GBM on the raw panel features (set "B") + the GRU encoder's 6
multi-horizon debut logits, stacked out-of-fold, 5-seed bag, HYip2 calibrator on 3-fold
cross-fit predictions. It beat deployed v2.4 on debut in era (AP@3 +0.020, calibration
0.88 -> 0.98) and out of era (+0.054). This asks whether the same recipe lifts the other
targets the sheet shows: TOP_100_PROSPECT, ESTABLISHED_MLB, STAR_PLUS_ELITE.

Control: at full size, the deployed v2.4 joint + its HYip2 calibrator, loaded from disk. In a
quick screen (--sample-frac < 1) the control is the SAME joint recipe retrained on the same
sampled players (scaled mcw, 3-fold cross-fit HYip2) - otherwise more data wins on its own.

Arms per event:
  v3        event-specific v3 model
  v3cond    (EST / STAR) P(debut by h) [v3 debut, raw] x P(event by h | debut by h) [GBM on
            debut rows], calibrated as a product on 3-fold cross-fit predictions
  mix_v3    0.5 control + 0.5 v3        (calibrated probabilities)
  mix_cond  0.5 control + 0.5 v3cond    (EST / STAR)

--sample-frac subsamples TRAINING players (seed --sample-seed) for quick screens and scales
min_child_weight by the same fraction (an absolute hessian floor; unscaled, a 25% screen tests
a 4x stricter regime — the 2026-09-23 lesson). Val is always complete.

    python -m prospects.model.train.exp_v3_all --sample-frac 0.25     # quick screen
    python -m prospects.model.train.exp_v3_all                        # full
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
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base, realized_by_h
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.exp_cal_topend import reliability
from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, _mono_string, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import sweep_val
from prospects.model.train.exp_macro_bc import feature_sets
from prospects.model.train.exp_seq_d import EMB, SEQ_OUT
from prospects.model.train.joint_xgb import _assemble, _prep_train

_RUN = config.run()
DB = str(config.model_db())
DEBUT = "MLB_DEBUT"
EMB_COLS = [f"seq_emb_{i}" for i in range(EMB)]
PARAMS = {"max_depth": 8, "min_child_weight": 100, "colsample_bytree": 0.6, "learning_rate": 0.03}
ROUNDS = 340
EVAL_H = (3, 6)
CEILING = ("ESTABLISHED_MLB", "STAR_PLUS_ELITE")


def fit(X, y, feats, seed, mcw_scale):
    p = dict(BASE_PARAMS)
    p.update(PARAMS)
    p["min_child_weight"] = PARAMS["min_child_weight"] * mcw_scale
    p["seed"] = seed
    p["monotone_constraints"] = _mono_string(feats)
    d = xgb.QuantileDMatrix(X, label=y, feature_names=feats)
    return xgb.train(p, d, num_boost_round=ROUNDS, verbose_eval=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--cal", default=str(_RUN.scratch / "v24_build" / "calibrators_hyip2.pkl"))
    ap.add_argument("--emb-cache", nargs="+", default=[str(REPO_ROOT / "runs" / "exp_v3_inera" / f"embeddings_{SEQ_OUT}.npz")],
                    help="one or more embedding caches; several are concatenated as features")
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--cal-min-snap-year", type=int, default=2008)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--events", nargs="*", default=list(("TOP_100_PROSPECT", DEBUT) + CEILING))
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--depth", type=int, default=None, help="override max_depth for every v3 component")
    ap.add_argument("--rounds", type=int, default=None, help="override the round count for every v3 component")
    ap.add_argument("--haz", action="store_true", help="add the discrete-time hazard arms (haz, mix_haz)")
    ap.add_argument("--only-haz", action="store_true", help="hazard arms only (skip v3 / v3cond)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "exp_v3_all"))
    args = ap.parse_args()
    global ROUNDS
    if args.depth:
        PARAMS["max_depth"] = args.depth
    if args.rounds:
        ROUNDS = args.rounds
    frac = args.sample_frac if 0 < args.sample_frac < 1 else 1.0
    out = Path(args.out_dir + ("" if frac == 1.0 else f"_q{int(frac * 100)}"))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[v3all] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bagA = pickle.load(fh)
    with open(args.cal, "rb") as fh:
        calsA = pickle.load(fh)["calibrators"]
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in bagA["keep_raw"] if c in live]
    fs = feature_sets(keep_raw)
    featsA = list(bagA["feature_names"])
    zs_emb = [np.load(f, allow_pickle=True) for f in args.emb_cache]
    emb_cols = [f"seq_emb_{j}_{i}" if len(zs_emb) > 1 else f"seq_emb_{i}"
                for j, z in enumerate(zs_emb) for i in range(z["emb"].shape[1])]
    feats = fs["B"] + emb_cols
    raw_all = sorted(set(keep_raw) | (set(fs["B"]) & live))

    def sample(df):
        if frac == 1.0:
            return df
        u = np.sort(df["player_id"].unique())
        k = np.random.default_rng(args.sample_seed).choice(u, int(len(u) * frac), replace=False)
        return df[df["player_id"].isin(k)]

    fit_base = _prep_train(sample(pd.read_csv(args.fit, low_memory=False)), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", DEBUT):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, sample(aug)], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    fit_base = attach_raw_features(fit_base, DB, raw_all, verbose=False)
    val_base = attach_raw_features(val_base, DB, raw_all, verbose=False)
    assert not (set(fit_base.player_id) & set(val_base.player_id)), "fit/val player overlap"

    # embeddings: out-of-fold for fit rows, full encoder for val rows (exp_v3_inera cache)
    emb_df = None
    off = 0
    for z in zs_emb:
        w = z["emb"].shape[1]
        part = pd.concat([pd.DataFrame({"player_id": z["pid"], "snap_year": z["snap"].astype(int)}),
                          pd.DataFrame(z["emb"], columns=emb_cols[off:off + w])], axis=1)
        part = part.drop_duplicates(["player_id", "snap_year"])
        emb_df = part if emb_df is None else emb_df.merge(part, on=["player_id", "snap_year"], how="inner")
        off += w
    fit_base = fit_base.merge(emb_df, on=["player_id", "snap_year"], how="left")
    val_base = val_base.merge(emb_df, on=["player_id", "snap_year"], how="left")
    cov_f = fit_base[emb_cols[0]].notna().mean()
    cov_v = val_base[emb_cols[0]].notna().mean()
    tick(f"fit {fit_base.player_id.nunique():,} players (frac {frac}), val {val_base.player_id.nunique():,}; "
         f"embedding coverage fit {cov_f:.1%} / val {cov_v:.1%}")
    if cov_f < 0.98 or cov_v < 0.98:
        raise SystemExit("embedding cache does not cover this split; rerun exp_v3_inera to rebuild it")

    fl, Y = _assemble(fit_base, H_MAX)
    fl = stamp_extra_cols(fl)
    del fit_base
    X = fl[feats].values.astype(np.float32)
    XA = fl[featsA].values.astype(np.float32) if frac < 1.0 else None
    h_arr = fl["h"].astype(int).to_numpy()
    yip_arr = fl["snap_offset"].to_numpy()
    era_ok = fl["snap_year"].to_numpy() >= args.cal_min_snap_year
    snap_arr = fl["snap_year"].to_numpy()
    pids = fl["player_id"].to_numpy()
    elig = {ev: ((fl[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in fl.columns
                 else np.ones(len(fl), bool)) for ev in EVENTS}
    del fl
    uniq = np.unique(pids)
    rng = np.random.default_rng(7)
    fold_of = {p: i % 3 for i, p in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[p] for p in pids])
    tick(f"{len(X):,} fit rows, {len(feats)} features")

    # val design matrices per horizon, control trajectories
    subs = [stamp_extra_cols(add_cond_cols(val_base, h))[feats].values.astype(np.float32)
            for h in range(1, H_MAX + 1)]
    sv = sweep_val(bagA["models"], featsA, val_base)
    yip_v = sv["snap_offset"].to_numpy()
    yf = sv["years_fwd"].to_numpy()
    vpid = sv["player_id"].to_numpy()

    def calibrate(raw, cal):
        raw = np.maximum.accumulate(raw, axis=1)
        P = np.column_stack([cal.predict(np.clip(raw[:, h - 1], 1e-7, 1 - 1e-7), np.full(len(raw), h), yip_v)
                             for h in range(1, H_MAX + 1)])
        return np.maximum.accumulate(P, axis=1)

    if frac < 1.0:
        # same-size control: the deployed joint recipe (exp5: multi-output, G3 params, the
        # deployed round count) on the sampled players, HYip2 on 3-fold cross-fit predictions
        pj = dict(BASE_PARAMS)
        pj.update(bagA.get("overrides", PARAMS))
        pj["min_child_weight"] = float(pj.get("min_child_weight", 100)) * frac
        pj["monotone_constraints"] = _mono_string(featsA)
        nr = int(bagA.get("num_rounds", 295))

        def fitj(rows, seed):
            q = dict(pj, seed=seed)
            return xgb.train(q, xgb.DMatrix(XA[rows], label=Y[rows], feature_names=featsA),
                             num_boost_round=nr, verbose_eval=False)
        oofj = np.full(Y.shape, np.nan)
        for f in range(3):
            b = fitj(fold != f, 700 + f)
            oofj[fold == f] = b.predict(xgb.DMatrix(XA[fold == f], feature_names=featsA))
            del b
        bj = fitj(np.ones(len(Y), bool), 777)
        rawj = {h: bj.predict(xgb.DMatrix(stamp_extra_cols(add_cond_cols(val_base, h))[featsA]
                                          .values.astype(np.float32), feature_names=featsA))
                for h in range(1, H_MAX + 1)}
        del bj, XA
        calsJ = {}
        for ev in EVENTS:
            k = EVENTS.index(ev)
            ok = elig[ev] & era_ok & np.isfinite(oofj[:, k])
            if Y[ok, k].sum() >= 25:
                calsJ[ev] = HYip2Calibrator().fit(oofj[ok, k], h_arr[ok], yip_arr[ok], Y[ok, k].astype(int))
        tick(f"same-size joint control trained ({nr} rounds, mcw {pj['min_child_weight']:g})")

    def control(ev):
        if frac < 1.0:
            k = EVENTS.index(ev)
            raw = np.column_stack([rawj[h][:, k] for h in range(1, H_MAX + 1)]).astype(np.float64)
            return calibrate(raw, calsJ[ev]) if ev in calsJ else np.maximum.accumulate(raw, axis=1)
        raw = sv[[f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]].to_numpy(dtype=np.float64)
        return calibrate(raw, calsA[ev]) if ev in calsA else np.maximum.accumulate(raw, axis=1)

    def train_event(y, m, tag):
        """3-fold OOF raw predictions on all rows + bagged val raw trajectory."""
        oof = np.full(len(y), np.nan)
        for f in range(3):
            tr = m & (fold != f)
            b = fit(X[tr], y[tr], feats, 300 + f, frac)
            ho = fold == f
            oof[ho] = b.predict(xgb.DMatrix(X[ho], feature_names=feats))
            del b
        bag = [fit(X[m], y[m], feats, 1100 + s, frac) for s in range(args.seeds)]
        rawv = np.column_stack([np.mean([b.predict(xgb.DMatrix(s_, feature_names=feats)) for b in bag], axis=0)
                                for s_ in subs])
        with open(out / f"bag_{tag}.pkl", "wb") as fh:
            pickle.dump({"models": bag, "feature_names": feats, "params": PARAMS, "rounds": ROUNDS,
                         "mcw_scale": frac}, fh)
        return oof, rawv

    def fit_cal(score, y, m):
        ok = m & era_ok & np.isfinite(score)
        return HYip2Calibrator().fit(np.clip(score[ok], 1e-7, 1 - 1e-7), h_arr[ok], yip_arr[ok], y[ok].astype(int))

    # ---- discrete-time hazard form (architecture A, 2026-09-23) ----------------------------
    # Row (player, snap, h) -> label "event happens IN year h" on rows still at risk (not yet
    # happened by h-1). P(by h) = 1 - prod_{t<=h} (1 - q_t). Each positive is counted once, in
    # its year, instead of being repeated at every later horizon of the cumulative form.
    kdf = pd.DataFrame({"pid": pids, "snap": snap_arr, "h": h_arr, "i": np.arange(len(pids))})
    prv = kdf.assign(h=kdf.h + 1).rename(columns={"i": "iprev"})[["pid", "snap", "h", "iprev"]]
    iprev = kdf.merge(prv, on=["pid", "snap", "h"], how="left")["iprev"].to_numpy()
    has_prev = np.isfinite(iprev)
    ip = np.where(has_prev, iprev, 0).astype(int)

    def hazard_arm(y_cum, m, tag):
        y_prev = np.where(has_prev, y_cum[ip], 0.0)
        at_risk = (h_arr == 1) | (has_prev & (y_prev == 0))
        y_h = (y_cum - y_prev).astype(np.float32)
        tr_m = m & at_risk
        q = np.full(len(y_cum), np.nan)
        for f in range(3):
            b = fit(X[tr_m & (fold != f)], y_h[tr_m & (fold != f)], feats, 800 + f, frac)
            q[fold == f] = b.predict(xgb.DMatrix(X[fold == f], feature_names=feats))
            del b
        surv = np.full(len(q), np.nan)
        for h in range(1, H_MAX + 1):
            idx = np.where(h_arr == h)[0]
            prev_s = np.ones(len(idx)) if h == 1 else np.where(has_prev[idx], surv[ip[idx]], np.nan)
            surv[idx] = (1 - q[idx]) * prev_s
        cal = fit_cal(1 - surv, y_cum, m)
        bag = [fit(X[tr_m], y_h[tr_m], feats, 1500 + s, frac) for s in range(args.seeds)]
        qv = np.column_stack([np.mean([b.predict(xgb.DMatrix(s_, feature_names=feats)) for b in bag], axis=0)
                              for s_ in subs])
        del bag
        return calibrate(1 - np.cumprod(1 - qv, axis=1), cal)

    kd = EVENTS.index(DEBUT)
    yd = Y[:, kd].astype(np.float32)
    debut_oof, debut_raw = None, None
    preds: dict = {}
    order = [DEBUT] + [e for e in args.events if e != DEBUT]   # debut first: v3cond needs it
    for ev in order:
        k = EVENTS.index(ev)
        y = Y[:, k].astype(np.float32)
        m = elig[ev]
        preds[(ev, "control")] = control(ev)
        if args.haz:
            preds[(ev, "haz")] = hazard_arm(y, m, ev)
            preds[(ev, "mix_haz")] = 0.5 * preds[(ev, "control")] + 0.5 * preds[(ev, "haz")]
            tick(f"{ev}: hazard form done")
            if args.only_haz:
                continue
        oof, rawv = train_event(y, m, ev)
        if ev == DEBUT:
            debut_oof, debut_raw = oof, rawv
        preds[(ev, "v3")] = calibrate(rawv, fit_cal(oof, y, m))
        preds[(ev, "mix_v3")] = 0.5 * preds[(ev, "control")] + 0.5 * preds[(ev, "v3")]
        tick(f"{ev}: v3 done")
        if ev in CEILING:
            md = m & (yd == 1)
            oofc = np.full(len(y), np.nan)
            for f in range(3):
                tr = md & (fold != f)
                b = fit(X[tr], y[tr], feats, 500 + f, frac)
                ho = fold == f
                oofc[ho] = b.predict(xgb.DMatrix(X[ho], feature_names=feats))
                del b
            bagc = [fit(X[md], y[md], feats, 1300 + s, frac) for s in range(args.seeds)]
            rawc = np.column_stack([np.mean([b.predict(xgb.DMatrix(s_, feature_names=feats)) for b in bagc], axis=0)
                                    for s_ in subs])
            with open(out / f"bag_{ev}_cond.pkl", "wb") as fh:
                pickle.dump({"models": bagc, "feature_names": feats, "params": PARAMS, "rounds": ROUNDS,
                             "mcw_scale": frac, "given": DEBUT}, fh)
            del bagc
            cal = fit_cal(debut_oof * oofc, y, m)
            preds[(ev, "v3cond")] = calibrate(debut_raw * rawc, cal)
            preds[(ev, "mix_cond")] = 0.5 * preds[(ev, "control")] + 0.5 * preds[(ev, "v3cond")]
            tick(f"{ev}: v3cond done")

    # ---- metrics + paired player bootstrap vs control ----------------------------------
    rows, boots = [], []
    brng = np.random.default_rng(0)
    for ev in order:
        if ev not in args.events:
            continue
        el = (sv[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in sv.columns else np.ones(len(sv), bool)
        for h in EVAL_H:
            mm = (yf >= h) & el
            yy = np.asarray(realized_by_h(sv[mm], ev, h), dtype=float)
            if yy.sum() < 5:
                continue
            pl = vpid[mm]
            upl = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in upl}
            draws = [np.concatenate([pos[q] for q in brng.choice(upl, len(upl), replace=True)])
                     for _ in range(args.n_boot)]
            draws = [ix for ix in draws if yy[ix].sum()]
            base = preds[(ev, "control")][mm, h - 1]
            for (e2, arm), P in preds.items():
                if e2 != ev:
                    continue
                p = P[mm, h - 1]
                rows.append({"event": ev, "h": h, "arm": arm, "n": int(mm.sum()), "pos": int(yy.sum()),
                             "ap": average_precision_score(yy, p), "auc": roc_auc_score(yy, p),
                             "calib": p.mean() / yy.mean(), "ece": reliability(p, yy)})
                if arm != "control":
                    d = np.array([average_precision_score(yy[ix], p[ix]) - average_precision_score(yy[ix], base[ix])
                                  for ix in draws])
                    boots.append({"event": ev, "h": h, "arm": arm, "d_ap": d.mean(),
                                  "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    res, bt = pd.DataFrame(rows), pd.DataFrame(boots)
    res.to_csv(out / "metrics.csv", index=False)
    bt.to_csv(out / "bootstrap.csv", index=False)
    np.savez_compressed(out / "val_preds.npz", pid=vpid, snap_year=sv["snap_year"].to_numpy(), yip=yip_v,
                        **{f"{e}__{a}": P for (e, a), P in preds.items()})
    pd.set_option("display.width", 220)
    print(f"\n===== held-out val (full); training frac {frac}, seed {args.sample_seed} =====")
    print(res.round(4).to_string(index=False))
    print(f"\n===== AP(arm) - AP({'same-size joint' if frac < 1 else 'deployed v2.4'}), paired player bootstrap ({args.n_boot}) =====")
    print(bt.round(4).to_string(index=False))
    tick("done")


if __name__ == "__main__":
    main()
