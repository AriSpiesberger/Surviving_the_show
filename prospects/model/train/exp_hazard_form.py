"""EXPERIMENT: hazard-form joint layer (proper discrete-time likelihood).

The v2.x joint layer predicts CUMULATIVE P(event by h) from (row,h) rows with
cumulative labels — a debut at year 2 is a positive at h=2..10 (the same
event counted up to 9x, correlated rows treated as independent), monotonicity
needs a constraint + cummax, and unresolved cells are dropped wholesale
(complete-case censoring that starves long horizons of recent eras).

H-form fixes the objective: per event, predict the CONDITIONAL hazard
  g_h = P(event at exactly snap+h | not by snap+h-1, features, h)
on the proper risk set (rows where the event hasn't fired before h, resolved
at h). That is the discrete-time survival likelihood — each event counted
once, each year's data contributing at the horizons it has resolved.
Cumulative F(h) = 1 - prod(1-g_j): monotone by construction, timing pmf = the
g's themselves.

Everything else matches v2.4 for a clean A/B: same FEAT2+raw160 features
(minus the monotone constraint — hazards may legitimately rise and fall in
h), recent-cohort augmentation, 2-seed bags per event, 2-fold cross-fit ->
HYip2 calibrators on the composed cumulative (2008+ snaps), clean val.

    python -m prospects.model.train.exp_hazard_form            # full
    python -m prospects.model.train.exp_hazard_form --quick    # smoke
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

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import (
    cdf_timing, per_h_metrics, score_lasso_timing, timing_report,
    weighted_ap_at,
)
from prospects.model.train.exp_cdf_timing2 import FEAT2, stamp_extra_cols

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_hazard_form"

G3_SLOW = {"max_depth": 8, "min_child_weight": 100,
           "colsample_bytree": 0.6, "learning_rate": 0.03}
# v2.4 reference on the clean val (from runs/v24_build.log)
V24_REF = {"deb_ap_h1": 0.4421, "deb_ap_h3": 0.6120, "deb_ap_h6": 0.6683,
           "wap_h6": 0.4759, "tim_mae": 1.034}


def train_haz(X_tr, y_tr, feats, rounds, estop, seed, X_es=None, y_es=None):
    params = dict(tree_method="hist", objective="binary:logistic",
                  eval_metric="logloss", seed=seed, verbosity=0, **G3_SLOW)
    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=feats)
    evals, kw = [(dtr, "t")], {}
    if X_es is not None:
        des = xgb.QuantileDMatrix(X_es, label=y_es, feature_names=feats,
                                  ref=dtr)
        evals.append((des, "es"))
        kw = dict(early_stopping_rounds=estop)
    return xgb.train(params, dtr, num_boost_round=rounds, evals=evals,
                     verbose_eval=False, **kw)


def risk_mask_and_label(long_df: pd.DataFrame, ev: str):
    """H-form risk set + label on the (row,h)-expanded long.
    In risk set at h iff the event hasn't fired before snap+h (i.e. trig is
    NaN or trig >= snap+h); label = fired at exactly snap+h. The long is
    already restricted to resolved cells (years_fwd >= h) and eligibility-
    gated upstream."""
    snap = long_df["snap_year"].astype(float).to_numpy()
    h = long_df["h"].astype(float).to_numpy()
    col = f"trigger_{ev}"
    trig = (pd.to_numeric(long_df[col], errors="coerce").to_numpy()
            if col in long_df.columns else np.full(len(long_df), np.nan))
    at_risk = np.isnan(trig) | (trig >= snap + h)
    y = (~np.isnan(trig)) & (trig == snap + h)
    if f"eligible_{ev}" in long_df.columns:
        at_risk &= (long_df[f"eligible_{ev}"] == 1).to_numpy()
    return at_risk, y.astype(np.int8)


def compose_cum(g_by_h: dict[int, np.ndarray], h_max: int) -> np.ndarray:
    """(n, h_max) cumulative from per-year hazards."""
    G = np.column_stack([np.clip(g_by_h[h], 0, 1)
                         for h in range(1, h_max + 1)])
    return 1.0 - np.cumprod(1.0 - G, axis=1)


def sweep_hazard(bsts_by_ev: dict, feats, df: pd.DataFrame,
                 h_max: int = H_MAX) -> pd.DataFrame:
    """Score a snap frame: per event, hazard at each h -> cumulative ->
    xp_<ev>_h{h} columns (v2.x schema)."""
    out = df.copy()
    X_by_h = {}
    for h in range(1, h_max + 1):
        sub = stamp_extra_cols(add_cond_cols(df, h))
        X_by_h[h] = sub[feats].values.astype(np.float32)
    new = {}
    for ev, bsts in bsts_by_ev.items():
        g = {}
        for h in range(1, h_max + 1):
            d = xgb.DMatrix(X_by_h[h], feature_names=list(feats))
            g[h] = np.mean([b.predict(d) for b in bsts], axis=0)
        F = compose_cum(g, h_max)
        for hi, h in enumerate(range(1, h_max + 1)):
            new[f"xp_{ev}_h{h}"] = F[:, hi]
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


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

    print(f"[hform] loading longs (+augmentation, v2.4 parity)")
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
        rounds_cap, estop = 100, 20
        args.bag_seeds = 1
        print(f"  [quick] {fit_base.player_id.nunique():,} players")

    fit_base = attach_raw_features(fit_base, args.db, keep_raw, verbose=False)
    fit_long, _Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    X = fit_long[feats].values.astype(np.float32)
    print(f"  expanded long: {len(fit_long):,} rows "
          f"[{(time.time()-t0)/60:.1f}m]")

    fpids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    snap_yr = fit_long["snap_year"].to_numpy()
    snap_key = (fit_long["player_id"].astype(str) + "@"
                + fit_long["snap_year"].astype(str)).to_numpy()
    uniq = np.unique(fpids)
    rng = np.random.default_rng(7)
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_m = np.isin(fpids, list(es_players))
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    hm = np.isin(fpids, list(half))

    # ---- per-event: ES rounds (debut only), 2-fold cross-fit, final bag --
    bsts: dict = {}
    oof_cum = {}          # honest composed-cumulative per (snap,h) for cal
    nrounds = None
    for ev in EVENTS:
        risk, y = risk_mask_and_label(fit_long, ev)
        n_pos = int(y[risk].sum())
        print(f"[hform] {ev}: risk rows {int(risk.sum()):,} "
              f"pos {n_pos:,} [{(time.time()-t0)/60:.1f}m]")
        if n_pos < 25:
            continue
        if nrounds is None:  # pick rounds once, on the debut-sized problem
            tr = risk & ~es_m
            es = risk & es_m
            b = train_haz(X[tr], y[tr], feats, rounds_cap, estop, args.seed,
                          X[es], y[es])
            nrounds = int(b.best_iteration) + 1
            print(f"  ES rounds = {nrounds}")
            del b
        # cross-fit: predict held-half hazards, compose per snap
        cum_parts = []
        for tr_m, ho_m in ((~hm, hm), (hm, ~hm)):
            b = train_haz(X[risk & tr_m], y[risk & tr_m], feats, nrounds,
                          0, args.seed)
            ho = ho_m  # predict ALL held rows (risk or not; mask later)
            d = xgb.DMatrix(X[ho], feature_names=feats)
            g = b.predict(d)
            cum_parts.append((ho, g))
            del b
        g_all = np.full(len(fit_long), np.nan, dtype=np.float64)
        for ho, g in cum_parts:
            g_all[ho] = g
        # compose cumulative per snap from per-h hazards (rows share snap_key)
        df_g = pd.DataFrame({"k": snap_key, "h": h_arr, "g": g_all})
        df_g = df_g.sort_values(["k", "h"])
        grp = df_g.groupby("k", sort=False)
        df_g["F"] = 1.0 - grp["g"].transform(
            lambda s: np.cumprod(1.0 - np.clip(s.to_numpy(), 0, 1)))
        F_rows = df_g.sort_index()["F"].to_numpy()
        oof_cum[ev] = F_rows
        # final bag on 100% of the risk set
        bsts[ev] = [train_haz(X[risk], y[risk], feats, nrounds, 0,
                              args.seed + 100 + s)
                    for s in range(args.bag_seeds)]
        print(f"  cross-fit + bag done [{(time.time()-t0)/60:.1f}m]")

    # ---- calibrators on honest composed cumulatives (2008+) -------------
    # NOTE: F here is cumulative over the row's OWN survival path; a snap's
    # (row,h) F is defined only where all steps j<=h were in the risk set —
    # rows after the event fired are excluded upstream, so cumprod over
    # present rows is the correct composition on resolved cells.
    from prospects.model.joint import realized_by_h
    cals: dict = {}
    era = snap_yr >= 2008
    for ev in EVENTS:
        if ev not in oof_cum:
            continue
        elig = (fit_long[f"eligible_{ev}"] == 1).to_numpy() \
            if f"eligible_{ev}" in fit_long.columns \
            else np.ones(len(fit_long), bool)
        snap = fit_long["snap_year"].astype(float).to_numpy()
        trig = pd.to_numeric(
            fit_long.get(f"trigger_{ev}", np.nan), errors="coerce"
        ).to_numpy()
        ycum = ((~np.isnan(trig)) & (trig > snap)
                & (trig <= snap + h_arr)).astype(int)
        ok = elig & era & np.isfinite(oof_cum[ev])
        if ycum[ok].sum() < 25:
            continue
        cals[ev] = HYip2Calibrator().fit(oof_cum[ev][ok], h_arr[ok],
                                         yip_arr[ok], ycum[ok])
    del X

    with open(out_dir / "hform_bags.pkl", "wb") as fh:
        pickle.dump({"bags": bsts, "feature_names": feats,
                     "keep_raw": keep_raw, "nrounds": nrounds,
                     "kind": "hform_per_event_hazard"}, fh)
    with open(out_dir / "calibrators_hyip2.pkl", "wb") as fh:
        pickle.dump({"calibrators": cals, "events": list(EVENTS),
                     "h_max": H_MAX, "kind": "hyip2_logistic_oof"}, fh)

    # ---- val eval --------------------------------------------------------
    print(f"\n[hform] scoring val [{(time.time()-t0)/60:.1f}m]")
    val_base = attach_raw_features(val_base, args.db, keep_raw, verbose=False)
    sv = sweep_hazard(bsts, feats, val_base)
    cal_new = {}
    yipv = sv["snap_offset"].to_numpy()
    for ev, cal in cals.items():
        cols = [f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]
        M = sv[cols].to_numpy(dtype=np.float64).copy()
        for hi, h in enumerate(range(1, H_MAX + 1)):
            M[:, hi] = cal.predict(M[:, hi], np.full(len(sv), h), yipv)
        M = np.maximum.accumulate(M, axis=1)
        for hi, h in enumerate(range(1, H_MAX + 1)):
            sv[f"cal_xp_{ev}_h{h}"] = M[:, hi]

    rows = per_h_metrics(sv, "", "H_raw(hazard-form)")
    rows += per_h_metrics(sv, "cal_", "H_cal(OOF,honest)")
    met = pd.DataFrame(rows)
    met.to_csv(out_dir / "per_event_h_metrics.csv", index=False)

    print(f"\n===== MLB_DEBUT vs v2.4 (clean val) =====")
    print(f"{'scorer':<22}{'h':>3}{'AP':>8}{'AUC':>8}{'calib':>7}")
    for scorer in met["scorer"].unique():
        for h in (1, 3, 6):
            r = met[(met.scorer == scorer) & (met.event == "MLB_DEBUT")
                    & (met.h == h)]
            if len(r):
                r = r.iloc[0]
                print(f"{scorer:<22}{h:>3}{r['ap']:>8.4f}{r['auc']:>8.4f}"
                      f"{r['calib']:>7.2f}")
    print(f"{'v2.4 ref':<22}  1{V24_REF['deb_ap_h1']:>8.4f}")
    print(f"{'v2.4 ref':<22}  3{V24_REF['deb_ap_h3']:>8.4f}")
    print(f"{'v2.4 ref':<22}  6{V24_REF['deb_ap_h6']:>8.4f}")
    for scorer in met["scorer"].unique():
        sub = met[(met.scorer == scorer) & (met.h == 6)]
        print(f"  wAP@6 {scorer:<22} "
              f"{weighted_ap_at(sub.to_dict(orient='records'), 6):.4f} "
              f"(v2.4 {V24_REF['wap_h6']:.4f})")

    tim = cdf_timing(sv, "MLB_DEBUT", "cal_")
    trep = timing_report(val_base, {
        "H_cdf_median(cal)": tim["t_med"].to_numpy(),
        "H_cdf_mean(cal)": tim["t_mean"].to_numpy(),
        "lasso_timing.pkl": score_lasso_timing(val_base, _RUN.timing,
                                               args.db),
    }, {"H_cdf_median(cal)": (tim["t_q25"].to_numpy(),
                              tim["t_q75"].to_numpy())})
    trep.to_csv(out_dir / "timing_metrics.csv", index=False)
    print(f"\n[hform] timing (v2.4 median MAE ref {V24_REF['tim_mae']:.3f}):")
    print(trep.to_string(index=False))

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "quick": bool(args.quick), "nrounds": nrounds,
                   "v24_ref": V24_REF,
                   "timing": trep.to_dict(orient="records"),
                   "elapsed_min": round((time.time() - t0) / 60, 1)},
                  fh, indent=2, default=float)
    print(f"\n[hform] wrote {out_dir} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
