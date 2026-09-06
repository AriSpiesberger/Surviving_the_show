"""EXPERIMENT 2: feature-expanded monotone XGB, HP screen, seed bag,
yip-aware OOF calibration. Headline metric: MLB_DEBUT AP @ h=3 (the buy-list
horizon), secondary the h=6 weighted composite.

Builds on exp_cdf_timing (experiment 1: monotone h + OOF h-covariate
calibration + CDF timing, which reached ranking parity with v2.1c and won on
timing/coherence). This round goes after the scores themselves:

  D-features: FEAT_COND plus signals already present in the OOF longs that
    v2.1c never fed the XGB —
      * hazard timing moments mean_t/sd_t for the 5 curve events (the hazard
        layer's own "when", not just "whether"),
      * p_ALL_STAR_ONCE / p_MAJOR_AWARD (collapsed probs of the two STAR
        components, individually),
      * horizon margins: h - mean_t_MLB_DEBUT, h - mean_t_ESTABLISHED_MLB and
        the debut z-score (h - mean_t)/sd_t — "is this horizon before or past
        the expected event year". Monotone-constrained +1 in h alongside
        h_centered so the trajectory stays coherent.

  HP screen: candidate configs trained on a 90/10 player split of FIT only,
    ranked by ES-slice debut AP@h3 (uniform mild optimism across candidates;
    honest numbers come from val at the end).

  Bag: winner refit on 100% of fit x 3 seeds, averaged.

  Calibration: 3-fold player-grouped cross-fit (winner HP, single seed) ->
    per-event logistic calibrator on [logit(p), h_c, yip_c, logit*h_c,
    logit*yip_c]. yip enters because per-yip debut calib in v2.1c runs
    0.87 (yip 0) -> 1.53 (yip 10): one global map can't fix a career-stage-
    dependent bias, a yip covariate can.

Baselines in the final table: A = stored v2.1c artifacts; C = experiment 1
metrics (read from runs/exp_cdf_timing/per_event_h_metrics.csv, not re-run).

Writes runs/exp_cdf_timing2/. Touches nothing under runs/current/.

Usage:
    python -m prospects.model.train.exp_cdf_timing2            # full
    python -m prospects.model.train.exp_cdf_timing2 --quick    # smoke test
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
from sklearn.metrics import average_precision_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import (
    EVENTS, FEAT_COND, H_MAX, PUBLISH_H, add_cond_cols, predict_trajectory,
    prep_base,
)
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import (
    EVENT_WEIGHTS, _logit, apply_baseline_calibrators, cdf_timing,
    inversion_stats, per_h_metrics, score_lasso_timing, timing_report,
    weighted_ap_at,
)

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_cdf_timing2"
EXP1_DIR = REPO_ROOT / "runs" / "exp_cdf_timing"

# ---- expanded feature set -------------------------------------------------
CURVE_EVS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
             "ELITE", "STAR"]
TIMING_FEATS = ([f"mean_t_{e}" for e in CURVE_EVS]
                + [f"sd_t_{e}" for e in CURVE_EVS])
EXTRA_PROBS = ["p_ALL_STAR_ONCE", "p_MAJOR_AWARD"]
MARGIN_FEATS = ["h_minus_mean_t_MLB_DEBUT", "h_minus_mean_t_ESTABLISHED_MLB",
                "z_h_debut"]
FEAT2 = list(FEAT_COND) + TIMING_FEATS + EXTRA_PROBS + MARGIN_FEATS
# Features that rise 1:1 with h within a snap — all must share the +1
# monotone constraint or the margins would let inversions back in.
MONO_UP = {"h_centered"} | set(MARGIN_FEATS)

MEAN_T_SENTINEL = 15.0   # mean_t is NaN when the event is ~impossible


def stamp_extra_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Fill/derive the exp2 feature columns on a frame that already has an
    `h` column (assembled long) or after add_cond_cols (inference sweep)."""
    out = df.copy()
    for c in TIMING_FEATS:
        if c not in out.columns:
            out[c] = MEAN_T_SENTINEL if c.startswith("mean_t") else 0.0
        elif c.startswith("mean_t"):
            out[c] = out[c].fillna(MEAN_T_SENTINEL)
        else:
            out[c] = out[c].fillna(0.0)
    for c in EXTRA_PROBS:
        out[c] = out[c].fillna(0.0) if c in out.columns else 0.0
    h = out["h"].astype(float)
    out["h_minus_mean_t_MLB_DEBUT"] = h - out["mean_t_MLB_DEBUT"]
    out["h_minus_mean_t_ESTABLISHED_MLB"] = h - out["mean_t_ESTABLISHED_MLB"]
    sd = out["sd_t_MLB_DEBUT"].astype(float).clip(lower=0.25)
    out["z_h_debut"] = (h - out["mean_t_MLB_DEBUT"]) / sd
    return out


def _mono_string(feats: list[str]) -> str:
    return "(" + ",".join("1" if f in MONO_UP else "0" for f in feats) + ")"


BASE_PARAMS = {
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 30,
    "reg_lambda": 1.0,
    "verbosity": 0,
}

# name -> (feats, param overrides, rounds cap, early stop)
CANDIDATES: dict[str, tuple[list, dict, int, int]] = {
    "c0_ctrl":   (list(FEAT_COND), {}, 800, 30),
    "f1_feats":  (FEAT2, {}, 800, 30),
    "h1_subcol": (FEAT2, {"subsample": 0.8, "colsample_bytree": 0.8}, 800, 30),
    "h2_deep":   (FEAT2, {"max_depth": 8, "min_child_weight": 100}, 800, 30),
    "h3_slow":   (FEAT2, {"learning_rate": 0.03, "subsample": 0.8,
                          "colsample_bytree": 0.8}, 2000, 50),
}


def train_one(X_tr, Y_tr, feats, overrides, rounds, estop, seed,
              X_es=None, Y_es=None):
    params = dict(BASE_PARAMS)
    params.update(overrides)
    params["seed"] = seed
    params["monotone_constraints"] = _mono_string(feats)
    dtr = xgb.DMatrix(X_tr, label=Y_tr, feature_names=feats)
    evals, kw = [(dtr, "train")], {}
    if X_es is not None:
        des = xgb.DMatrix(X_es, label=Y_es, feature_names=feats)
        evals.append((des, "es"))
        kw = dict(early_stopping_rounds=estop)
    return xgb.train(params, dtr, num_boost_round=rounds, evals=evals,
                     verbose_eval=False, **kw)


def predict_rows(bst, X, feats):
    return bst.predict(xgb.DMatrix(X, feature_names=feats))


def sweep_val(bsts: list, feats: list[str], df: pd.DataFrame,
              h_max: int = H_MAX) -> pd.DataFrame:
    """Trajectory sweep with a bag of boosters (averaged), exp2 features."""
    out = df.copy()
    preds_by_h = {}
    for h in range(1, h_max + 1):
        sub = stamp_extra_cols(add_cond_cols(df, h))
        X = sub[feats].values.astype(np.float32)
        P = np.mean([predict_rows(b, X, feats) for b in bsts], axis=0)
        preds_by_h[h] = P
    for k, ev in enumerate(EVENTS):
        M = np.column_stack([preds_by_h[h][:, k] for h in range(1, h_max + 1)])
        for hi, h in enumerate(range(1, h_max + 1)):
            out[f"xp_{ev}_h{h}"] = M[:, hi]
    return out


# ---- yip-aware calibrator -------------------------------------------------
class HYipCalibrator:
    """p_cal = sigmoid(b . [logit(p), h_c, yip_c, logit*h_c, logit*yip_c])."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=3000)

    @staticmethod
    def _feats(p, h, yip):
        lp = _logit(p)
        hc = np.asarray(h, dtype=np.float64) - 5.0
        yc = np.clip(np.asarray(yip, dtype=np.float64), 0, 10) - 3.0
        return np.column_stack([lp, hc, yc, lp * hc, lp * yc])

    def fit(self, p, h, yip, y):
        self.lr.fit(self._feats(p, h, yip), np.asarray(y, dtype=int))
        return self

    def predict(self, p, h, yip):
        return self.lr.predict_proba(self._feats(p, h, yip))[:, 1]


def apply_cal_trajectory(df: pd.DataFrame, cals: dict,
                         h_max: int = H_MAX) -> pd.DataFrame:
    yip = df["snap_offset"].to_numpy()
    out = df.copy()
    for ev in EVENTS:
        cal = cals.get(ev)
        cols = [f"xp_{ev}_h{h}" for h in range(1, h_max + 1)]
        M = out[cols].to_numpy(dtype=np.float64).copy()
        if cal is not None:
            for hi, h in enumerate(range(1, h_max + 1)):
                M[:, hi] = cal.predict(M[:, hi], np.full(len(out), h), yip)
        M = np.maximum.accumulate(M, axis=1)
        for hi, h in enumerate(range(1, h_max + 1)):
            out[f"cal_xp_{ev}_h{h}"] = M[:, hi]
    return out


def debut_ap_at(df_long_h, Y, P, h: int) -> float:
    m = (df_long_h == h)
    k = EVENTS.index("MLB_DEBUT")
    y = Y[m, k].astype(int)
    return float(average_precision_score(y, P[m, k])) if y.sum() else np.nan


def per_yip_debut_calib(val: pd.DataFrame, col: str, h: int) -> pd.DataFrame:
    from prospects.model.joint import realized_by_h
    rows = []
    d0 = val[(val["eligible_MLB_DEBUT"] == 1) & (val["years_fwd"] >= h)]
    for yip in range(11):
        d = d0[d0["snap_offset"] == yip]
        d = d[d[col].notna()]
        if len(d) < 50:
            continue
        y = realized_by_h(d, "MLB_DEBUT", h)
        if y.sum() < 5:
            continue
        p = d[col].astype(float).to_numpy()
        rows.append({"yip": yip, "n": len(d), "pos": int(y.sum()),
                     "calib": float(p.mean() / y.mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--db", default=DB)
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--bag-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[exp2] loading longs")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, args.max_entry)
    val_base = prep_base(pd.read_csv(args.val), args.db,
                         max_entry=args.max_entry)

    candidates = dict(CANDIDATES)
    folds, bag_seeds = args.folds, args.bag_seeds
    if args.quick:
        rng = np.random.default_rng(0)
        pids = fit_base["player_id"].unique()
        keep = set(rng.choice(pids, size=max(200, len(pids) // 7),
                              replace=False))
        fit_base = fit_base[fit_base["player_id"].isin(keep)]
        candidates = {k: (f, o, 100, 20) for k, (f, o, r, e)
                      in list(candidates.items())[:3]}
        folds, bag_seeds = 2, 2
        print(f"  [quick] {fit_base.player_id.nunique():,} players, "
              f"3 candidates, 100 rounds")

    fit_long, Y_fit = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    print(f"  fit long: {len(fit_long):,} rows / "
          f"{fit_long.player_id.nunique():,} players "
          f"[{time.time()-t0:,.0f}s]")

    fit_pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    uniq = np.unique(fit_pids)
    rng = np.random.default_rng(7)
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_mask = np.isin(fit_pids, list(es_players))

    X2 = fit_long[FEAT2].values.astype(np.float32)
    idx2 = {f: i for i, f in enumerate(FEAT2)}

    def _X(feats, mask):
        cols = [idx2[f] for f in feats]
        return X2[np.ix_(mask, cols)] if mask is not None else X2[:, cols]

    # ---- HP screen on the ES split -------------------------------------
    print(f"\n[exp2] screening {len(candidates)} candidates "
          f"(ES slice: {len(es_players):,} players)")
    print(f"{'cand':<12}{'rounds':>7}{'AP_h1':>8}{'AP_h2':>8}{'AP_h3':>8}"
          f"{'wAP_h6':>8}{'mins':>6}")
    screen = {}
    tr_m, es_m = ~es_mask, es_mask
    for name, (feats, over, rounds, estop) in candidates.items():
        tc = time.time()
        bst = train_one(_X(feats, tr_m), Y_fit[tr_m], feats, over, rounds,
                        estop, args.seed, _X(feats, es_m), Y_fit[es_m])
        bi = int(bst.best_iteration)
        P = predict_rows(bst, _X(feats, es_m), feats)
        h_es = h_arr[es_mask]
        ap1 = debut_ap_at(h_es, Y_fit[es_mask], P, 1)
        ap2 = debut_ap_at(h_es, Y_fit[es_mask], P, 2)
        ap3 = debut_ap_at(h_es, Y_fit[es_mask], P, 3)
        wrows = []
        for k, ev in enumerate(EVENTS):
            m6 = h_es == PUBLISH_H
            y = Y_fit[es_mask][m6, k].astype(int)
            apv = (float(average_precision_score(y, P[m6, k]))
                   if y.sum() else np.nan)
            wrows.append({"scorer": name, "event": ev, "h": PUBLISH_H,
                          "ap": apv})
        wap6 = weighted_ap_at(wrows, PUBLISH_H)
        screen[name] = {"best_iter": bi, "ap_h1": ap1, "ap_h2": ap2,
                        "ap_h3": ap3, "wap_h6": wap6,
                        "feats": feats, "overrides": over}
        print(f"{name:<12}{bi:>7}{ap1:>8.4f}{ap2:>8.4f}{ap3:>8.4f}"
              f"{wap6:>8.4f}{(time.time()-tc)/60:>6.1f}")
        del bst, P

    winner = max(screen, key=lambda k: screen[k]["ap_h3"])
    wcfg = screen[winner]
    print(f"\n[exp2] winner by debut AP@h3: {winner} "
          f"(rounds={wcfg['best_iter'] + 1})")

    # ---- bag winner on 100% fit ----------------------------------------
    feats, over = wcfg["feats"], wcfg["overrides"]
    nrounds = wcfg["best_iter"] + 1
    print(f"[exp2] bagging x{bag_seeds} seeds on 100% fit")
    bag = []
    for s in range(bag_seeds):
        bag.append(train_one(_X(feats, None), Y_fit, feats, over, nrounds,
                             0, args.seed + 100 + s))
        print(f"  seed {args.seed + 100 + s} done [{time.time()-t0:,.0f}s]")
    with open(out_dir / "joint_xgb_exp2_bag.pkl", "wb") as fh:
        pickle.dump({"models": bag, "feature_names": list(feats),
                     "events": list(EVENTS), "h_max": H_MAX,
                     "winner": winner, "overrides": over,
                     "num_rounds": nrounds,
                     "kind": "exp2_monotone_bag"}, fh)

    # ---- cross-fit -> yip-aware calibrators ----------------------------
    print(f"[exp2] {folds}-fold cross-fit for calibration")
    fold_of = {p: i % folds for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in fit_pids])
    oof = np.full((len(fit_long), len(EVENTS)), np.nan)
    for f in range(folds):
        tr, ho = fold_idx != f, fold_idx == f
        b = train_one(_X(feats, tr), Y_fit[tr], feats, over, nrounds, 0,
                      args.seed + 200 + f)
        oof[ho] = predict_rows(b, _X(feats, ho), feats)
        print(f"  fold {f} done [{time.time()-t0:,.0f}s]")
        del b

    cals: dict = {}
    for k, ev in enumerate(EVENTS):
        elig = (fit_long[f"eligible_{ev}"] == 1).to_numpy() \
            if f"eligible_{ev}" in fit_long.columns \
            else np.ones(len(fit_long), bool)
        ok = elig & np.isfinite(oof[:, k])
        y = Y_fit[ok, k].astype(int)
        if y.sum() < 25 or y.sum() == len(y):
            continue
        cals[ev] = HYipCalibrator().fit(oof[ok, k], h_arr[ok], yip_arr[ok], y)
        print(f"  {ev:<22} n={int(ok.sum()):,} pos={int(y.sum()):,} "
              f"coef={np.round(cals[ev].lr.coef_.ravel(), 3).tolist()}")
    with open(out_dir / "calibrators_hyip.pkl", "wb") as fh:
        pickle.dump({"calibrators": cals, "events": list(EVENTS),
                     "h_max": H_MAX, "kind": "hyip_logistic_oof"}, fh)

    # ---- val eval -------------------------------------------------------
    print(f"\n[exp2] scoring val ({len(val_base):,} snaps)")
    with open(_RUN.joint_xgb, "rb") as fh:
        bundle_a = pickle.load(fh)
    val_a = predict_trajectory(bundle_a, val_base)
    if _RUN.calibrators.exists():
        val_a = apply_baseline_calibrators(val_a, _RUN.calibrators)

    val_d_raw = sweep_val(bag, feats, val_base)          # no cummax yet
    inv_d = inversion_stats(val_d_raw, "")
    val_d = val_d_raw.copy()
    for ev in EVENTS:  # cummax for raw metrics
        cols = [f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]
        val_d[cols] = np.maximum.accumulate(
            val_d[cols].to_numpy(dtype=np.float64), axis=1)
    val_d = apply_cal_trajectory(val_d, cals)

    rows = []
    rows += per_h_metrics(val_a, "", "A_raw(v2.1c)")
    if _RUN.calibrators.exists():
        rows += per_h_metrics(val_a, "calA_", "A_cal(VAL-FIT,optim)")
    rows += per_h_metrics(val_d, "", "D_raw(bag)")
    rows += per_h_metrics(val_d, "cal_", "D_cal(OOF,honest)")
    met = pd.DataFrame(rows)
    # C from experiment 1, for the record
    exp1_csv = EXP1_DIR / "per_event_h_metrics.csv"
    if exp1_csv.exists():
        c_met = pd.read_csv(exp1_csv)
        c_met = c_met[c_met.scorer.isin(["C_raw(monotone)",
                                         "C_cal(OOF,honest)"])]
        c_met["scorer"] = c_met["scorer"].str.replace("C_", "C1_")
        met = pd.concat([met, c_met], ignore_index=True)
    met.to_csv(out_dir / "per_event_h_metrics.csv", index=False)

    print(f"\n===== MLB_DEBUT @ h=3 (PRIMARY) and h=1/2/6 =====")
    print(f"{'scorer':<24}{'h':>3}{'n':>7}{'AP':>8}{'AUC':>8}"
          f"{'Brier':>9}{'calib':>7}")
    for scorer in met["scorer"].unique():
        for h in (1, 2, 3, 6):
            r = met[(met.scorer == scorer) & (met.event == "MLB_DEBUT")
                    & (met.h == h)]
            if len(r):
                r = r.iloc[0]
                print(f"{scorer:<24}{h:>3}{r['n']:>7,.0f}{r['ap']:>8.4f}"
                      f"{r['auc']:>8.4f}{r['brier']:>9.4f}"
                      f"{r['calib']:>7.2f}")

    print(f"\n===== weighted AP @ h=6 =====")
    for scorer in met["scorer"].unique():
        sub = met[(met.scorer == scorer) & (met.h == PUBLISH_H)]
        wap = weighted_ap_at(sub.to_dict(orient="records"), PUBLISH_H)
        print(f"  {scorer:<24} {wap:.4f}")

    print(f"\n===== per-yip debut calib @ h=3 (want 1.00) =====")
    tabs = {}
    tabs["A_raw"] = per_yip_debut_calib(val_a, "xp_MLB_DEBUT_h3", 3)
    if _RUN.calibrators.exists():
        tabs["A_cal"] = per_yip_debut_calib(val_a, "calA_xp_MLB_DEBUT_h3", 3)
    tabs["D_cal"] = per_yip_debut_calib(val_d, "cal_xp_MLB_DEBUT_h3", 3)
    yips = sorted(set().union(*[set(t["yip"]) for t in tabs.values()]))
    print(f"{'yip':>4}" + "".join(f"{k:>10}" for k in tabs))
    for yip in yips:
        cells = []
        for k, t in tabs.items():
            r = t[t.yip == yip]
            cells.append(f"{r.iloc[0]['calib']:>10.2f}" if len(r)
                         else f"{'—':>10}")
        print(f"{yip:>4}" + "".join(cells))
    pd.concat([t.assign(scorer=k) for k, t in tabs.items()]
              ).to_csv(out_dir / "per_yip_debut_calib_h3.csv", index=False)

    print(f"\n===== inversions pre-cummax (D bag) =====")
    for ev, s in inv_d.items():
        print(f"  {ev:<22} frac={s['frac_snaps_with_inversion']:.3f} "
              f"mean_drop={s['mean_total_drop']:.4f}")

    # ---- timing ---------------------------------------------------------
    print(f"\n[exp2] timing (MLB_DEBUT)")
    tim_d = cdf_timing(val_d, "MLB_DEBUT", "cal_")
    lasso_pred = score_lasso_timing(val_base, _RUN.timing, args.db)
    est = {
        "D_cdf_mean(cal,OOF)": tim_d["t_mean"].to_numpy(),
        "D_cdf_median(cal,OOF)": tim_d["t_med"].to_numpy(),
        "lasso_timing.pkl": lasso_pred,
        "hazard_mean_t": (val_base["mean_t_MLB_DEBUT"].to_numpy()
                          if "mean_t_MLB_DEBUT" in val_base.columns
                          else None),
    }
    windows = {"D_cdf_median(cal,OOF)": (tim_d["t_q25"].to_numpy(),
                                         tim_d["t_q75"].to_numpy())}
    trep = timing_report(val_base, est, windows)
    trep.to_csv(out_dir / "timing_metrics.csv", index=False)
    print(trep.to_string(index=False))

    summary = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": bool(args.quick),
        "winner": winner,
        "screen": {k: {kk: vv for kk, vv in v.items()
                       if kk not in ("feats",)}
                   for k, v in screen.items()},
        "timing": trep.to_dict(orient="records"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[exp2] wrote {out_dir}  "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
