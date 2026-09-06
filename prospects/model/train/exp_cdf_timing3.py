"""EXPERIMENT 3: un-bottleneck the refinement head with raw panel features.

The conditional XGB refines the hazard trajectory while seeing almost none of
the evidence the hazards saw: of the 327 raw landmark features (age-vs-level,
level-adjusted rates, trajectory deltas, 76 point-in-time scouting columns),
FEAT_COND carries only a 5-col scouting summary + 3 acquisition flags. This
experiment joins the raw snap vector from the OOF panel cache
(runs/current/scratch/oof/panel_cache.npz, keyed by (player_id, landmark S))
onto every (row,h) pair so the head can do residual correction against the
hazard layer's verdict.

Stages:
  0. Gain-based selection: quick XGB on a subsample over [exp2-winner feats +
     ALL raw features]; keep the top --top-k raw features by total gain
     (memory: the full join would be ~2.3M x 450 cols).
  1. Screen: exp2-winner HP + rawK, and a colsample-0.6 variant, on the SAME
     internal ES split as exp2 (rng seed 7 over sorted pids -> identical
     players), judged by debut AP@h3. Exp2's screen numbers are directly
     comparable and serve as control.
  2. Bag winner x3 seeds on 100% fit; 3-fold cross-fit -> yip-aware
     calibrator (HYipCalibrator from exp2).
  3. Val eval vs A (v2.1c) and exp2's D (read from its metrics CSV).

Raw columns are prefixed "rw_" (names like years_in_pro collide with
FEAT_BASE). Rows with no panel-cache match keep NaN (XGB routes missing).

Writes runs/exp_cdf_timing3/. Touches nothing under runs/current/.

Usage:
    python -m prospects.model.train.exp_cdf_timing3            # full
    python -m prospects.model.train.exp_cdf_timing3 --quick    # smoke test
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
from sklearn.metrics import average_precision_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import (
    EVENTS, H_MAX, PUBLISH_H, predict_trajectory, prep_base,
)
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import (
    apply_baseline_calibrators, cdf_timing, inversion_stats, per_h_metrics,
    score_lasso_timing, timing_report, weighted_ap_at,
)
from prospects.model.train.exp_cdf_timing2 import (
    BASE_PARAMS, CANDIDATES, FEAT2, HYipCalibrator, MONO_UP,
    apply_cal_trajectory, debut_ap_at, per_yip_debut_calib, stamp_extra_cols,
    sweep_val,
)

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_cdf_timing3"
EXP2_DIR = REPO_ROOT / "runs" / "exp_cdf_timing2"
PANEL_CACHE = REPO_ROOT / "runs" / "current" / "scratch" / "oof" / "panel_cache.npz"


def _mono_string(feats: list[str]) -> str:
    return "(" + ",".join("1" if f in MONO_UP else "0" for f in feats) + ")"


def train_one(X_tr, Y_tr, feats, overrides, rounds, estop, seed,
              X_es=None, Y_es=None):
    params = dict(BASE_PARAMS)
    params.update(overrides)
    params["seed"] = seed
    params["monotone_constraints"] = _mono_string(feats)
    dtr = xgb.QuantileDMatrix(X_tr, label=Y_tr, feature_names=feats)
    evals, kw = [(dtr, "train")], {}
    if X_es is not None:
        des = xgb.QuantileDMatrix(X_es, label=Y_es, feature_names=feats,
                                  ref=dtr)
        evals.append((des, "es"))
        kw = dict(early_stopping_rounds=estop)
    return xgb.train(params, dtr, num_boost_round=rounds, evals=evals,
                     verbose_eval=False, **kw)


def predict_rows(bst, X, feats):
    return bst.predict(xgb.DMatrix(X, feature_names=feats))


def load_panel_map():
    d = np.load(PANEL_CACHE, allow_pickle=True)
    X_lm = d["X_lm"]
    pids = d["pids"]
    S = d["S_yrs"]
    if X_lm.shape[1] != len(FEATURE_NAMES):
        raise SystemExit(
            f"panel cache is {X_lm.shape[1]} wide, FEATURE_NAMES is "
            f"{len(FEATURE_NAMES)} — contract drift, rebuild the cache")
    key_to_row = {(p, int(s)): i for i, (p, s) in enumerate(zip(pids, S))}
    return X_lm, key_to_row


def raw_rows_for(df: pd.DataFrame, X_lm, key_to_row) -> np.ndarray:
    """(len(df), 327) float32 raw snap features; NaN where no cache match."""
    idx = np.array([key_to_row.get((p, int(s)), -1)
                    for p, s in zip(df["player_id"], df["snap_year"])],
                   dtype=np.int64)
    out = np.full((len(df), X_lm.shape[1]), np.nan, dtype=np.float32)
    ok = idx >= 0
    out[ok] = X_lm[idx[ok]]
    return out, float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--db", default=DB)
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--top-k", type=int, default=160)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--bag-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # exp2 winner config (fallback: f1_feats defaults)
    winner2, over2, screen2 = "f1_feats", {}, {}
    s2 = EXP2_DIR / "summary.json"
    if s2.exists():
        j = json.loads(s2.read_text())
        winner2 = j.get("winner", winner2)
        screen2 = j.get("screen", {})
        over2 = CANDIDATES[winner2][1] if winner2 in CANDIDATES else {}
    featsD = CANDIDATES[winner2][0] if winner2 in CANDIDATES else list(FEAT2)
    print(f"[exp3] inheriting exp2 winner: {winner2} overrides={over2}")

    print(f"[exp3] loading longs + panel cache")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, args.max_entry)
    val_base = prep_base(pd.read_csv(args.val), args.db,
                         max_entry=args.max_entry)
    X_lm, key_to_row = load_panel_map()

    folds, bag_seeds, topk = args.folds, args.bag_seeds, args.top_k
    sel_rounds, screen_cap, screen_estop = 150, 800, 30
    if args.quick:
        rng = np.random.default_rng(0)
        pids = fit_base["player_id"].unique()
        keep = set(rng.choice(pids, size=max(200, len(pids) // 7),
                              replace=False))
        fit_base = fit_base[fit_base["player_id"].isin(keep)]
        folds, bag_seeds, topk = 2, 2, 60
        sel_rounds, screen_cap, screen_estop = 40, 100, 20
        print(f"  [quick] {fit_base.player_id.nunique():,} players")

    fit_long, Y_fit = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    raw_fit, cov_fit = raw_rows_for(fit_long, X_lm, key_to_row)
    print(f"  fit long: {len(fit_long):,} rows, raw-feature coverage "
          f"{cov_fit:.1%} [{time.time()-t0:,.0f}s]")

    fit_pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    uniq = np.unique(fit_pids)
    rng = np.random.default_rng(7)          # SAME ES split as exp2
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_mask = np.isin(fit_pids, list(es_players))

    XD = fit_long[featsD].values.astype(np.float32)

    # ---- stage 0: gain-based raw-feature selection ----------------------
    print(f"\n[exp3] stage 0: selecting top {topk} raw features by gain")
    rng0 = np.random.default_rng(11)
    nsub = min(400_000, len(fit_long))
    sub = rng0.choice(len(fit_long), size=nsub, replace=False)
    raw_names = [f"rw_{n}" for n in FEATURE_NAMES]
    feats_all = featsD + raw_names
    Xsub = np.hstack([XD[sub], raw_fit[sub]])
    bst0 = train_one(Xsub, Y_fit[sub], feats_all, over2, sel_rounds, 0,
                     args.seed)
    gain = bst0.get_score(importance_type="total_gain")
    raw_rank = sorted(((gain.get(n, 0.0), n) for n in raw_names),
                      reverse=True)
    keep_raw = [n for g, n in raw_rank[:topk] if g > 0]
    print(f"  kept {len(keep_raw)} raw features; top 12: "
          f"{[n[3:] for _, n in raw_rank[:12]]}")
    del bst0, Xsub

    keep_idx = [raw_names.index(n) for n in keep_raw]
    X3 = np.hstack([XD, raw_fit[:, keep_idx]])
    feats3 = featsD + keep_raw
    del XD, raw_fit

    # ---- stage 1: screen -------------------------------------------------
    print(f"\n[exp3] screening (control = exp2 {winner2}: "
          f"ap_h3={screen2.get(winner2, {}).get('ap_h3', float('nan'))})")
    cands3 = {
        "e1_raw":    dict(over2),
        "e2_rawcol": {**over2, "colsample_bytree": 0.6},
    }
    print(f"{'cand':<12}{'rounds':>7}{'AP_h1':>8}{'AP_h2':>8}{'AP_h3':>8}"
          f"{'wAP_h6':>8}{'mins':>6}")
    screen = {}
    tr_m, es_m = ~es_mask, es_mask
    for name, over in cands3.items():
        tc = time.time()
        bst = train_one(X3[tr_m], Y_fit[tr_m], feats3, over, screen_cap,
                        screen_estop, args.seed, X3[es_m], Y_fit[es_m])
        bi = int(bst.best_iteration)
        P = predict_rows(bst, X3[es_m], feats3)
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
                        "ap_h3": ap3, "wap_h6": wap6, "overrides": over}
        print(f"{name:<12}{bi:>7}{ap1:>8.4f}{ap2:>8.4f}{ap3:>8.4f}"
              f"{wap6:>8.4f}{(time.time()-tc)/60:>6.1f}")
        del bst, P

    winner = max(screen, key=lambda k: screen[k]["ap_h3"])
    wcfg = screen[winner]
    ctrl_ap3 = screen2.get(winner2, {}).get("ap_h3")
    print(f"\n[exp3] winner: {winner} ap_h3={wcfg['ap_h3']:.4f} "
          f"(exp2 control {ctrl_ap3})")

    # ---- stage 2: bag + cross-fit + calibrators -------------------------
    over, nrounds = wcfg["overrides"], wcfg["best_iter"] + 1
    print(f"[exp3] bagging x{bag_seeds} on 100% fit ({nrounds} rounds)")
    bag = []
    for s in range(bag_seeds):
        bag.append(train_one(X3, Y_fit, feats3, over, nrounds, 0,
                             args.seed + 100 + s))
        print(f"  seed {args.seed + 100 + s} done [{time.time()-t0:,.0f}s]")
    with open(out_dir / "joint_xgb_exp3_bag.pkl", "wb") as fh:
        pickle.dump({"models": bag, "feature_names": list(feats3),
                     "keep_raw": keep_raw, "events": list(EVENTS),
                     "h_max": H_MAX, "winner": winner, "overrides": over,
                     "num_rounds": nrounds, "inherits": winner2,
                     "kind": "exp3_rawfeat_bag"}, fh)

    print(f"[exp3] {folds}-fold cross-fit for calibration")
    fold_of = {p: i % folds for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in fit_pids])
    oof = np.full((len(fit_long), len(EVENTS)), np.nan)
    for f in range(folds):
        tr, ho = fold_idx != f, fold_idx == f
        b = train_one(X3[tr], Y_fit[tr], feats3, over, nrounds, 0,
                      args.seed + 200 + f)
        oof[ho] = predict_rows(b, X3[ho], feats3)
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
    with open(out_dir / "calibrators_hyip.pkl", "wb") as fh:
        pickle.dump({"calibrators": cals, "events": list(EVENTS),
                     "h_max": H_MAX, "kind": "hyip_logistic_oof_exp3"}, fh)
    del X3, oof

    # ---- stage 3: val eval ----------------------------------------------
    print(f"\n[exp3] scoring val ({len(val_base):,} snaps)")
    raw_val, cov_val = raw_rows_for(val_base, X_lm, key_to_row)
    print(f"  val raw-feature coverage {cov_val:.1%}")
    for j, n in enumerate(keep_raw):
        val_base[n] = raw_val[:, keep_idx[j]]
    del raw_val, X_lm

    with open(_RUN.joint_xgb, "rb") as fh:
        bundle_a = pickle.load(fh)
    val_a = predict_trajectory(bundle_a, val_base)
    if _RUN.calibrators.exists():
        val_a = apply_baseline_calibrators(val_a, _RUN.calibrators)

    val_e_raw = sweep_val(bag, feats3, val_base)
    inv_e = inversion_stats(val_e_raw, "")
    val_e = val_e_raw.copy()
    for ev in EVENTS:
        cols = [f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]
        val_e[cols] = np.maximum.accumulate(
            val_e[cols].to_numpy(dtype=np.float64), axis=1)
    val_e = apply_cal_trajectory(val_e, cals)

    rows = []
    rows += per_h_metrics(val_a, "", "A_raw(v2.1c)")
    if _RUN.calibrators.exists():
        rows += per_h_metrics(val_a, "calA_", "A_cal(VAL-FIT,optim)")
    rows += per_h_metrics(val_e, "", "E_raw(rawfeat bag)")
    rows += per_h_metrics(val_e, "cal_", "E_cal(OOF,honest)")
    met = pd.DataFrame(rows)
    d_csv = EXP2_DIR / "per_event_h_metrics.csv"
    if d_csv.exists():
        d_met = pd.read_csv(d_csv)
        d_met = d_met[d_met.scorer.isin(["D_raw(bag)", "D_cal(OOF,honest)"])]
        met = pd.concat([met, d_met], ignore_index=True)
    met.to_csv(out_dir / "per_event_h_metrics.csv", index=False)

    print(f"\n===== MLB_DEBUT (PRIMARY: h=3) =====")
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

    print(f"\n===== per-yip debut calib @ h=3 =====")
    tabs = {"A_raw": per_yip_debut_calib(val_a, "xp_MLB_DEBUT_h3", 3),
            "E_cal": per_yip_debut_calib(val_e, "cal_xp_MLB_DEBUT_h3", 3)}
    yips = sorted(set().union(*[set(t["yip"]) for t in tabs.values()]))
    print(f"{'yip':>4}" + "".join(f"{k:>10}" for k in tabs))
    for yip in yips:
        cells = []
        for k, t in tabs.items():
            r = t[t.yip == yip]
            cells.append(f"{r.iloc[0]['calib']:>10.2f}" if len(r)
                         else f"{'—':>10}")
        print(f"{yip:>4}" + "".join(cells))

    print(f"\n===== inversions pre-cummax (E bag) =====")
    for ev, s in inv_e.items():
        print(f"  {ev:<22} frac={s['frac_snaps_with_inversion']:.3f} "
              f"mean_drop={s['mean_total_drop']:.4f}")

    print(f"\n[exp3] timing (MLB_DEBUT)")
    tim_e = cdf_timing(val_e, "MLB_DEBUT", "cal_")
    lasso_pred = score_lasso_timing(val_base, _RUN.timing, args.db)
    est = {
        "E_cdf_mean(cal,OOF)": tim_e["t_mean"].to_numpy(),
        "E_cdf_median(cal,OOF)": tim_e["t_med"].to_numpy(),
        "lasso_timing.pkl": lasso_pred,
    }
    windows = {"E_cdf_median(cal,OOF)": (tim_e["t_q25"].to_numpy(),
                                         tim_e["t_q75"].to_numpy())}
    trep = timing_report(val_base, est, windows)
    trep.to_csv(out_dir / "timing_metrics.csv", index=False)
    print(trep.to_string(index=False))

    summary = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": bool(args.quick),
        "inherits_exp2_winner": winner2,
        "winner": winner,
        "n_raw_features": len(keep_raw),
        "raw_coverage_fit": cov_fit, "raw_coverage_val": cov_val,
        "top_raw": [n[3:] for n in keep_raw[:25]],
        "screen": {k: {kk: vv for kk, vv in v.items()}
                   for k, v in screen.items()},
        "exp2_control_ap_h3": ctrl_ap3,
        "timing": trep.to_dict(orient="records"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[exp3] wrote {out_dir}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
