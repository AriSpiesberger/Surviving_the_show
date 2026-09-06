"""EXPERIMENT 4: refine around exp3's winner (raw-feature pass-through).

exp3 (E) was a decisive val win over v2.1c: debut AP@h3 0.682 vs 0.647,
weighted AP@h6 0.549 vs 0.514, per-yip debut calib flat (0.93-1.07), CDF
timing MAE 0.975 vs 1.163. This round screens the neighborhood:

  g0_ctrl    exp3 e2_rawcol exact (top-160 raw, depth 8, mcw 100,
             colsample 0.6) — control, same ES split as exp2/exp3
  g1_col04   colsample 0.4 (more feature bagging over the wide matrix)
  g2_raw240  top-240 raw features
  g3_slow    lr 0.03, cap 2000 rounds
  g4_sub     subsample 0.8

Winner by ES debut AP@h3 -> 5-seed bag -> 3-fold cross-fit -> calibrator
with quadratic flex (HYip2: adds h_c^2, yip_c^2 — exp3's linear-in-h map
under-flexed at the short-h STAR extreme). Full val tables incl. per-event
calib at h3/h6 and STAR per-h.

Also fixes exp3's fragmented per-column inserts (pd.concat once).

Writes runs/exp_cdf_timing4/. Touches nothing under runs/current/.
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
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import (
    EVENTS, H_MAX, PUBLISH_H, add_cond_cols, predict_trajectory, prep_base,
)
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import (
    _logit, apply_baseline_calibrators, cdf_timing, inversion_stats,
    per_h_metrics, score_lasso_timing, timing_report, weighted_ap_at,
)
from prospects.model.train.exp_cdf_timing2 import (
    BASE_PARAMS, FEAT2, MONO_UP, debut_ap_at, per_yip_debut_calib,
    stamp_extra_cols,
)
from prospects.model.train.exp_cdf_timing3 import (
    load_panel_map, raw_rows_for,
)

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_cdf_timing4"
EXP3_DIR = REPO_ROOT / "runs" / "exp_cdf_timing3"

DEEP = {"max_depth": 8, "min_child_weight": 100}


def _mono_string(feats):
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


class HYip2Calibrator:
    """logit(p), h_c, yip_c, interactions + quadratic h/yip flex."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=4000)

    @staticmethod
    def _feats(p, h, yip):
        lp = _logit(p)
        hc = np.asarray(h, dtype=np.float64) - 5.0
        yc = np.clip(np.asarray(yip, dtype=np.float64), 0, 10) - 3.0
        return np.column_stack([lp, hc, yc, lp * hc, lp * yc,
                                hc * hc, yc * yc])

    def fit(self, p, h, yip, y):
        self.lr.fit(self._feats(p, h, yip), np.asarray(y, dtype=int))
        return self

    def predict(self, p, h, yip):
        return self.lr.predict_proba(self._feats(p, h, yip))[:, 1]


def apply_cal_trajectory(df, cals, h_max=H_MAX):
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
        new = {f"cal_xp_{ev}_h{h}": M[:, hi]
               for hi, h in enumerate(range(1, h_max + 1))}
        out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    return out


def sweep_val(bsts, feats, df, h_max=H_MAX):
    out = df.copy()
    preds_by_h = {}
    for h in range(1, h_max + 1):
        sub = stamp_extra_cols(add_cond_cols(df, h))
        X = sub[feats].values.astype(np.float32)
        preds_by_h[h] = np.mean(
            [predict_rows(b, X, feats) for b in bsts], axis=0)
    new = {}
    for k, ev in enumerate(EVENTS):
        M = np.column_stack([preds_by_h[h][:, k]
                             for h in range(1, h_max + 1)])
        for hi, h in enumerate(range(1, h_max + 1)):
            new[f"xp_{ev}_h{h}"] = M[:, hi]
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


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
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[exp4] loading longs + panel cache")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, args.max_entry)
    val_base = prep_base(pd.read_csv(args.val), args.db,
                         max_entry=args.max_entry)
    X_lm, key_to_row = load_panel_map()

    folds, bag_seeds = args.folds, args.bag_seeds
    sel_rounds, cap, estop = 150, 800, 30
    topk_max = 240
    if args.quick:
        rng = np.random.default_rng(0)
        pids = fit_base["player_id"].unique()
        keep = set(rng.choice(pids, size=max(200, len(pids) // 7),
                              replace=False))
        fit_base = fit_base[fit_base["player_id"].isin(keep)]
        folds, bag_seeds, topk_max = 2, 2, 80
        sel_rounds, cap, estop = 40, 100, 20
        print(f"  [quick] {fit_base.player_id.nunique():,} players")

    fit_long, Y_fit = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    raw_fit, cov_fit = raw_rows_for(fit_long, X_lm, key_to_row)
    print(f"  fit long: {len(fit_long):,} rows, raw coverage {cov_fit:.1%} "
          f"[{time.time()-t0:,.0f}s]")

    fit_pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    uniq = np.unique(fit_pids)
    rng = np.random.default_rng(7)          # SAME ES split as exp2/exp3
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_mask = np.isin(fit_pids, list(es_players))

    featsD = list(FEAT2)
    XD = fit_long[featsD].values.astype(np.float32)

    # ---- stage 0: rank raw features by gain (once, at topk_max) --------
    print(f"\n[exp4] stage 0: ranking raw features by gain")
    rng0 = np.random.default_rng(11)
    sub = rng0.choice(len(fit_long),
                      size=min(400_000, len(fit_long)), replace=False)
    raw_names = [f"rw_{n}" for n in FEATURE_NAMES]
    bst0 = train_one(np.hstack([XD[sub], raw_fit[sub]]), Y_fit[sub],
                     featsD + raw_names, DEEP, sel_rounds, 0, args.seed)
    gain = bst0.get_score(importance_type="total_gain")
    ranked = [n for g, n in sorted(((gain.get(n, 0.0), n)
                                    for n in raw_names), reverse=True)
              if g > 0][:topk_max]
    print(f"  ranked {len(ranked)} raw features with gain>0")
    del bst0

    keep_idx = [raw_names.index(n) for n in ranked]
    X4 = np.hstack([XD, raw_fit[:, keep_idx]])   # width = base + topk_max
    del XD, raw_fit

    def _cols(n_raw):
        feats = featsD + ranked[:n_raw]
        idxs = list(range(len(featsD))) + [len(featsD) + i
                                           for i in range(n_raw)]
        return feats, np.asarray(idxs)

    n160 = min(160, len(ranked))
    n240 = len(ranked)
    cands = {
        "g0_ctrl":   (n160, {**DEEP, "colsample_bytree": 0.6}, cap, estop),
        "g1_col04":  (n160, {**DEEP, "colsample_bytree": 0.4}, cap, estop),
        "g2_raw240": (n240, {**DEEP, "colsample_bytree": 0.6}, cap, estop),
        "g3_slow":   (n160, {**DEEP, "colsample_bytree": 0.6,
                             "learning_rate": 0.03}, 2000, 50),
        "g4_sub":    (n160, {**DEEP, "colsample_bytree": 0.6,
                             "subsample": 0.8}, cap, estop),
    }
    if args.quick:
        cands = {k: v for k, v in list(cands.items())[:2]}

    print(f"\n[exp4] screening {len(cands)} candidates")
    print(f"{'cand':<12}{'nraw':>5}{'rounds':>7}{'AP_h1':>8}{'AP_h2':>8}"
          f"{'AP_h3':>8}{'wAP_h6':>8}{'mins':>6}")
    screen = {}
    tr_m, es_m = ~es_mask, es_mask
    for name, (n_raw, over, rounds, es_stop) in cands.items():
        tc = time.time()
        feats, ci = _cols(n_raw)
        bst = train_one(X4[np.ix_(tr_m, ci)], Y_fit[tr_m], feats, over,
                        rounds, es_stop, args.seed,
                        X4[np.ix_(es_m, ci)], Y_fit[es_m])
        bi = int(bst.best_iteration)
        P = predict_rows(bst, X4[np.ix_(es_m, ci)], feats)
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
        screen[name] = {"best_iter": bi, "n_raw": n_raw, "ap_h1": ap1,
                        "ap_h2": ap2, "ap_h3": ap3, "wap_h6": wap6,
                        "overrides": over}
        print(f"{name:<12}{n_raw:>5}{bi:>7}{ap1:>8.4f}{ap2:>8.4f}"
              f"{ap3:>8.4f}{wap6:>8.4f}{(time.time()-tc)/60:>6.1f}")
        del bst, P

    winner = max(screen, key=lambda k: screen[k]["ap_h3"])
    wcfg = screen[winner]
    print(f"\n[exp4] winner: {winner} ap_h3={wcfg['ap_h3']:.4f}")

    feats, ci = _cols(wcfg["n_raw"])
    over, nrounds = wcfg["overrides"], wcfg["best_iter"] + 1
    Xw = X4[:, ci]
    del X4

    print(f"[exp4] bagging x{bag_seeds} on 100% fit ({nrounds} rounds)")
    bag = []
    for s in range(bag_seeds):
        bag.append(train_one(Xw, Y_fit, feats, over, nrounds, 0,
                             args.seed + 100 + s))
        print(f"  seed {args.seed + 100 + s} done [{time.time()-t0:,.0f}s]")
    with open(out_dir / "joint_xgb_exp4_bag.pkl", "wb") as fh:
        pickle.dump({"models": bag, "feature_names": list(feats),
                     "keep_raw": ranked[:wcfg["n_raw"]],
                     "events": list(EVENTS), "h_max": H_MAX,
                     "winner": winner, "overrides": over,
                     "num_rounds": nrounds,
                     "kind": "exp4_rawfeat_bag"}, fh)

    print(f"[exp4] {folds}-fold cross-fit for calibration")
    fold_of = {p: i % folds for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in fit_pids])
    oof = np.full((len(fit_long), len(EVENTS)), np.nan)
    for f in range(folds):
        tr, ho = fold_idx != f, fold_idx == f
        b = train_one(Xw[tr], Y_fit[tr], feats, over, nrounds, 0,
                      args.seed + 200 + f)
        oof[ho] = predict_rows(b, Xw[ho], feats)
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
        cals[ev] = HYip2Calibrator().fit(oof[ok, k], h_arr[ok],
                                         yip_arr[ok], y)
    with open(out_dir / "calibrators_hyip2.pkl", "wb") as fh:
        pickle.dump({"calibrators": cals, "events": list(EVENTS),
                     "h_max": H_MAX, "kind": "hyip2_logistic_oof"}, fh)
    del Xw, oof

    # ---- val eval -------------------------------------------------------
    print(f"\n[exp4] scoring val ({len(val_base):,} snaps)")
    raw_val, cov_val = raw_rows_for(val_base, X_lm, key_to_row)
    print(f"  val raw coverage {cov_val:.1%}")
    keep_raw = ranked[:wcfg["n_raw"]]
    val_base = pd.concat([val_base, pd.DataFrame(
        {n: raw_val[:, keep_idx[j]] for j, n in enumerate(ranked)
         if n in set(keep_raw)},
        index=val_base.index)], axis=1)
    del raw_val, X_lm

    with open(_RUN.joint_xgb, "rb") as fh:
        bundle_a = pickle.load(fh)
    val_a = predict_trajectory(bundle_a, val_base)
    if _RUN.calibrators.exists():
        val_a = apply_baseline_calibrators(val_a, _RUN.calibrators)

    val_f_raw = sweep_val(bag, feats, val_base)
    inv_f = inversion_stats(val_f_raw, "")
    val_f = val_f_raw.copy()
    for ev in EVENTS:
        cols = [f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]
        val_f[cols] = np.maximum.accumulate(
            val_f[cols].to_numpy(dtype=np.float64), axis=1)
    val_f = apply_cal_trajectory(val_f, cals)

    rows = []
    rows += per_h_metrics(val_a, "", "A_raw(v2.1c)")
    rows += per_h_metrics(val_f, "", "F_raw(exp4 bag)")
    rows += per_h_metrics(val_f, "cal_", "F_cal(OOF,honest)")
    met = pd.DataFrame(rows)
    e_csv = EXP3_DIR / "per_event_h_metrics.csv"
    if e_csv.exists():
        e_met = pd.read_csv(e_csv)
        e_met = e_met[e_met.scorer.isin(["E_raw(rawfeat bag)",
                                         "E_cal(OOF,honest)"])]
        met = pd.concat([met, e_met], ignore_index=True)
    met.to_csv(out_dir / "per_event_h_metrics.csv", index=False)

    print(f"\n===== MLB_DEBUT (PRIMARY: h=3) =====")
    print(f"{'scorer':<24}{'h':>3}{'AP':>8}{'AUC':>8}{'Brier':>9}"
          f"{'calib':>7}")
    for scorer in met["scorer"].unique():
        for h in (1, 2, 3, 6):
            r = met[(met.scorer == scorer) & (met.event == "MLB_DEBUT")
                    & (met.h == h)]
            if len(r):
                r = r.iloc[0]
                print(f"{scorer:<24}{h:>3}{r['ap']:>8.4f}{r['auc']:>8.4f}"
                      f"{r['brier']:>9.4f}{r['calib']:>7.2f}")

    print(f"\n===== weighted AP @ h=6 =====")
    for scorer in met["scorer"].unique():
        sub_ = met[(met.scorer == scorer) & (met.h == PUBLISH_H)]
        print(f"  {scorer:<24} "
              f"{weighted_ap_at(sub_.to_dict(orient='records'), PUBLISH_H):.4f}")

    print(f"\n===== per-event @ h=3 and h=6 (F_cal) =====")
    for h in (3, PUBLISH_H):
        print(f"  h={h}:")
        for ev in EVENTS:
            r = met[(met.scorer == "F_cal(OOF,honest)")
                    & (met.event == ev) & (met.h == h)]
            if len(r):
                r = r.iloc[0]
                print(f"    {ev:<20} AP={r['ap']:.4f} AUC={r['auc']:.4f} "
                      f"calib={r['calib']:.2f}")

    print(f"\n===== STAR_PLUS_ELITE per-h calib (F_cal vs exp3 E_cal) =====")
    for h in range(1, H_MAX + 1):
        rf = met[(met.scorer == "F_cal(OOF,honest)")
                 & (met.event == "STAR_PLUS_ELITE") & (met.h == h)]
        re_ = met[(met.scorer == "E_cal(OOF,honest)")
                  & (met.event == "STAR_PLUS_ELITE") & (met.h == h)]
        f_c = rf.iloc[0]["calib"] if len(rf) else float("nan")
        e_c = re_.iloc[0]["calib"] if len(re_) else float("nan")
        print(f"  h={h:>2}  F={f_c:>5.2f}  E={e_c:>5.2f}")

    print(f"\n===== per-yip debut calib @ h=3 =====")
    tabs = {"A_raw": per_yip_debut_calib(val_a, "xp_MLB_DEBUT_h3", 3),
            "F_cal": per_yip_debut_calib(val_f, "cal_xp_MLB_DEBUT_h3", 3)}
    yips = sorted(set().union(*[set(t["yip"]) for t in tabs.values()]))
    print(f"{'yip':>4}" + "".join(f"{k:>10}" for k in tabs))
    for yip in yips:
        cells = []
        for k, t in tabs.items():
            r = t[t.yip == yip]
            cells.append(f"{r.iloc[0]['calib']:>10.2f}" if len(r)
                         else f"{'—':>10}")
        print(f"{yip:>4}" + "".join(cells))

    print(f"\n===== inversions pre-cummax (F bag) =====")
    for ev, s in inv_f.items():
        print(f"  {ev:<22} frac={s['frac_snaps_with_inversion']:.3f} "
              f"mean_drop={s['mean_total_drop']:.4f}")

    print(f"\n[exp4] timing (MLB_DEBUT)")
    tim_f = cdf_timing(val_f, "MLB_DEBUT", "cal_")
    lasso_pred = score_lasso_timing(val_base, _RUN.timing, args.db)
    est = {
        "F_cdf_mean(cal,OOF)": tim_f["t_mean"].to_numpy(),
        "F_cdf_median(cal,OOF)": tim_f["t_med"].to_numpy(),
        "lasso_timing.pkl": lasso_pred,
    }
    windows = {"F_cdf_median(cal,OOF)": (tim_f["t_q25"].to_numpy(),
                                         tim_f["t_q75"].to_numpy())}
    trep = timing_report(val_base, est, windows)
    trep.to_csv(out_dir / "timing_metrics.csv", index=False)
    print(trep.to_string(index=False))

    summary = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": bool(args.quick),
        "winner": winner,
        "screen": screen,
        "raw_coverage_fit": cov_fit, "raw_coverage_val": cov_val,
        "timing": trep.to_dict(orient="records"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[exp4] wrote {out_dir}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
