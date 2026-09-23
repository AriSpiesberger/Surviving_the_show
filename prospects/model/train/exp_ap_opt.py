"""Optimise the thing the sheet is judged on: AP of P(MLB debut <= 3y). (2026-09-20)

Production's joint model is one XGBoost fit to 4 events x 10 horizons, log-loss, a fixed
number of rounds tuned for that blend. This campaign starts from the best feature set found
so far (single stage: all raw panel features + the sequence embedding, recency-weighted) and
pulls the levers that target debut@3 AP directly:

  debut-only target | early stopping on aucpr over h=3 rows | horizon focus (drop or
  down-weight far horizons) | HP search | seed bag of the winner

Honest selection. With ~14 configs and 3 origins, picking the best OUT-OF-ERA score would be
selection on the test set. Every config is scored on an INNER holdout (10% of the training
players, h=3 rows, never the eval cohort); the winner is chosen by mean inner AP, and only
then is its out-of-era AP read, with paired bootstrap against production (A_oof_capY) and
against the 4-target B1+emb arm.

Fair walk-forward throughout (runs/exp_walkforward_h/Y*/capY; embeddings stacked
out-of-fold exactly as in exp_seq_d, and cached to runs/exp_seq_d/emb_Y<origin>.npz).

    python -m prospects.model.train.exp_ap_opt
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score

from prospects.config import REPO_ROOT
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base
from prospects.model.joint2 import attach_raw_features
from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, _mono_string, stamp_extra_cols
from prospects.model.train.exp_macro_bc import OUT_DIR as BC_DIR, feature_sets
from prospects.model.train.exp_seq_d import (
    EMB, OUT_DIR as D_DIR, SEQ_OUT, WFH, Tokens, embed, sample_frame, train_encoder,
)
from prospects.model.train.exp_walkforward2 import DB, EVAL_H, GAP
from prospects.model.train.joint_xgb import _assemble

OUT_DIR = REPO_ROOT / "runs" / "exp_ap_opt"
K = EVENTS.index("MLB_DEBUT")
EMB_COLS = [f"seq_emb_{i}" for i in range(EMB)]

BASE = {"max_depth": 8, "min_child_weight": 100, "colsample_bytree": 0.6, "learning_rate": 0.03}
# name -> (param overrides, horizon scheme, positive weight)
#   horizon scheme: "all" | "le4" (train only h<=4) | "w35" (h>=4 rows weighted 0.35)
CONFIGS = {
    "c00 base (fixed 340, logloss)": (dict(BASE), "all", 1.0),
    "c01 es-aucpr":                  (dict(BASE), "all", 1.0),
    "c02 es + h<=4":                 (dict(BASE), "le4", 1.0),
    "c03 es + far-h x0.35":          (dict(BASE), "w35", 1.0),
    "c04 es + depth6 mcw30":         ({**BASE, "max_depth": 6, "min_child_weight": 30}, "w35", 1.0),
    "c05 es + depth10 mcw200":       ({**BASE, "max_depth": 10, "min_child_weight": 200}, "w35", 1.0),
    "c06 es + depth5 mcw50 col.4":   ({**BASE, "max_depth": 5, "min_child_weight": 50, "colsample_bytree": 0.4}, "w35", 1.0),
    "c07 es + col.4":                ({**BASE, "colsample_bytree": 0.4}, "w35", 1.0),
    "c08 es + col.8 sub.8":          ({**BASE, "colsample_bytree": 0.8, "subsample": 0.8}, "w35", 1.0),
    "c09 es + lr.015":               ({**BASE, "learning_rate": 0.015}, "w35", 1.0),
    "c10 es + mds1 (imbalance)":     ({**BASE, "max_delta_step": 1}, "w35", 1.0),
    "c11 es + pos x3":               (dict(BASE), "w35", 3.0),
    "c12 es + lambda5 gamma1":       ({**BASE, "reg_lambda": 5.0, "gamma": 1.0}, "w35", 1.0),
    "c13 es + depth6 mcw30 col.4 lr.02": ({**BASE, "max_depth": 6, "min_child_weight": 30,
                                           "colsample_bytree": 0.4, "learning_rate": 0.02}, "w35", 1.0),
}


def embeddings(Y, tok, folds, rest, evl, epochs, log):
    """Out-of-fold sequence embeddings for one origin (cached)."""
    snap = Y + GAP
    f = D_DIR / f"emb_Y{Y}_{SEQ_OUT}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return pd.concat([pd.DataFrame({"player_id": z["pid"], "snap_year": z["snap"]}),
                          pd.DataFrame(z["emb"], columns=EMB_COLS)], axis=1)
    train_players = set(pd.concat(folds).player_id.unique())
    tok.fit_scaler(train_players, snap)
    samples = [sample_frame(x, snap - 1) for x in folds]
    parts = []

    def pack(frame, e):
        return pd.concat([frame[["player_id", "snap_year"]].reset_index(drop=True),
                          pd.DataFrame(e, columns=EMB_COLS)], axis=1)

    for k in range(3):
        d_tr = pd.concat([samples[g] for g in range(3) if g != k], ignore_index=True)
        net = train_encoder(tok, d_tr, epochs, 100 + k, lambda m: None)
        b = folds[k].drop_duplicates(["player_id", "snap_year"])
        parts.append(pack(b, embed(net, tok, b)))
        log(f"    encoder fold {k} done")
    net = train_encoder(tok, pd.concat(samples, ignore_index=True), epochs, 200, lambda m: None)
    b = rest.drop_duplicates(["player_id", "snap_year"])
    b = b[~b.player_id.isin(train_players)]
    if len(b):
        parts.append(pack(b, embed(net, tok, b)))
    emb_df = pd.concat(parts, ignore_index=True)
    ev = evl[evl.snap_year == snap].drop_duplicates(["player_id", "snap_year"])
    emb_df = pd.concat([emb_df[emb_df.snap_year != snap], pack(ev, embed(net, tok, ev))], ignore_index=True)
    emb_df = emb_df.drop_duplicates(["player_id", "snap_year"], keep="last")
    D_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, pid=emb_df.player_id.to_numpy(), snap=emb_df.snap_year.to_numpy(),
                        emb=emb_df[EMB_COLS].to_numpy(np.float32))
    log("    encoder full done; embeddings cached")
    return emb_df


def fit(X, y, w, feats, params, rounds, seed, Xes=None, yes=None):
    p = dict(BASE_PARAMS)
    p.update(params)
    p["seed"] = seed
    p["monotone_constraints"] = _mono_string(feats)
    dtr = xgb.QuantileDMatrix(X, label=y, weight=w, feature_names=feats)
    if Xes is None:
        return xgb.train(p, dtr, num_boost_round=rounds, verbose_eval=False)
    p["eval_metric"] = "aucpr"
    des = xgb.QuantileDMatrix(Xes, label=yes, feature_names=feats, ref=dtr)
    return xgb.train(p, dtr, num_boost_round=rounds, evals=[(des, "es")],
                     early_stopping_rounds=60, maximize=True, verbose_eval=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origins", nargs="*", type=int, default=[2016, 2014, 2012])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bag", type=int, default=5)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--bag-only", nargs="*", default=None,
                    help="skip the grid; bag these config prefixes using rounds from grid.csv")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"{m}  [{(time.time() - t0) / 60:.0f}m]", flush=True)
    configs = {k: v for k, v in CONFIGS.items() if not args.configs or any(k.startswith(c) for c in args.configs)}

    with open(REPO_ROOT / "runs" / "current" / "models" / "joint_xgb_v2.3.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in keep_raw if c in live]
    featsB = feature_sets(keep_raw)["B"]
    feats = featsB + EMB_COLS
    raw_all = sorted(set(featsB) & live)
    tok = Tokens()

    data, rows = {}, []
    for Y in args.origins:
        snap = Y + GAP
        odir = WFH / f"Y{Y}" / "capY"
        log(f"\n===== origin Y={Y} (score snap {snap}) =====")
        folds = [prep_base(pd.read_csv(odir / f"oof_fold{k}.csv", low_memory=False), DB) for k in range(3)]
        rest = prep_base(pd.read_csv(odir / "rest_long.csv", low_memory=False), DB)
        evl = prep_base(pd.read_csv(odir / "eval_long.csv", low_memory=False), DB)
        emb_df = embeddings(Y, tok, folds, rest, evl, args.epochs, log)

        both = pd.concat(folds + [rest[rest.snap_year < snap]], ignore_index=True)
        both = both[(both.snap_year < snap) & (both.get("eligible_MLB_DEBUT", 1) == 1)]
        both = attach_raw_features(both, DB, raw_all, verbose=False)
        both = both.merge(emb_df, on=["player_id", "snap_year"], how="left")
        fit_long, Yf = _assemble(both, H_MAX)
        fit_long = stamp_extra_cols(fit_long)
        del both
        X = fit_long[feats].values.astype(np.float32)
        y = Yf[:, K].astype(np.float32)
        h = fit_long["h"].astype(int).to_numpy()
        rec = (0.5 ** ((Y - fit_long["snap_year"].to_numpy()) / 4.0)).astype(np.float32)
        pl = fit_long["player_id"].to_numpy()
        uniq = np.unique(pl)
        hold = set(np.random.default_rng(5).choice(uniq, size=len(uniq) // 10, replace=False))
        es = np.isin(pl, list(hold))
        es3 = es & (h == 3)

        ev = evl[(evl.snap_year == snap) & (evl.get("eligible_MLB_DEBUT", 1) == 1)].copy()
        ev = attach_raw_features(ev, DB, raw_all, verbose=False).merge(emb_df, on=["player_id", "snap_year"], how="left")
        trig = pd.to_numeric(ev["trigger_MLB_DEBUT"], errors="coerce")
        yev = ((trig > snap) & (trig <= snap + EVAL_H)).fillna(False).to_numpy().astype(int)
        Xev = [stamp_extra_cols(add_cond_cols(ev, hh))[feats].values.astype(np.float32) for hh in (1, 2, 3)]
        log(f"    rows {len(X):,} | inner holdout h=3 rows {int(es3.sum()):,} ({int(y[es3].sum())} debuts) | "
            f"eval n={len(yev):,} pos={int(yev.sum())}")

        def outer(models):
            P = [np.mean([m.predict(xgb.DMatrix(x, feature_names=feats)) for m in models], axis=0) for x in Xev]
            return np.maximum.accumulate(np.column_stack(P), axis=1)[:, 2]

        def weights(scheme, posw):
            w = rec.copy()
            m = ~es
            if scheme == "le4":
                m = m & (h <= 4)
            elif scheme == "w35":
                w = w * np.where(h >= 4, 0.35, 1.0).astype(np.float32)
            w = w * np.where(y > 0, posw, 1.0).astype(np.float32)
            return m, w

        if args.bag_only:
            g = pd.read_csv(OUT_DIR / "grid.csv")
            for pref in args.bag_only:
                name = next(k for k in CONFIGS if k.startswith(pref))
                params, scheme, posw = CONFIGS[name]
                nround = int(g[(g.Y == Y) & (g.config == name)].rounds.iloc[0])
                m, w = weights(scheme, posw)
                m = (m | es) if scheme != "le4" else (m | (es & (h <= 4)))      # refit on 100%
                bag = [fit(X[m], y[m], w[m], feats, params, nround, 1000 + s_) for s_ in range(args.bag)]
                pb = outer(bag)
                np.savez_compressed(OUT_DIR / f"preds_Y{Y}_{pref}_bag.npz", p=pb, y=yev,
                                    pid=ev["player_id"].to_numpy())
                log(f"    {name:36s} {args.bag}-seed bag, {nround} rounds: out-of-era AP@3 "
                    f"{average_precision_score(yev, pb):.4f}")
                del bag
            del X
            continue
        for name, (params, scheme, posw) in configs.items():
            m, w = weights(scheme, posw)
            if name.startswith("c00"):
                bst = fit(X[m], y[m], w[m], feats, params, 340, 42)
                nround = 340
            else:
                bst = fit(X[m], y[m], w[m], feats, params, 1500, 42, X[es3], y[es3])
                nround = int(bst.best_iteration) + 1
            inner = average_precision_score(y[es3], bst.predict(xgb.DMatrix(X[es3], feature_names=feats),
                                                                iteration_range=(0, nround)))
            p_out = outer([bst])
            rows.append({"Y": Y, "config": name, "rounds": nround, "inner_ap3": float(inner),
                         "outer_ap3": float(average_precision_score(yev, p_out))})
            np.savez_compressed(OUT_DIR / f"preds_Y{Y}_{name.split()[0]}.npz", p=p_out, y=yev,
                                pid=ev["player_id"].to_numpy())
            log(f"    {name:36s} rounds {nround:4d}  inner AP@3 {inner:.4f}  out-of-era AP@3 {rows[-1]['outer_ap3']:.4f}")
            del bst
            pd.DataFrame(rows).to_csv(OUT_DIR / "grid.csv", index=False)

    if not args.bag_only:
        grid = pd.DataFrame(rows)
        summ = grid.groupby("config").agg(inner=("inner_ap3", "mean"), outer=("outer_ap3", "mean"),
                                          rounds=("rounds", "mean")).sort_values("inner", ascending=False)
        print("\n===== grid: mean over origins (chosen by INNER, outer is read-only) =====")
        print(summ.round(4).to_string())
        print("\nre-run with --bag-only <prefix> ... to bag configs")
        return

    rng = np.random.default_rng(0)
    for pref in args.bag_only:
        print(f"\n===== {pref} (bagged) vs controls: out-of-era AP, debut <= 3y, paired bootstrap =====")
        for ctrl in ("A_oof_capY", "D_B1+emb"):
            ds = []
            for Y in args.origins:
                zn = np.load(OUT_DIR / f"preds_Y{Y}_{pref}_bag.npz", allow_pickle=True)
                new = pd.Series(zn["p"], index=zn["pid"])
                yy = pd.Series(zn["y"], index=zn["pid"])
                z = np.load(BC_DIR / f"preds_Y{Y}_{ctrl}.npz", allow_pickle=True)
                c = pd.Series(z["cal3"], index=z["pid"])
                idx = new.index.intersection(c.index)
                a_, b_, t = new.loc[idx].to_numpy(), c.loc[idx].to_numpy(), yy.loc[idx].to_numpy()
                dd = []
                for _ in range(400):
                    k = rng.integers(0, len(t), len(t))
                    if t[k].sum():
                        dd.append(average_precision_score(t[k], a_[k]) - average_precision_score(t[k], b_[k]))
                dd = np.array(dd)
                ds.append(dd)
                print(f"  Y{Y}  {pref} {average_precision_score(t, a_):.4f}  vs {ctrl:11s} "
                      f"{average_precision_score(t, b_):.4f}   d={dd.mean():+.4f} "
                      f"[{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]")
            mm = np.mean([x[:min(map(len, ds))] for x in ds], axis=0)
            print(f"  MEAN vs {ctrl}: {mm.mean():+.4f} [{np.percentile(mm, 2.5):+.4f}, "
                  f"{np.percentile(mm, 97.5):+.4f}]  P(>0)={np.mean(mm > 0):.2f}")
    log("done")


if __name__ == "__main__":
    main()
