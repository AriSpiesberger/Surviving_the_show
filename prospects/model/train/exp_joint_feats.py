"""Feature-set screen for the production joint recipe (2026-09-23).

The deployed joint model (exp5 recipe) carries only 8 of the panel's 19 ranking features: none
of the top-100 list history (ever / best / recent / times / years-since) and not
ever_org_ranked / times_org_ranked / log_best_org_rank. The user: "ranking stats should be used
in general". This trains the joint recipe twice on the SAME sampled players — as deployed, and
with extra raw panel features — and compares per-event AP on the full held-out val.

Both arms: multi-output XGB, exp5 params and round count, min_child_weight scaled by the sample
fraction, HYip2 calibrators fit on 3-fold cross-fit predictions (snaps >= 2008).

    python -m prospects.model.train.exp_joint_feats --sample-frac 0.25          # rank features
    python -m prospects.model.train.exp_joint_feats --add rw_foo rw_bar
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
from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, _mono_string, stamp_extra_cols
from prospects.model.train.joint_xgb import _assemble, _prep_train

_RUN = config.run()
DB = str(config.model_db())
RANK_EXTRA = ["rw_ever_top100", "rw_best_top100_rank", "rw_recent_top100_rank", "rw_times_top100",
              "rw_years_since_first_top100", "rw_log_best_top100_rank", "rw_ever_org_ranked",
              "rw_times_org_ranked", "rw_log_best_org_rank", "rw_scout_org_rank", "rw_scout_ovr_rank_c",
              "rw_scout_org_rank_c"]
EVAL_H = (3, 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--add", nargs="*", default=None, help="raw panel features to add (default: ranking)")
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--sample-frac", type=float, default=0.25)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--tag", default="rank")
    args = ap.parse_args()
    frac = args.sample_frac if 0 < args.sample_frac < 1 else 1.0
    out = REPO_ROOT / "runs" / f"exp_joint_feats_{args.tag}_q{int(frac * 100)}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[jf] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bag = pickle.load(fh)
    featsA = list(bag["feature_names"])
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    add = [f for f in (args.add if args.add is not None else RANK_EXTRA) if f in live and f not in featsA]
    featsB = featsA + add
    keep = sorted(set(bag["keep_raw"]) | set(add))
    tick(f"adding {len(add)} features: {add}")

    def sample(df):
        if frac == 1.0:
            return df
        u = np.sort(df["player_id"].unique())
        k = np.random.default_rng(args.sample_seed).choice(u, int(len(u) * frac), replace=False)
        return df[df["player_id"].isin(k)]

    fit_base = _prep_train(sample(pd.read_csv(args.fit, low_memory=False)), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, sample(aug)], ignore_index=True)
    val = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    fit_base = attach_raw_features(fit_base, DB, keep, verbose=False)
    val = attach_raw_features(val, DB, keep, verbose=False)
    fl, Y = _assemble(fit_base, H_MAX)
    fl = stamp_extra_cols(fl)
    del fit_base
    h_arr = fl["h"].astype(int).to_numpy()
    yip_arr = fl["snap_offset"].to_numpy()
    era_ok = fl["snap_year"].to_numpy() >= 2008
    pids = fl["player_id"].to_numpy()
    elig = {ev: ((fl[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in fl.columns
                 else np.ones(len(fl), bool)) for ev in EVENTS}
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(7).permutation(np.unique(pids)))}
    fold = np.array([fold_of[p] for p in pids])
    subs = {h: stamp_extra_cols(add_cond_cols(val, h)) for h in range(1, H_MAX + 1)}
    yip_v = val["snap_offset"].to_numpy()
    yf = val["years_fwd"].to_numpy()
    vpid = val["player_id"].to_numpy()
    tick(f"{len(fl):,} fit rows; val {val.player_id.nunique():,} players")

    p = dict(BASE_PARAMS)
    p.update(bag.get("overrides", {}))
    p["min_child_weight"] = float(p.get("min_child_weight", 100)) * frac
    nr = int(bag.get("num_rounds", 295))

    def arm(feats, seed):
        X = fl[feats].values.astype(np.float32)
        q = dict(p, monotone_constraints=_mono_string(feats))

        def fitj(rows, s):
            return xgb.train(dict(q, seed=s), xgb.DMatrix(X[rows], label=Y[rows], feature_names=feats),
                             num_boost_round=nr, verbose_eval=False)
        oof = np.full(Y.shape, np.nan)
        for f in range(3):
            b = fitj(fold != f, seed + f)
            oof[fold == f] = b.predict(xgb.DMatrix(X[fold == f], feature_names=feats))
            del b
        b = fitj(np.ones(len(Y), bool), seed + 10)
        raw = {h: b.predict(xgb.DMatrix(subs[h][feats].values.astype(np.float32), feature_names=feats))
               for h in range(1, H_MAX + 1)}
        P = {}
        for k, ev in enumerate(EVENTS):
            ok = elig[ev] & era_ok & np.isfinite(oof[:, k])
            R = np.maximum.accumulate(np.column_stack([raw[h][:, k] for h in range(1, H_MAX + 1)]), axis=1)
            if Y[ok, k].sum() < 25:
                P[ev] = R
                continue
            c = HYip2Calibrator().fit(oof[ok, k], h_arr[ok], yip_arr[ok], Y[ok, k].astype(int))
            P[ev] = np.maximum.accumulate(np.column_stack(
                [c.predict(R[:, h - 1], np.full(len(R), h), yip_v) for h in range(1, H_MAX + 1)]), axis=1)
        return P

    PA = arm(featsA, 700)
    tick("baseline (deployed feature set) done")
    PB = arm(featsB, 700)
    tick("with added features done")

    rows, boots = [], []
    rng = np.random.default_rng(0)
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"):
        el = (val[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in val.columns else np.ones(len(val), bool)
        for h in EVAL_H:
            mm = (yf >= h) & el
            yy = np.asarray(realized_by_h(val[mm], ev, h), dtype=float)
            a, b = PA[ev][mm, h - 1], PB[ev][mm, h - 1]
            for name, pp in (("deployed_feats", a), ("plus_" + args.tag, b)):
                rows.append({"event": ev, "h": h, "arm": name, "pos": int(yy.sum()),
                             "ap": average_precision_score(yy, pp), "auc": roc_auc_score(yy, pp),
                             "calib": pp.mean() / yy.mean()})
            pl = vpid[mm]
            upl = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in upl}
            d = []
            for _ in range(args.n_boot):
                ix = np.concatenate([pos[q] for q in rng.choice(upl, len(upl), replace=True)])
                if yy[ix].sum():
                    d.append(average_precision_score(yy[ix], b[ix]) - average_precision_score(yy[ix], a[ix]))
            d = np.array(d)
            boots.append({"event": ev, "h": h, "d_ap": d.mean(), "lo": np.percentile(d, 2.5),
                          "hi": np.percentile(d, 97.5)})
    res, bt = pd.DataFrame(rows), pd.DataFrame(boots)
    res.to_csv(out / "metrics.csv", index=False)
    bt.to_csv(out / "bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(f"\n===== AP(plus_{args.tag}) - AP(deployed features), paired player bootstrap =====")
    print(bt.round(4).to_string(index=False))
    tick("done")


if __name__ == "__main__":
    main()
