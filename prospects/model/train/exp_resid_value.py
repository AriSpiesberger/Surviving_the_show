"""Incremental-information test for new feature sources over the PRODUCTION v3 (2026-09-24).

Retraining the joint recipe with and without a feature set (exp_joint_feats) swings every
event's AP by ~+-0.01 on its own, which hides modest real signal. This asks the sharper
question directly: given production v3's calibrated prediction on held-out players, do the
new features still carry information about the outcome?

For each event and horizon, on the held-out val players (eligible, resolved):
  A  logistic recalibration of logit(P_v3)             (what v3 already knows)
  B  shallow GBM on [logit(P_v3), yip, new features]   (v3 + the new source)
2-fold cross-fitting by player: each half is scored by models fit on the other half.
Reported: AP(A), AP(B0) (the same GBM stage WITHOUT new features — a control for the stage itself),
AP(B), and the paired player bootstrap of AP(B) - AP(B0). A source is worth building
into v3 only if B beats A.

    python -m prospects.model.train.exp_resid_value --extra reference/transactions/tx_features.csv:2010
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import prep_base, realized_by_h
from prospects.model.joint2 import apply_calibrators_frame, load_calibrators, score_trajectory

_RUN = config.run()
DB = str(config.model_db())
EV4 = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra", nargs="+", required=True,
                    help="CSV[:fill_from_year] keyed by player_id, snap_year; several are merged")
    ap.add_argument("--tag", default="sources")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    cache = REPO_ROOT / "runs" / "experiments" / "exp_resid_value" / "v3_val_scored.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        s = pd.read_pickle(cache)
    else:
        val = prep_base(pd.read_csv(_RUN.oof_val_long, low_memory=False), DB, max_entry=2020)
        s, _ = score_trajectory(str(_RUN.models / "v3.pkl"), val, DB)
        s = apply_calibrators_frame(s, load_calibrators(_RUN.models / "calibrators_v3.pkl"))
        s.to_pickle(cache)
    cols = []
    for spec in args.extra:
        path, _, fill = spec.partition(":")
        e = pd.read_csv(path)
        c = [x for x in e.columns if x not in ("player_id", "snap_year")]
        s = s.drop(columns=[x for x in c if x in s.columns]).merge(e, on=["player_id", "snap_year"], how="left")
        if fill:
            m = s.snap_year >= int(fill)
            s.loc[m, c] = s.loc[m, c].fillna(0.0)
        cols += c
    print(f"[rv] {len(cols)} new features from {len(args.extra)} source(s); val rows {len(s):,}")
    yf, vpid, yip = s.years_fwd.to_numpy(), s.player_id.to_numpy(), s.snap_offset.to_numpy()
    upl = np.unique(vpid)
    half = np.isin(vpid, np.random.default_rng(11).choice(upl, upl.size // 2, replace=False))
    rng = np.random.default_rng(0)
    rows = []
    for ev in EV4:
        el = (s[f"eligible_{ev}"] == 1).to_numpy()
        for h in (3, 6):
            m = (yf >= h) & el
            y = np.asarray(realized_by_h(s[m], ev, h), dtype=int)
            if y.sum() < 20:
                continue
            base = logit(s.loc[m, f"xp_{ev}_h{h}"].to_numpy(float))
            XB = np.column_stack([base, yip[m], s.loc[m, cols].to_numpy(float)])
            hm = half[m]
            pa, pb, p0 = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), np.nan)
            X0 = XB[:, :2]
            for side in (True, False):
                tr, te = hm == side, hm != side
                pa[te] = LogisticRegression(max_iter=1000).fit(base[tr, None], y[tr]).predict_proba(base[te, None])[:, 1]
                prm = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.05, "min_child_weight": 5,
                       "subsample": 0.8, "colsample_bytree": 0.8, "base_score": float(y[tr].mean()),
                       "monotone_constraints": "(1" + ",0" * (XB.shape[1] - 1) + ")", "verbosity": 0}
                d = xgb.DMatrix(XB[tr], label=y[tr], base_margin=base[tr])
                b = xgb.train(prm, d, num_boost_round=150)
                pb[te] = b.predict(xgb.DMatrix(XB[te], base_margin=base[te]))
                # B0: the same second stage WITHOUT the new features (isolates the features' value)
                p0m = dict(prm, monotone_constraints="(1,0)")
                b0 = xgb.train(p0m, xgb.DMatrix(X0[tr], label=y[tr], base_margin=base[tr]), num_boost_round=150)
                p0[te] = b0.predict(xgb.DMatrix(X0[te], base_margin=base[te]))
            pl = vpid[m]
            up = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in up}
            dd = []
            for _ in range(args.n_boot):
                ix = np.concatenate([pos[q] for q in rng.choice(up, len(up), replace=True)])
                if y[ix].sum():
                    dd.append(average_precision_score(y[ix], pb[ix]) - average_precision_score(y[ix], p0[ix]))
            dd = np.array(dd)
            rows.append({"event": ev, "h": h, "pos": int(y.sum()), "ap_v3": average_precision_score(y, pa),
                         "ap_B0": average_precision_score(y, p0), "ap_B_plus": average_precision_score(y, pb),
                         "d_ap(B-B0)": dd.mean(),
                         "lo": np.percentile(dd, 2.5), "hi": np.percentile(dd, 97.5)})
    res = pd.DataFrame(rows)
    out = REPO_ROOT / "runs" / "experiments" / "exp_resid_value"
    res.to_csv(out / f"{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
