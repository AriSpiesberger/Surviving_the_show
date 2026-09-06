"""Null test: is the val-slice reliability gap explainable by sampling noise?

Every mechanical explanation for the val S-shape is ruled out (functional
form, score distribution, clustering, model strength, era). Remaining
question: how big do bucket diffs get for a RANDOM val-sized player sample
under perfect calibration? Simulate with the cached honest OOF bag
predictions: repeatedly hold out n_val players from FIT, fit the HYip2
calibrator on the rest (2008+), and measure the held sample's pooled
reliability gap on high-score rows (p >= 0.40, debut h=3) — exactly the
statistic where val shows +11pts.

If val's observed gap is inside the null distribution -> sampling noise;
report the percentile and stop chasing it. If far outside -> the split
itself is structurally different; investigate val_pids provenance.

    python -m prospects.model.train.exp_cal_null
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, prep_base
from prospects.model.joint2 import HYip2Calibrator
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols

_RUN = config.run()
DB = str(config.model_db())
CAL_DIR = REPO_ROOT / "runs" / "exp_cal_bagfit"
OUT = REPO_ROOT / "runs" / "exp_cal_bagfit" / "null_test.json"

# Observed on val (v2.3 cals, debut h=3, 2008+, snap rows with p>=0.40):
# pooled predicted ~0.703, realized ~0.814 -> gap +11.1pts. Recomputed
# below is better, but hardcode the reference printed in the report.
VAL_GAP_REF = 0.111


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(CAL_DIR / "oof_bag_preds_f6s1.npz"))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-val-players", type=int, default=2494)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    t0 = time.time()

    fit_base = _prep_train(pd.read_csv(_RUN.oof_stacked_long), DB, 2020)
    fit_long, Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    oof = np.load(args.oof)["oof"]
    assert len(oof) == len(fit_long), "OOF cache / long mismatch"

    k = EVENTS.index("MLB_DEBUT")
    elig = (fit_long["eligible_MLB_DEBUT"] == 1).to_numpy()
    era = (fit_long["snap_year"].to_numpy() >= 2008)
    okall = elig & era & np.isfinite(oof[:, k])
    p_raw = oof[okall, k]
    y = Y[okall, k].astype(int)
    h = fit_long["h"].astype(int).to_numpy()[okall]
    yip = fit_long["snap_offset"].to_numpy()[okall]
    pids = fit_long["player_id"].to_numpy()[okall]
    h3 = h == 3
    uniq = np.unique(pids)
    print(f"[null] {okall.sum():,} rows / {len(uniq):,} players; "
          f"{args.reps} reps of {args.n_val_players} held players")

    rng = np.random.default_rng(args.seed)
    gaps = []
    for rep in range(args.reps):
        held = set(rng.choice(uniq, size=args.n_val_players, replace=False))
        hm = np.isin(pids, list(held))
        cal = HYip2Calibrator().fit(p_raw[~hm], h[~hm], yip[~hm], y[~hm])
        m = hm & h3
        pc = cal.predict(p_raw[m], h[m], yip[m])
        sel = pc >= 0.40
        if sel.sum() < 50:
            continue
        gaps.append(float(y[m][sel].mean() - pc[sel].mean()))
        if (rep + 1) % 25 == 0:
            g = np.asarray(gaps)
            print(f"  rep {rep+1}: null mean={g.mean():+.3f} "
                  f"sd={g.std():.3f} max={g.max():+.3f} "
                  f"[{(time.time()-t0)/60:.1f}m]")

    g = np.asarray(gaps)
    pct = float((g < VAL_GAP_REF).mean())
    z = float((VAL_GAP_REF - g.mean()) / g.std())
    print(f"\n[null] gap statistic = realized - predicted on held rows with "
          f"p>=0.40 (debut h=3)")
    print(f"  null: mean {g.mean():+.4f}  sd {g.std():.4f}  "
          f"p5 {np.quantile(g, .05):+.4f}  p95 {np.quantile(g, .95):+.4f}  "
          f"max {g.max():+.4f}")
    print(f"  VAL observed: {VAL_GAP_REF:+.4f}  -> percentile "
          f"{pct:.1%}, z = {z:+.2f}")
    verdict = ("SAMPLING NOISE (inside null)" if pct < 0.975
               else "STRUCTURAL (outside null) — investigate the split")
    print(f"  verdict: {verdict}")
    OUT.write_text(json.dumps({
        "reps": len(g), "null_mean": float(g.mean()),
        "null_sd": float(g.std()), "null_p95": float(np.quantile(g, .95)),
        "null_max": float(g.max()), "val_gap": VAL_GAP_REF,
        "percentile": pct, "z": z, "verdict": verdict,
    }, indent=2))
    print(f"  wrote {OUT}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
