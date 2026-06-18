"""Per-yip P(MLB_DEBUT) thresholds at a target PRECISION, conditional stack.

Mirrors gen_yip_thresholds.py but for the conditional joint XGB
(joint_cond.predict_trajectory -> xp_<event>_h{h}) instead of the old v2 head,
and keys off the buy-list debut horizon (P(debut <= Hy) = xp_MLB_DEBUT_h{H}).

For each yip (snap_offset) we rank the eligible, resolved val rows by p_debut
descending and walk down the ranking; the threshold is the LOWEST p at which the
running precision (TP / n_selected) still clears --target. Buying at that cutoff
gives >= target of picks actually debuting within H years on the val slice.

Caveat (same as the v2 script): thresholds are tuned on the val slice the XGB
early-stops on, so they carry mild in-sample optimism -- a buy-list knob, not a
reported metric.

    # partial model on the partial (mid-season) val set, buy horizon h=3
    python -m scripts_v17.validate.gen_yip_thresholds_cond \
        --val-long results/training/v2.0b_valpartial_partialHaz_long.csv \
        --xgb models/joint_xgb_v2.0b_partial_oof.pkl \
        --horizon 3 --target 0.70 --out models/yip_thresholds_partial_p70_h3.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.joint_cond import (  # noqa: E402
    predict_trajectory, prep_base, realized_by_h,
)

DB = str(REPO_ROOT / "prospects_snapshot.db")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-long", required=True)
    ap.add_argument("--xgb", required=True)
    ap.add_argument("--horizon", type=int, default=3,
                    help="Debut window H: score xp_MLB_DEBUT_h{H} vs "
                         "realized-within-H, on rows resolved at H. Matches the "
                         "buy-list --debut-horizon (default 3).")
    ap.add_argument("--target", type=float, default=0.70,
                    help="Minimum precision the selected set must hold.")
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--min-n", type=int, default=15,
                    help="Skip a yip with fewer than this many eligible rows.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    H = args.horizon
    ev = "MLB_DEBUT"
    p_col = f"xp_{ev}_h{H}"

    print(f"Loading {Path(args.val_long).name} + {Path(args.xgb).name}")
    df = pd.read_csv(args.val_long)
    df = prep_base(df, DB, max_entry=args.max_entry)
    df = predict_trajectory(pickle.load(open(args.xgb, "rb")), df)

    # Resolved at H (label trustworthy), in the buy universe (not yet debuted),
    # and carrying a debut score.
    d = df[(df["years_fwd"] >= H) & (df.get(f"eligible_{ev}", 1) == 1)].copy()
    d["y"] = realized_by_h(d, ev, H).astype(float)
    d = d.dropna(subset=[p_col])

    thr = {}
    print(f"\nPer-yip P({ev} <= {H}y) threshold @ precision >= {args.target:.0%}")
    print(f"(val={Path(args.val_long).name}, resolved at h={H}, eligible only)")
    print(f"{'yip':>3}{'n':>7}{'base%':>8}{'thr':>8}{'n>=thr':>8}"
          f"{'precision':>11}{'recall':>8}")
    for yip in range(11):
        s = d[d["snap_offset"] == yip].sort_values(p_col, ascending=False)
        n = len(s)
        if n < args.min_n:
            continue
        y = s["y"].values
        p = s[p_col].values
        tot_pos = float(y.sum())
        cum_prec = np.cumsum(y) / np.arange(1, n + 1)
        ok = np.where(cum_prec >= args.target)[0]
        base = y.mean()
        if len(ok):
            k = int(ok[-1])               # deepest rank still >= target
            t = float(p[k])
            prec = float(cum_prec[k])
            rec = float(np.sum(y[:k + 1]) / tot_pos) if tot_pos > 0 else float("nan")
            thr[str(yip)] = round(t, 4)
            print(f"{yip:>3}{n:>7}{base*100:>7.1f}%{t:>8.3f}{k+1:>8}"
                  f"{prec:>11.3f}{rec:>8.3f}")
        else:
            print(f"{yip:>3}{n:>7}{base*100:>7.1f}%{'—':>8}{'0':>8}"
                  f"{'never':>11}{'—':>8}")

    if args.out:
        json.dump(thr, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    print(f"thresholds: {thr}")


if __name__ == "__main__":
    main()
