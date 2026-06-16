"""Generate per-yip P(MLB_DEBUT) thresholds at a target precision.

Reproducibility (v2.1): the buy list's per-yip cutoffs were previously a loose
JSON with no generating script. This regenerates models/yip_thresholds_p70.json
from the current debut model on the SYMMETRIC resolved val (years_fwd >= W, no
label selection). NOTE: thresholds are tuned on the same val slice the model
early-stops on, so they carry mild in-sample optimism -- they are a buy-list
knob, not a reported metric.

    python -m scripts_v17.validate.gen_yip_thresholds --target 0.70
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts_v17.train.fit_joint_xgb_v2 import EVENTS, FEAT, _prep

DEBUT_XGB = "models/joint_xgb_v2.0b_oof.pkl"   # debut@5 head
VAL = "results/training/v2.0b_oof_val_long.csv"
OUT = "models/yip_thresholds_p70.json"
RESOLVED_W = 5   # debut observation window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.70)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    val = _prep(pd.read_csv(VAL), "prospects_snapshot.db", 2020).reset_index(drop=True)
    # symmetric resolved + buy universe (not yet debuted at snap)
    d = val[(val.years_fwd >= RESOLVED_W) & (val.eligible_MLB_DEBUT == 1)].copy()
    b = pickle.load(open(DEBUT_XGB, "rb"))
    X = b["scaler"].transform(d[b["feature_names"]].values.astype(np.float32))
    d["p"] = b["model"].predict(xgb.DMatrix(X, feature_names=list(b["feature_names"])),
                                iteration_range=(0, b["best_iteration"] + 1))[
        :, b["events"].index("MLB_DEBUT")]

    thr = {}
    print(f"per-yip P(debut) threshold @ precision>={args.target} (symmetric val):")
    for yip in range(11):
        s = d[d.snap_offset == yip].sort_values("p", ascending=False)
        if len(s) < 15:
            continue
        y = s.realized_MLB_DEBUT.astype(float).values
        p = s.p.values
        cum = np.cumsum(y) / np.arange(1, len(y) + 1)
        ok = np.where(cum >= args.target)[0]
        if len(ok):
            thr[str(yip)] = round(float(p[ok[-1]]), 4)
            print(f"  yip={yip:2d} n={len(s):5d} base={y.mean():.2f} "
                  f"thr={thr[str(yip)]:.3f} n>=thr={ok[-1]+1}")
        else:
            print(f"  yip={yip:2d} n={len(s):5d} base={y.mean():.2f} "
                  f"-> never reaches {args.target}")
    json.dump(thr, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}: {thr}")


if __name__ == "__main__":
    main()
