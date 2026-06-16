"""Per-snap BUY LISTS for the held-out 2021-entry class.

Runs the production buy-list builder (build_v2.0_buylist.py) once per snap year
2022..2026 on the 2021 cohort, so each CSV is the buy list *as it would have
looked that year* — only players who hadn't debuted yet (the builder's
eligible_MLB_DEBUT==1 universe filter). Then appends realized_* outcomes so you
can see whether the flagged players actually panned out.

    python -m scripts_v17.validate.gen_wf2021_buylists
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

LONG = "results/scored/wf2021_cohort_long.csv"
XGB = "models/joint_xgb_v2.0b_oof.pkl"
XGB_CEILING = "models/joint_xgb_v2.0b_ceiling_w0.pkl"
YIP_THRESHOLDS = "models/yip_thresholds_p70.json"
PRICES = "data/prices_bowman_chrome_auto_v13.csv"
OUTDIR = Path("results/buy_lists/wf2021")
TMP = Path("scratch/wf2021")
MAX_YIP = "4"
REAL = ["realized_MLB_DEBUT", "realized_ESTABLISHED_MLB",
        "realized_STAR_PLUS_ELITE", "realized_TOP_100_PROSPECT"]


def main():
    big = pd.read_csv(LONG)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for sy in range(2022, 2027):
        sub = big[big.snap_year == sy]
        lp = TMP / f"long_{sy}.csv"
        sub.to_csv(lp, index=False)
        all_p = OUTDIR / f"2021class_snap{sy}_ALL.csv"
        fin_p = OUTDIR / f"2021class_snap{sy}_FINAL.csv"
        r = subprocess.run(
            [sys.executable, "scripts_v17/buylist/build_v2.0_buylist.py",
             "--long", str(lp), "--xgb", XGB, "--xgb-ceiling", XGB_CEILING,
             "--prices", PRICES, "--max-yip", MAX_YIP,
             "--yip-thresholds", YIP_THRESHOLDS,
             "--out-all", str(all_p), "--out-final", str(fin_p)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"snap{sy} FAILED:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
            continue
        # append realized outcomes for evaluation
        real = sub[["player_id"] + [c for c in REAL if c in sub.columns]]
        n = {}
        for p in (all_p, fin_p):
            d = pd.read_csv(p).merge(real, on="player_id", how="left")
            d.to_csv(p, index=False)
            n[p] = len(d)
        print(f"snap{sy}: ALL={n[all_p]:4d}  FINAL(per-yip 70%)={n[fin_p]:3d}"
              f"  -> {fin_p.name}")


if __name__ == "__main__":
    main()
