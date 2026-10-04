"""Full-size spec subsets from one v3 bundle (2026-09-28).

A v3 bundle keeps every component head it trained, so any subset of its spec can be scored
without retraining: copy the bundle with the new spec, refit the held-out recalibration on the
copy, run the evaluation packet. Prints weighted AP and per-event AP for each variant.

    python -m prospects.model.train.spec_subsets --bundle runs/experiments/v3_td_heldout/v3.pkl \
        --cal runs/experiments/v3_td_heldout/calibrators_v3.pkl \
        --variant "star_tdcond_haz:STAR_PLUS_ELITE=tdcond,haz"
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    b0 = pickle.load(open(a.bundle, "rb"))
    c0 = pickle.load(open(a.cal, "rb"))
    root = Path(a.out_root or Path(a.bundle).parent / "subsets")
    rows = []
    for v in a.variant:
        name, body = v.split(":", 1)
        spec = {ev: list(c) for ev, c in b0["spec"].items()}
        for part in body.split(";"):
            ev, comps = part.split("=")
            missing = [c for c in comps.split(",") if c != "base" and c not in b0["heads"][ev]]
            assert not missing, f"{name}: {ev} has no trained head for {missing}"
            spec[ev] = comps.split(",")
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        b = dict(b0, spec=spec)
        pickle.dump(b, open(d / "v3.pkl", "wb"))
        c = dict(c0, v3_spec=spec)
        c["calibrators"] = {e: k for e, k in c0["calibrators"].items() if e not in b0["events"]}
        pickle.dump(c, open(d / "cal.pkl", "wb"))
        py = sys.executable
        subprocess.run([py, "-m", "prospects.model.v3", "recalibrate", "--heldout-xgb", str(d / "v3.pkl"),
                        "--heldout-cal", str(d / "cal.pkl"), "--write", str(d / "cal.pkl")], check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run([py, "-m", "prospects.evaluation.run", "--xgb", str(d / "v3.pkl"), "--calibrators",
                        str(d / "cal.pkl"), "--threshold", "0.6", "--out-dir", str(d / "evaluation")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        hl = json.load(open(d / "evaluation" / "headline.json"))
        ph = pd.read_csv(d / "evaluation" / "per_horizon.csv")
        r = {"variant": name, "weighted_ap": hl["weighted_ap"]}
        for _, x in ph[ph.horizon.isin([3, 6])].iterrows():
            r[f"{x.event[:4]}_h{x.horizon}"] = x.ap
        rows.append(r)
        print(pd.DataFrame([r]).round(4).to_string(index=False), flush=True)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
