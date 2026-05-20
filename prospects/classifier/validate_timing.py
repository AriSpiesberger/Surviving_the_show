"""Validate the survival-model timing predictions against realized outcomes.

For every holdout player whose event actually triggered, compute the
realized time-to-event and compare against the predicted conditional
mean t_EVENT_mean. Report:

  - n with realized event, MAE, ME (bias), median |error|, RMSE
  - Coverage: fraction of realized t within predicted +/- SD
  - Decile calibration of predicted t vs realized t
  - Per-event ordering check: among players who realized BOTH events,
    is t_DEBUT_realized < t_EST_realized? Compare against predictions.

Usage:
    python -m prospects.classifier.validate_timing \\
        --val validation_v1.7.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np


EVENTS = ("MLB_DEBUT", "ESTABLISHED_MLB", "STAR", "ELITE")


def _f(x):
    try:
        v = float(x)
        if v != v: return None
        return v
    except (TypeError, ValueError):
        return None


def _i(x):
    try:
        return int(x) if x != "" and x is not None else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="validation_v1.7.csv")
    args = ap.parse_args()

    with open(args.val, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} val rows from {args.val}\n")

    for ev in EVENTS:
        # Realized-trigger players only
        triggered = []
        for r in rows:
            if int(_f(r.get(f"realized_{ev}")) or 0) != 1:
                continue
            snap = _i(r.get("snap_year"))
            trig = _i(r.get(f"trigger_year_{ev}"))
            t_pred = _f(r.get(f"t_{ev}_mean"))
            t_sd = _f(r.get(f"t_{ev}_sd"))
            if snap is None or trig is None or t_pred is None:
                continue
            t_real = trig - snap
            triggered.append({
                "name": r.get("name"),
                "snap": snap, "trig": trig,
                "t_real": t_real,
                "t_pred": t_pred,
                "t_sd": t_sd or 0.0,
                "err": t_pred - t_real,
                "p": _f(r.get(f"p_{ev}")) or 0.0,
            })

        n = len(triggered)
        print("=" * 80)
        print(f"EVENT: {ev}    n_triggered={n}")
        if n == 0:
            print("  (no triggered cases in holdout)")
            continue
        errs = np.array([t["err"] for t in triggered])
        reals = np.array([t["t_real"] for t in triggered])
        preds = np.array([t["t_pred"] for t in triggered])
        sds = np.array([t["t_sd"] for t in triggered])
        in_band = ((preds - sds <= reals) & (reals <= preds + sds)).mean()

        print(f"  Realized t : mean={reals.mean():.2f}  median={np.median(reals):.2f}  "
              f"min={reals.min()} max={reals.max()}")
        print(f"  Predicted t: mean={preds.mean():.2f}  median={np.median(preds):.2f}  "
              f"min={preds.min():.2f} max={preds.max():.2f}")
        print(f"  Error (pred-real): "
              f"mean={errs.mean():+.2f} (bias)  "
              f"|err| mean={np.abs(errs).mean():.2f}  median={np.median(np.abs(errs)):.2f}  "
              f"RMSE={np.sqrt((errs**2).mean()):.2f}")
        print(f"  Coverage in pred +/- 1 SD: {in_band*100:.1f}%")

        # Predicted-decile calibration: bin by predicted t, show mean realized
        order = np.argsort(preds)
        for d in range(5):
            lo = (d * n) // 5
            hi = ((d + 1) * n) // 5
            if hi <= lo: continue
            idx = order[lo:hi]
            print(f"    pred-quintile {d+1}: "
                  f"pred_mean={preds[idx].mean():.2f}  "
                  f"real_mean={reals[idx].mean():.2f}  "
                  f"(n={hi-lo})")

        # Worst miss examples
        worst = sorted(triggered, key=lambda t: -abs(t["err"]))[:5]
        print("  Worst misses:")
        for t in worst:
            print(f"    {t['name'][:24]:<24} snap={t['snap']}  "
                  f"real={t['t_real']:>2}y  pred={t['t_pred']:>5.2f}y  "
                  f"err={t['err']:+.2f}y")
        print()

    # Ordering check: among holdout players who triggered MLB_DEBUT,
    # does the predicted t_DEBUT vs t_EST (and vs t_STAR if triggered)
    # match the realized ordering?
    print("=" * 80)
    print("Structural ordering check (realized cases only):")
    pairs = [("MLB_DEBUT", "ESTABLISHED_MLB"),
             ("MLB_DEBUT", "STAR"),
             ("ESTABLISHED_MLB", "STAR")]
    for a, b in pairs:
        both = []
        for r in rows:
            if (int(_f(r.get(f"realized_{a}")) or 0) != 1
                    or int(_f(r.get(f"realized_{b}")) or 0) != 1):
                continue
            snap = _i(r.get("snap_year"))
            ta = _i(r.get(f"trigger_year_{a}"))
            tb = _i(r.get(f"trigger_year_{b}"))
            pa = _f(r.get(f"t_{a}_mean"))
            pb = _f(r.get(f"t_{b}_mean"))
            if None in (snap, ta, tb, pa, pb):
                continue
            both.append({
                "ra": ta - snap, "rb": tb - snap,
                "pa": pa, "pb": pb,
            })
        if not both:
            print(f"  {a} vs {b}: no joint-realized cases")
            continue
        realized_ordered = sum(1 for d in both if d["ra"] <= d["rb"])
        predicted_ordered = sum(1 for d in both if d["pa"] <= d["pb"])
        print(f"  {a:<18} <= {b:<18}  n={len(both):<5} "
              f"realized: {realized_ordered}/{len(both)} ({100*realized_ordered/len(both):.1f}%)  "
              f"predicted: {predicted_ordered}/{len(both)} ({100*predicted_ordered/len(both):.1f}%)")


if __name__ == "__main__":
    main()
