"""Aggregate calibration reliability per event.

Bins predictions by predicted-probability RANGE (not by quantile of small
sub-cohorts). This pools across buckets to get useful sample sizes per
bin, so the question becomes:

  "If the model predicts P=5% for a player, does the realized rate of
   those predictions actually sit at 5%?"

The earlier per-bucket calibration was statistically noisy because each
bucket had 50-100 players split into 4 bins — 4-7 positives per cell
gives Wilson CIs from 0.01 to 0.30, basically useless for calibration
validation.

Output per event: bins, n, predicted mean, realized rate, Wilson 95% CI.
Also computes ECE (expected calibration error) as a single summary.

Usage:
    python -m prospects.classifier.calibration_aggregate \\
        --val _bucket_val_event_classifiers_v1.11_bigcal.csv \\
        --out-prefix calib_agg_v1.11
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np


EVENTS = ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "ALL_STAR_ONCE", "STAR")

# Predicted-probability bin edges. Logarithmic-ish to give resolution at
# the rare-event end where almost all rare-event predictions live.
BIN_EDGES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0001]


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _ece(p: np.ndarray, y: np.ndarray, edges: list[float]) -> float:
    """Expected calibration error using the same bins."""
    n = len(p)
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        total += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return total


def _evaluate(rows: list[dict], event: str) -> dict:
    p_col = f"p_{event}"
    real_col = f"realized_{event}"
    elig_col = f"eligible_at_snap_{event}"
    items = []
    for r in rows:
        if r.get(p_col) in (None, ""):
            continue
        if r.get(real_col) in (None, ""):
            continue
        if elig_col in r:
            try:
                if int(r[elig_col]) != 1:
                    continue
            except (TypeError, ValueError):
                pass
        items.append(r)
    if not items:
        return {}
    p = np.array([_f(r[p_col]) or 0 for r in items])
    y = np.array([int(_f(r[real_col]) or 0) for r in items])

    bins = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pos = int(y[mask].sum())
        pmean = float(p[mask].mean())
        real = pos / n
        wl, wh = _wilson(pos, n)
        bins.append({
            "lo": lo, "hi": hi,
            "n": n, "pos": pos,
            "pred_mean": pmean,
            "real_rate": real,
            "wilson_lo": wl, "wilson_hi": wh,
            "in_band": wl <= pmean <= wh,
            "gap": pmean - real,
        })
    return {
        "bins": bins,
        "total_n": len(p),
        "total_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "mean_p": float(p.mean()),
        "ece": _ece(p, y, BIN_EDGES),
    }


def _print_event(event: str, res: dict, lines: list[str]) -> None:
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"  EVENT: {event}   "
                 f"(n={res['total_n']:,}, positives={res['total_pos']}, "
                 f"base={res['base_rate']:.4f}, "
                 f"mean_p={res['mean_p']:.4f}, "
                 f"ECE={res['ece']:.4f})")
    lines.append("=" * 100)
    lines.append(f"  {'pred range':<14} {'n':>5} {'pos':>4} "
                 f"{'pred_mean':>9} {'real_rate':>9} "
                 f"{'wilson95%':<18} {'gap':>7} {'in':>3}")
    in_band_count = 0
    for b in res["bins"]:
        rng = f"[{b['lo']:.3f},{b['hi']:.3f})"
        wilson = f"[{b['wilson_lo']:.3f},{b['wilson_hi']:.3f}]"
        ib = "Y" if b["in_band"] else "N"
        if b["in_band"]:
            in_band_count += 1
        lines.append(
            f"  {rng:<14} {b['n']:>5d} {b['pos']:>4d} "
            f"{b['pred_mean']:>9.4f} {b['real_rate']:>9.4f} "
            f"{wilson:<18} {b['gap']:>+7.4f} {ib:>3}"
        )
    lines.append(f"  bins_in_band: {in_band_count}/{len(res['bins'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val",
                    default="_bucket_val_event_classifiers_v1.11_bigcal.csv")
    ap.add_argument("--out-prefix", default="calib_agg_v1.11")
    args = ap.parse_args()

    with open(args.val, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} rows from {args.val}")

    lines = []
    lines.append("=" * 100)
    lines.append("  AGGREGATE CALIBRATION (pooled across draft buckets)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("For each event, predictions are binned by absolute predicted")
    lines.append("probability ranges. Per bin: n, positives, predicted mean,")
    lines.append("realized rate, 95% Wilson CI. 'in_band' = predicted bin mean")
    lines.append("falls inside the Wilson CI on the realized rate. Filter:")
    lines.append("eligible-at-snap (event had not yet triggered at snapshot).")
    lines.append("ECE = expected calibration error (weighted |pred - real|).")

    csv_rows = []
    for event in EVENTS:
        res = _evaluate(rows, event)
        if not res:
            continue
        _print_event(event, res, lines)
        for b in res["bins"]:
            csv_rows.append({
                "event": event,
                "pred_lo": b["lo"],
                "pred_hi": b["hi"],
                "n": b["n"],
                "positives": b["pos"],
                "pred_mean": round(b["pred_mean"], 5),
                "real_rate": round(b["real_rate"], 5),
                "wilson_lo": round(b["wilson_lo"], 5),
                "wilson_hi": round(b["wilson_hi"], 5),
                "gap": round(b["gap"], 5),
                "in_band": int(b["in_band"]),
            })

    text = "\n".join(lines)
    with open(f"{args.out_prefix}.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nWrote {args.out_prefix}.txt")

    if csv_rows:
        with open(f"{args.out_prefix}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Wrote {args.out_prefix}.csv")


if __name__ == "__main__":
    main()
