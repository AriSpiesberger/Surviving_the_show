"""Per-bucket reliability (calibration) tables.

Answers: when the model says P=X% for a player in bucket B, what's the
realized rate? Aggregate calibration can look clean while individual
buckets are biased — e.g., the model is well-calibrated on the IFA
mid-bucket but systematically over-predicts on R1 top-decile.

For each (event, draft_bucket):
  - Bin predictions into adaptive bins (decile boundaries when there's
    enough data, fewer bins when the bucket is small).
  - For each bin: n, predicted mean, realized rate, gap (pred - real),
    95% Wilson confidence interval on the realized rate.

Eligibility filter applied per event (eligible_at_snap_<E> == 1).

Usage:
    python -m prospects.classifier.calibration_report \\
        --val _bucket_val_event_classifiers_v1.11_bigcal.csv \\
        --out-prefix calib_v1.11
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np


BUCKETS = ("R1", "R2-R3", "R4-R10", "R10+", "IFA")
EVENTS = ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "ALL_STAR_ONCE", "STAR")


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _bucket(r):
    intl = int(_f(r.get("is_international")) or 0)
    if intl == 1:
        return "IFA"
    rd = _f(r.get("draft_round"))
    if rd is None:
        return "UNK"
    rd = int(rd)
    if rd == 1: return "R1"
    if rd <= 3: return "R2-R3"
    if rd <= 10: return "R4-R10"
    return "R10+"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval on a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n
                                      + z**2 / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _adaptive_bins(preds: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Bin predictions. Use deciles when n >= 200, quartiles 50-200,
    halves 20-50, single bin <20."""
    n = len(preds)
    if n == 0:
        return []
    if n >= 200:
        n_bins = 10
    elif n >= 50:
        n_bins = 4
    elif n >= 20:
        n_bins = 2
    else:
        n_bins = 1
    order = np.argsort(preds)
    bins = []
    for i in range(n_bins):
        lo = (i * n) // n_bins
        hi = ((i + 1) * n) // n_bins
        if hi <= lo:
            continue
        idx = order[lo:hi]
        p = preds[idx]
        y = labels[idx]
        k = int(y.sum())
        cn = len(y)
        wl, wh = _wilson(k, cn)
        bins.append({
            "n": cn,
            "pos": k,
            "pred_lo": float(p.min()),
            "pred_hi": float(p.max()),
            "pred_mean": float(p.mean()),
            "real_rate": float(k / cn),
            "wilson_lo": float(wl),
            "wilson_hi": float(wh),
            "gap": float(p.mean() - k / cn),
            "in_band": (wl <= p.mean() <= wh),
        })
    return bins


def _eval(rs: list[dict], event: str) -> dict:
    p_col = f"p_{event}"
    real_col = f"realized_{event}"
    elig_col = f"eligible_at_snap_{event}"
    by_bucket: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for b in BUCKETS:
        items = []
        for r in rs:
            if _bucket(r) != b:
                continue
            if r.get(p_col) in (None, ""):
                continue
            if elig_col in r:
                try:
                    if int(r[elig_col]) != 1:
                        continue
                except (TypeError, ValueError):
                    pass
            items.append(r)
        if not items:
            by_bucket[b] = (np.array([]), np.array([]))
            continue
        p = np.array([_f(r[p_col]) or 0 for r in items])
        y = np.array([int(_f(r[real_col]) or 0) for r in items])
        by_bucket[b] = (p, y)
    return by_bucket


def _write_report(path: str, all_data: dict) -> None:
    lines = []
    lines.append("=" * 100)
    lines.append("  PER-BUCKET RELIABILITY (CALIBRATION) REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("For each (event, bucket): bin predictions adaptively")
    lines.append("(deciles if n>=200, quartiles 50-200, halves 20-50). For")
    lines.append("each bin show predicted mean, realized rate, 95% Wilson CI,")
    lines.append("and whether predicted mean falls within the CI (in_band=Y).")
    lines.append("If pred_mean is consistently OUTSIDE the Wilson CI within a")
    lines.append("bucket, calibration is failing for that bucket.")

    for event in EVENTS:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"  EVENT: {event}")
        lines.append("=" * 100)
        by_bucket = all_data[event]
        for b in BUCKETS:
            preds, labels = by_bucket.get(b, (np.array([]), np.array([])))
            n = len(preds)
            if n == 0:
                continue
            pos = int(labels.sum())
            lines.append("")
            lines.append(f"--- {b}  (n={n}, positives={pos}, "
                         f"base={pos/n:.3f}) ---")
            bins = _adaptive_bins(preds, labels)
            if not bins:
                lines.append("  (no bins)")
                continue
            lines.append(f"  {'bin':<3} {'n':>4} {'pos':>3} "
                         f"{'pred_range':<16} "
                         f"{'pred_mean':>9} {'real':>6} "
                         f"{'wilson95%':<18} {'gap':>7} {'in_band':>7}")
            for i, bn in enumerate(bins, 1):
                wilson_str = f"[{bn['wilson_lo']:.3f},{bn['wilson_hi']:.3f}]"
                rng = f"[{bn['pred_lo']:.3f},{bn['pred_hi']:.3f}]"
                in_band = "Y" if bn["in_band"] else "N"
                lines.append(
                    f"  {i:<3d} {bn['n']:>4d} {bn['pos']:>3d} "
                    f"{rng:<16} "
                    f"{bn['pred_mean']:>9.4f} {bn['real_rate']:>6.3f} "
                    f"{wilson_str:<18} {bn['gap']:>+7.3f} {in_band:>7}"
                )
            # Bucket-level summary: how many bins fall in band?
            in_n = sum(1 for bn in bins if bn["in_band"])
            lines.append(f"  bins_in_band: {in_n}/{len(bins)}")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nWrote {path}")


def _write_csv(path: str, all_data: dict) -> None:
    rows = []
    for event in EVENTS:
        by_bucket = all_data[event]
        for b in BUCKETS:
            preds, labels = by_bucket.get(b, (np.array([]), np.array([])))
            if len(preds) == 0:
                continue
            bins = _adaptive_bins(preds, labels)
            for i, bn in enumerate(bins, 1):
                rows.append({
                    "event": event,
                    "bucket": b,
                    "bin": i,
                    "n": bn["n"],
                    "positives": bn["pos"],
                    "pred_lo": round(bn["pred_lo"], 5),
                    "pred_hi": round(bn["pred_hi"], 5),
                    "pred_mean": round(bn["pred_mean"], 5),
                    "real_rate": round(bn["real_rate"], 5),
                    "wilson_lo": round(bn["wilson_lo"], 5),
                    "wilson_hi": round(bn["wilson_hi"], 5),
                    "gap": round(bn["gap"], 5),
                    "in_band": int(bn["in_band"]),
                })
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val",
                    default="_bucket_val_event_classifiers_v1.11_bigcal.csv")
    ap.add_argument("--out-prefix", default="calib_v1.11")
    args = ap.parse_args()

    with open(args.val, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} validation rows from {args.val}")

    all_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for event in EVENTS:
        all_data[event] = _eval(rows, event)

    _write_report(f"{args.out_prefix}_calibration.txt", all_data)
    _write_csv(f"{args.out_prefix}_calibration.csv", all_data)


if __name__ == "__main__":
    main()
