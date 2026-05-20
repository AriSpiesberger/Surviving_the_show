"""Per-bucket precision/recall at top-N% of predictions.

For each (event × draft-bucket), shows:
  - Top-1%, top-5%, top-20% of model predictions: how many players,
    how many were actual positives (precision), what share of all
    bucket positives were captured (recall)
  - "Lift" = precision / base_rate (how much better than random for
    the bucket)

Eligible-at-snap filter applied per event so the eval measures real
prediction skill, not label leakage.

Usage:
    python -m prospects.classifier.bucket_topn \\
        --val _bucket_val_event_classifiers_v1.12_bigcal.csv \\
        --out-prefix topn_v1.12
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np


BUCKETS = ("R1", "R2-R3", "R4-R10", "R10+", "IFA")
EVENTS = ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "ALL_STAR_ONCE", "STAR")
TOP_PCTS = (0.01, 0.05, 0.20)


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


def _topn(preds: np.ndarray, labels: np.ndarray, pct: float) -> dict:
    n = len(preds)
    if n == 0 or labels.sum() == 0:
        return {"k": 0, "tp": 0, "precision": float("nan"),
                "recall": float("nan"), "lift": float("nan"),
                "pred_threshold": float("nan")}
    k = max(1, int(math.ceil(n * pct)))
    order = np.argsort(preds)[::-1]
    top = order[:k]
    tp = int(labels[top].sum())
    precision = tp / k
    base = float(labels.mean())
    lift = precision / base if base > 0 else float("nan")
    return {
        "k": k,
        "tp": tp,
        "precision": precision,
        "recall": tp / int(labels.sum()),
        "lift": lift,
        "pred_threshold": float(preds[top].min()),
    }


def _eval(rows: list[dict], event: str, bucket: str) -> dict:
    p_col = f"p_{event}"
    real_col = f"realized_{event}"
    elig_col = f"eligible_at_snap_{event}"
    items = []
    for r in rows:
        if _bucket(r) != bucket:
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
        return {}
    p = np.array([_f(r[p_col]) or 0 for r in items])
    y = np.array([int(_f(r[real_col]) or 0) for r in items])
    out = {"n": len(p), "pos": int(y.sum()),
           "base": float(y.mean())}
    for pct in TOP_PCTS:
        out[pct] = _topn(p, y, pct)
    return out


def _print_event(event: str, results: dict, lines: list[str]) -> None:
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"  EVENT: {event}")
    lines.append("=" * 100)
    # Each pct gets a sub-header per bucket
    lines.append(f"{'Bucket':<8} {'n':>5} {'pos':>4} {'base':>6} "
                 + " | ".join(
                     f"{'top'+str(int(p*100))+'%':<25}"
                     for p in TOP_PCTS
                 ))
    lines.append(f"{'':>8} {'':>5} {'':>4} {'':>6} "
                 + " | ".join(
                     f"{'k':>3} {'tp':>2} {'prec':>5} {'rec':>5} {'lift':>5}"
                     for _ in TOP_PCTS
                 ))
    lines.append("-" * 100)
    for b in BUCKETS:
        r = results.get(b)
        if r is None or r.get("n", 0) == 0:
            lines.append(f"{b:<8} (no eligible rows)")
            continue
        row = f"{b:<8} {r['n']:>5d} {r['pos']:>4d} {r['base']:>6.3f} "
        cells = []
        for pct in TOP_PCTS:
            t = r[pct]
            if t["k"] == 0:
                cells.append(f"{'n/a':>25}")
            else:
                prec_s = (f"{t['precision']:>5.2f}"
                          if t["precision"] == t["precision"] else "  n/a")
                rec_s = (f"{t['recall']:>5.2f}"
                         if t["recall"] == t["recall"] else "  n/a")
                lift_s = (f"{t['lift']:>5.1f}"
                          if t["lift"] == t["lift"] else "  n/a")
                cells.append(
                    f"{t['k']:>3d} {t['tp']:>2d} {prec_s} {rec_s} {lift_s}"
                )
        row += " | ".join(cells)
        lines.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val",
                    default="_bucket_val_event_classifiers_v1.12_bigcal.csv")
    ap.add_argument("--out-prefix", default="topn_v1.12")
    args = ap.parse_args()

    with open(args.val, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} validation rows from {args.val}")

    all_results: dict[str, dict[str, dict]] = {}
    for event in EVENTS:
        all_results[event] = {}
        for b in BUCKETS:
            r = _eval(rows, event, b)
            if r:
                all_results[event][b] = r

    lines = []
    lines.append("=" * 100)
    lines.append("  PER-BUCKET TOP-N% PRECISION / RECALL / LIFT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("For each (event × bucket): pick the top-N% of model")
    lines.append("predictions and measure how many are actual positives.")
    lines.append("  k       = number of players in top-N% slice")
    lines.append("  tp      = of those, how many actually triggered the event")
    lines.append("  prec    = tp / k")
    lines.append("  rec     = tp / total bucket positives")
    lines.append("  lift    = prec / base_rate (how much better than random "
                 "within the bucket)")
    lines.append("")
    lines.append("Filter: eligible-at-snap (event not yet triggered at snap).")

    for event in EVENTS:
        _print_event(event, all_results[event], lines)

    text = "\n".join(lines)
    out_txt = f"{args.out_prefix}_topn.txt"
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nWrote {out_txt}")

    # CSV
    csv_rows = []
    for event in EVENTS:
        for b in BUCKETS:
            r = all_results[event].get(b)
            if r is None:
                continue
            for pct in TOP_PCTS:
                t = r[pct]
                csv_rows.append({
                    "event": event,
                    "bucket": b,
                    "n": r["n"],
                    "positives": r["pos"],
                    "base_rate": round(r["base"], 5),
                    "top_pct": pct,
                    "k": t["k"],
                    "tp": t["tp"],
                    "precision": (round(t["precision"], 4)
                                  if t["precision"] == t["precision"] else None),
                    "recall": (round(t["recall"], 4)
                               if t["recall"] == t["recall"] else None),
                    "lift": (round(t["lift"], 4)
                             if t["lift"] == t["lift"] else None),
                    "pred_threshold": (round(t["pred_threshold"], 5)
                                       if t["pred_threshold"] == t["pred_threshold"]
                                       else None),
                })
    if csv_rows:
        out_csv = f"{args.out_prefix}_topn.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
