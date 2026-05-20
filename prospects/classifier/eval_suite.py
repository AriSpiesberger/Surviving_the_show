"""Full evaluation suite for a survival classifier output.

Consumes a validation CSV produced by validation_predictions.py and
prints / writes a detailed report including:

  - Aggregate metrics per event: AUC, Brier, log loss, ECE
  - Decile calibration tables (predicted bin -> observed rate)
  - Precision@N for N in {10, 25, 50, 100}
  - Cohort breakdowns: drafted vs IFA, draft round buckets

Usage:
    python -m prospects.classifier.eval_suite \\
        --val validation_v1.7.csv \\
        --out eval_v1.7.txt
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


EVENTS = ("MLB_DEBUT", "ESTABLISHED_MLB", "STAR", "ELITE")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ece(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: weighted |pred_mean - real_rate| per bin."""
    if len(preds) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    total = 0.0
    for i in range(n_bins):
        mask = (preds >= edges[i]) & (preds < edges[i + 1])
        if not mask.any():
            continue
        w = mask.sum() / len(preds)
        total += w * abs(preds[mask].mean() - labels[mask].mean())
    return total


def _decile_table(preds: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Bucket by predicted-decile; report n, pred_mean, real_rate per bucket."""
    if len(preds) == 0:
        return []
    order = np.argsort(preds)
    n = len(preds)
    rows = []
    for d in range(10):
        lo = (d * n) // 10
        hi = ((d + 1) * n) // 10
        if hi <= lo:
            continue
        idx = order[lo:hi]
        rows.append({
            "decile": d + 1,
            "n": int(hi - lo),
            "pred_low": float(preds[idx].min()),
            "pred_high": float(preds[idx].max()),
            "pred_mean": float(preds[idx].mean()),
            "real_rate": float(labels[idx].mean()),
            "positives": int(labels[idx].sum()),
        })
    return rows


def _aggregate(preds: np.ndarray, labels: np.ndarray) -> dict:
    n = len(preds)
    pos = int(labels.sum())
    if pos == 0 or pos == n:
        return {"n": n, "pos": pos, "auc": float("nan"),
                "brier": brier_score_loss(labels, preds) if n else float("nan"),
                "logloss": float("nan"),
                "ece": _ece(preds, labels),
                "pred_mean": float(preds.mean()) if n else 0,
                "real_rate": float(labels.mean()) if n else 0}
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = float("nan")
    try:
        clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        ll = log_loss(labels, clipped)
    except Exception:
        ll = float("nan")
    return {
        "n": n, "pos": pos,
        "auc": float(auc),
        "brier": float(brier_score_loss(labels, preds)),
        "logloss": float(ll),
        "ece": _ece(preds, labels),
        "pred_mean": float(preds.mean()),
        "real_rate": float(labels.mean()),
    }


def _precision_at_n(preds: np.ndarray, labels: np.ndarray,
                    ns=(10, 25, 50, 100)) -> dict:
    order = np.argsort(preds)[::-1]
    out = {}
    for n in ns:
        if n > len(order):
            out[n] = float("nan")
            continue
        top = order[:n]
        out[n] = float(labels[top].mean())
    return out


def _cohort_split(rows: list[dict]) -> dict:
    """Return rows partitioned into drafted vs IFA, plus per-round buckets."""
    drafted = [r for r in rows if int(r.get("is_international") or 0) == 0]
    ifa = [r for r in rows if int(r.get("is_international") or 0) == 1]

    def _round_bucket(r):
        rd = _f(r.get("draft_round"))
        if rd is None:
            return None
        rd = int(rd)
        if rd == 1: return "R1"
        if rd <= 3: return "R2-3"
        if rd <= 10: return "R4-10"
        if rd <= 20: return "R11-20"
        return "R21+"

    by_round: dict[str, list[dict]] = defaultdict(list)
    for r in drafted:
        b = _round_bucket(r)
        if b:
            by_round[b].append(r)

    return {
        "ALL": rows,
        "DRAFTED": drafted,
        "IFA": ifa,
        **by_round,
    }


def _print_aggregate(label: str, agg: dict, p_at_n: dict, lines: list):
    lines.append(
        f"{label:<14} n={agg['n']:<6,d} pos={agg['pos']:<5d} "
        f"AUC={agg['auc']:6.3f}  Brier={agg['brier']:7.5f}  "
        f"LogLoss={agg['logloss']:6.3f}  ECE={agg['ece']:6.4f}  "
        f"pred={agg['pred_mean']*100:5.2f}%  real={agg['real_rate']*100:5.2f}%  "
        f"P@10={p_at_n.get(10, float('nan')):.2f} "
        f"P@25={p_at_n.get(25, float('nan')):.2f} "
        f"P@50={p_at_n.get(50, float('nan')):.2f} "
        f"P@100={p_at_n.get(100, float('nan')):.2f}"
    )


def _print_decile(label: str, table: list[dict], lines: list):
    lines.append(f"\n--- {label} decile calibration ---")
    lines.append(f"{'dec':>3} {'n':>5} {'pred_range':>17} "
                 f"{'pred_mean':>10} {'real_rate':>10} {'pos':>4}")
    for row in table:
        lines.append(
            f"{row['decile']:>3d} {row['n']:>5d} "
            f"[{row['pred_low']:5.3f}, {row['pred_high']:5.3f}]   "
            f"{row['pred_mean']:>10.4f} {row['real_rate']:>10.4f} "
            f"{row['positives']:>4d}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="validation_v1.7.csv")
    ap.add_argument("--out", default="eval_v1.7.txt")
    args = ap.parse_args()

    with open(args.val, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} val rows from {args.val}")
    cohorts = _cohort_split(rows)

    lines = []
    lines.append(f"Evaluation suite for {args.val}")
    lines.append("=" * 100)
    lines.append(f"Cohort sizes: ALL={len(cohorts['ALL']):,}  "
                 f"DRAFTED={len(cohorts['DRAFTED']):,}  "
                 f"IFA={len(cohorts['IFA']):,}  |  "
                 + "  ".join(
                     f"{k}={len(cohorts.get(k, [])):,}"
                     for k in ("R1", "R2-3", "R4-10", "R11-20", "R21+")
                 ))

    for event in EVENTS:
        col = f"p_{event}"
        real_col = f"realized_{event}"
        if not rows or col not in rows[0]:
            continue
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"EVENT: {event}")
        lines.append("=" * 100)
        for c_name, c_rows in cohorts.items():
            preds = np.array([_f(r[col]) or 0 for r in c_rows])
            labels = np.array([int(_f(r[real_col]) or 0) for r in c_rows],
                              dtype=np.int8)
            agg = _aggregate(preds, labels)
            p_at_n = _precision_at_n(preds, labels)
            _print_aggregate(c_name, agg, p_at_n, lines)
        # Decile table on ALL only (subgroup deciles get noisy fast)
        preds_all = np.array([_f(r[col]) or 0 for r in cohorts["ALL"]])
        labels_all = np.array([int(_f(r[real_col]) or 0)
                               for r in cohorts["ALL"]], dtype=np.int8)
        _print_decile("ALL", _decile_table(preds_all, labels_all), lines)

    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
