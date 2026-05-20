"""Full per-bucket evaluation report.

Reads a validation CSV (from validation_predictions.py) and produces a
comprehensive report per (event x draft-bucket) cell:

  - Aggregate metrics: n, positives, base rate, mean predicted, AUC,
    Brier skill, log-loss skill, ECE
  - Precision@top-N% (top decile and top quartile of predictions land
    on what fraction of actual positives)
  - Recall@top-N% (what share of actual positives are in the top decile/
    quartile of predictions)

Eligibility filter is applied per event — only rows where the event had
NOT yet triggered as of the snapshot year are counted, so the eval
measures real prediction skill (not label leakage).

Output:
  - Text report (eval_v1.11_bucket_report.txt) for reading
  - CSV (eval_v1.11_bucket_report.csv) for downstream analysis

Usage:
    python -m prospects.classifier.bucket_report \\
        --val _bucket_val_event_classifiers_v1.11_bigcal.csv \\
        --out-prefix eval_v1.11
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


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


def _ece(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    if len(preds) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    total = 0.0
    for i in range(n_bins):
        mask = (preds >= edges[i]) & (preds < edges[i + 1])
        if not mask.any():
            continue
        total += (mask.sum() / len(preds)) * abs(
            preds[mask].mean() - labels[mask].mean()
        )
    return total


def _precision_recall_at_pct(preds: np.ndarray, labels: np.ndarray,
                             pct: float) -> tuple[float, float]:
    n = len(preds)
    if n == 0 or labels.sum() == 0:
        return float("nan"), float("nan")
    k = max(1, int(round(n * pct)))
    order = np.argsort(preds)[::-1]
    top = order[:k]
    tp = int(labels[top].sum())
    precision = tp / k
    recall = tp / int(labels.sum())
    return precision, recall


def _eval_bucket(rs: list[dict], event: str) -> dict | None:
    p_col = f"p_{event}"
    real_col = f"realized_{event}"
    elig_col = f"eligible_at_snap_{event}"
    items = []
    for r in rs:
        if r.get(p_col) in (None, ""):
            continue
        if r.get(real_col) in (None, ""):
            continue
        if elig_col in r and r[elig_col] not in (None, "", "1"):
            try:
                if int(r[elig_col]) != 1:
                    continue
            except (TypeError, ValueError):
                pass
        items.append(r)
    if not items:
        return None
    p = np.array([_f(r[p_col]) or 0 for r in items])
    y = np.array([int(_f(r[real_col]) or 0) for r in items])
    n = len(y)
    pos = int(y.sum())
    base = float(y.mean())

    if 0 < pos < n:
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = float("nan")
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        ll = log_loss(y, clipped)
        base_pred = np.clip(np.full_like(p, base), 1e-7, 1 - 1e-7)
        base_ll = log_loss(y, base_pred) if 0 < base < 1 else float("nan")
        ll_skill = (1 - ll / base_ll) if (base_ll and base_ll > 0) else float("nan")
    else:
        auc = float("nan")
        ll = float("nan")
        ll_skill = float("nan")

    brier = brier_score_loss(y, p)
    base_brier = base * (1 - base)
    brier_skill = (1 - brier / base_brier) if base_brier > 0 else float("nan")

    p10, r10 = _precision_recall_at_pct(p, y, 0.10)
    p25, r25 = _precision_recall_at_pct(p, y, 0.25)

    return {
        "n": n, "pos": pos, "base": base, "mean_p": float(p.mean()),
        "auc": auc, "brier": brier, "brier_skill": brier_skill,
        "ll": ll, "ll_skill": ll_skill, "ece": _ece(p, y),
        "p_at_10pct": p10, "r_at_10pct": r10,
        "p_at_25pct": p25, "r_at_25pct": r25,
    }


def _fmt(v, w=6, prec=3, sign=False):
    if v is None or (isinstance(v, float) and (v != v)):
        return f"{'n/a':>{w}}"
    fmt = f">+{w}.{prec}f" if sign else f">{w}.{prec}f"
    return f"{v:{fmt}}"


def _write_text(report_path: str, all_results: dict) -> None:
    lines = []
    lines.append("=" * 100)
    lines.append("  v1.11 BUCKET EVALUATION REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Filter: eligible-at-snap (rows where the event had NOT yet")
    lines.append("triggered at snapshot year). Brier skill = 1 - model_Brier /")
    lines.append("base_rate_Brier (positive = beats trivial base-rate predictor).")
    lines.append("Precision/Recall @ N% computed on top-N% of model predictions.")
    lines.append("")

    for event in EVENTS:
        res_by_bucket = all_results.get(event, {})
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"  EVENT: {event}")
        lines.append("=" * 100)
        header = (f"{'Bucket':<8} {'n':>5} {'pos':>4} {'base':>7} "
                  f"{'mean_p':>8} {'AUC':>5} {'Brsk':>7} {'LLsk':>7} "
                  f"{'ECE':>6} {'P@10%':>6} {'R@10%':>6} "
                  f"{'P@25%':>6} {'R@25%':>6}")
        lines.append(header)
        lines.append("-" * len(header))
        for b in BUCKETS:
            r = res_by_bucket.get(b)
            if r is None:
                lines.append(f"{b:<8} (no eligible rows)")
                continue
            lines.append(
                f"{b:<8} {r['n']:>5d} {r['pos']:>4d} "
                f"{r['base']:>7.3f} {r['mean_p']:>8.4f} "
                f"{_fmt(r['auc'], 5, 2)} "
                f"{_fmt(r['brier_skill'], 7, 3, sign=True)} "
                f"{_fmt(r['ll_skill'], 7, 3, sign=True)} "
                f"{_fmt(r['ece'], 6, 3)} "
                f"{_fmt(r['p_at_10pct'], 6, 3)} {_fmt(r['r_at_10pct'], 6, 3)} "
                f"{_fmt(r['p_at_25pct'], 6, 3)} {_fmt(r['r_at_25pct'], 6, 3)}"
            )

    # Gain matrix: Brier-skill compact view across event x bucket
    lines.append("")
    lines.append("=" * 100)
    lines.append("  GAIN MATRIX (Brier-skill vs bucket base rate)")
    lines.append("=" * 100)
    lines.append(f"{'Event':<22}" + "".join(f"{b:>9}" for b in BUCKETS))
    for event in EVENTS:
        row = f"{event:<22}"
        for b in BUCKETS:
            r = all_results.get(event, {}).get(b)
            row += f"{_fmt(r['brier_skill'] if r else None, 9, 3, sign=True)}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    lines.append("  AUC MATRIX (ranking quality)")
    lines.append("=" * 100)
    lines.append(f"{'Event':<22}" + "".join(f"{b:>9}" for b in BUCKETS))
    for event in EVENTS:
        row = f"{event:<22}"
        for b in BUCKETS:
            r = all_results.get(event, {}).get(b)
            row += f"{_fmt(r['auc'] if r else None, 9, 3)}"
        lines.append(row)

    text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nWrote {report_path}")


def _write_csv(csv_path: str, all_results: dict) -> None:
    rows = []
    for event in EVENTS:
        for b in BUCKETS:
            r = all_results.get(event, {}).get(b)
            if r is None:
                continue
            rows.append({
                "event": event,
                "bucket": b,
                "n": r["n"],
                "positives": r["pos"],
                "base_rate": round(r["base"], 4),
                "mean_predicted": round(r["mean_p"], 5),
                "auc": (round(r["auc"], 4)
                        if r["auc"] == r["auc"] else None),
                "brier": round(r["brier"], 5),
                "brier_skill": round(r["brier_skill"], 4),
                "log_loss": (round(r["ll"], 4)
                             if r["ll"] == r["ll"] else None),
                "ll_skill": (round(r["ll_skill"], 4)
                             if r["ll_skill"] == r["ll_skill"] else None),
                "ece": round(r["ece"], 4),
                "precision_at_10pct": (round(r["p_at_10pct"], 4)
                                        if r["p_at_10pct"] == r["p_at_10pct"] else None),
                "recall_at_10pct": (round(r["r_at_10pct"], 4)
                                     if r["r_at_10pct"] == r["r_at_10pct"] else None),
                "precision_at_25pct": (round(r["p_at_25pct"], 4)
                                        if r["p_at_25pct"] == r["p_at_25pct"] else None),
                "recall_at_25pct": (round(r["r_at_25pct"], 4)
                                     if r["r_at_25pct"] == r["r_at_25pct"] else None),
            })
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val",
                    default="_bucket_val_event_classifiers_v1.11_bigcal.csv")
    ap.add_argument("--out-prefix", default="eval_v1.11")
    args = ap.parse_args()

    with open(args.val, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} validation rows from {args.val}")

    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    for r in rows:
        b = _bucket(r)
        if b in by_bucket:
            by_bucket[b].append(r)
    print("Per-bucket cohort sizes:")
    for b in BUCKETS:
        print(f"  {b}: {len(by_bucket[b]):,}")

    all_results: dict[str, dict[str, dict]] = {}
    for event in EVENTS:
        all_results[event] = {}
        for b in BUCKETS:
            r = _eval_bucket(by_bucket[b], event)
            if r is not None:
                all_results[event][b] = r

    _write_text(f"{args.out_prefix}_bucket_report.txt", all_results)
    _write_csv(f"{args.out_prefix}_bucket_report.csv", all_results)


if __name__ == "__main__":
    main()
