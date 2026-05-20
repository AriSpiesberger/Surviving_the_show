"""Per-round prediction summary on the live 2026 cohort.

Sanity check: a well-behaved model should rank R1 > R2 > R3 > ... on
expected outcomes, with IFAs distributed across the curve (mostly
trailing R1 but ahead of late rounds). If predictions for R1 picks
look similar to R20 picks, the model isn't using pedigree.

Usage:
    python -m prospects.classifier.per_round_predictions \\
        --grades grades_probs_2026_v7.csv
"""
from __future__ import annotations

import argparse
import csv
from statistics import mean, median


EVENTS = ("p_MLB_DEBUT", "p_ESTABLISHED_MLB", "p_STAR", "p_ELITE")
RAW_EVENTS = ("p_MLB_DEBUT_raw", "p_ESTABLISHED_MLB_raw",
              "p_STAR_raw", "p_ELITE_raw")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _bucket_for(r):
    intl = int(_f(r.get("is_international")) or 0)
    if intl == 1:
        return "IFA"
    rd = _f(r.get("draft_round"))
    if rd is None:
        return "UNK"
    rd = int(rd)
    if rd == 1: return "R1"
    if rd == 2: return "R2"
    if rd == 3: return "R3"
    if rd <= 5: return "R4-5"
    if rd <= 10: return "R6-10"
    if rd <= 20: return "R11-20"
    return "R21+"


BUCKET_ORDER = ("R1", "R2", "R3", "R4-5", "R6-10", "R11-20", "R21+", "IFA", "UNK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="grades_probs_2026_v7.csv")
    ap.add_argument("--out", default="per_round_v7.txt")
    args = ap.parse_args()

    with open(args.grades, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} rows from {args.grades}")

    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKET_ORDER}
    for r in rows:
        by_bucket[_bucket_for(r)].append(r)

    lines = []
    lines.append(f"Per-round predictions from {args.grades}")
    lines.append("=" * 105)

    # Calibrated probabilities
    lines.append("\nCalibrated probabilities (mean / median):")
    header = (f"  {'bucket':<8} {'n':>5}  "
              f"{'P(MLB) mean':>12} {'P(MLB) med':>11}  "
              f"{'P(Est) mean':>12} {'P(Est) med':>11}  "
              f"{'P(STAR) mean':>13} {'P(STAR) med':>12}")
    lines.append(header)
    lines.append("-" * len(header))
    for b in BUCKET_ORDER:
        rs = by_bucket[b]
        if not rs:
            continue
        mlb = [_f(r["p_MLB_DEBUT"]) for r in rs if _f(r["p_MLB_DEBUT"]) is not None]
        est = [_f(r["p_ESTABLISHED_MLB"]) for r in rs if _f(r["p_ESTABLISHED_MLB"]) is not None]
        star = [_f(r["p_STAR"]) for r in rs if _f(r["p_STAR"]) is not None]
        lines.append(
            f"  {b:<8} {len(rs):>5,d}  "
            f"{mean(mlb):>12.4f} {median(mlb):>11.4f}  "
            f"{mean(est):>12.4f} {median(est):>11.4f}  "
            f"{mean(star):>13.4f} {median(star):>12.4f}"
        )

    # Top-quantile counts: how many in each bucket land in the cohort top-N
    lines.append("\nTop-N representation per bucket (where the high-edge prospects live):")
    composites = [(r["player_id"], _f(r["composite_score"]) or 0.0, r) for r in rows]
    composites.sort(key=lambda t: -t[1])
    for cutoff in (10, 25, 50, 100, 250):
        top_pids = {pid for pid, _, _ in composites[:cutoff]}
        line = f"  top {cutoff:>4}:  "
        for b in BUCKET_ORDER:
            rs = by_bucket[b]
            in_top = sum(1 for r in rs if r["player_id"] in top_pids)
            line += f"{b}={in_top:>3} "
        lines.append(line)

    # Average composite per bucket — sanity check that R1 > R20 etc.
    lines.append("\nComposite score (mean / median) per bucket:")
    lines.append(f"  {'bucket':<8} {'n':>5}  {'mean':>10}  {'median':>10}  "
                 f"{'p25':>8}  {'p75':>8}")
    for b in BUCKET_ORDER:
        rs = by_bucket[b]
        if not rs:
            continue
        comps = sorted(_f(r["composite_score"]) or 0.0 for r in rs)
        n = len(comps)
        lines.append(
            f"  {b:<8} {n:>5,d}  "
            f"{mean(comps):>10.3f}  {median(comps):>10.3f}  "
            f"{comps[n//4]:>8.3f}  {comps[(3*n)//4]:>8.3f}"
        )

    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
