"""Compare model versions head-to-head by draft bucket.

For each (model, event, bucket), reports n, base rate, mean predicted,
AUC, Brier skill, log-loss skill against the bucket's base rate. Skill
metrics > 0 mean the model beats predicting the bucket-level base rate
uniformly; skill < 0 means the model is *worse* than that baseline.

Usage:
    python -m prospects.classifier.bucket_compare \\
        --models models/event_classifiers_v1.4_platt.pkl \\
                 models/event_classifiers_v1.8_full.pkl
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


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


BUCKETS = ("R1", "R2-R3", "R4-R10", "R10+", "IFA")


def _f(x):
    try:
        v = float(x)
        if v != v: return None
        return v
    except (TypeError, ValueError):
        return None


def _skill(metric_model, metric_base):
    if metric_base is None or metric_base == 0:
        return float("nan")
    return 1.0 - metric_model / metric_base


def _evaluate_event(rows: list[dict], event: str, p_col: str, real_col: str):
    """For each bucket, compute n, base, mean_p, AUC, Brier skill, LL skill.

    Only counts rows where the event had NOT YET triggered as of the
    snapshot year (`eligible_at_snap_<event>` = 1). Without this filter,
    AUC for events like TOP_100_PROSPECT looks artificially high because
    many R1 picks were already on the BBC top-100 list at snapshot time,
    so the prediction trivially matches the realized label."""
    elig_col = f"eligible_at_snap_{event}"
    by_bucket: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    for r in rows:
        if r.get(p_col) in (None, "") or r.get(real_col) in (None, ""):
            continue
        # Filter to not-yet-triggered-at-snap. If the column is absent
        # (older val CSVs), fall back to the old behavior.
        if elig_col in r:
            try:
                if int(r[elig_col]) != 1:
                    continue
            except (TypeError, ValueError):
                pass
        b = _bucket(r)
        if b == "UNK":
            continue
        by_bucket[b].append(r)
    out = {}
    for b in BUCKETS:
        rs = by_bucket[b]
        if not rs:
            out[b] = None
            continue
        p = np.array([_f(r[p_col]) or 0 for r in rs])
        y = np.array([int(_f(r[real_col]) or 0) for r in rs])
        n = len(y)
        base = float(y.mean())
        mean_p = float(p.mean())
        pos = int(y.sum())
        if pos == 0 or pos == n:
            auc = float("nan")
        else:
            try:
                auc = roc_auc_score(y, p)
            except Exception:
                auc = float("nan")
        brier = brier_score_loss(y, p)
        brier_base = base * (1 - base)  # Brier of predicting base everywhere
        brier_skill = _skill(brier, brier_base) if brier_base > 0 else float("nan")
        try:
            ll = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))
        except Exception:
            ll = float("nan")
        if 0 < base < 1:
            base_pred = np.full_like(p, base)
            ll_base = log_loss(y, np.clip(base_pred, 1e-7, 1 - 1e-7))
            ll_skill = _skill(ll, ll_base)
        else:
            ll_skill = float("nan")
        out[b] = {
            "n": n, "pos": pos, "base": base, "mean_p": mean_p,
            "auc": auc, "brier": brier, "brier_skill": brier_skill,
            "ll": ll, "ll_skill": ll_skill,
        }
    return out


def _print_event_compare(event: str, results_by_model: dict):
    print(f"\n{'='*108}")
    print(f"EVENT: {event}")
    print(f"{'='*108}")
    print(f"{'Bucket':<8} | "
          + " | ".join(f"{m:^46}" for m in results_by_model))
    print(f"{'':<8} | "
          + " | ".join(f"{'n':>5} {'base':>6} {'mean_p':>7} {'AUC':>5} "
                       f"{'Brsk':>6} {'LLsk':>6}"
                       for _ in results_by_model))
    print("-" * 108)
    for b in BUCKETS:
        cells = []
        for m, res in results_by_model.items():
            row = res.get(b)
            if row is None:
                cells.append(f"{'-':>46}")
            else:
                cells.append(
                    f"{row['n']:>5d} {row['base']:>6.3f} {row['mean_p']:>7.3f} "
                    f"{row['auc']:>5.2f} {row['brier_skill']:>+6.2f} "
                    f"{row['ll_skill']:>+6.2f}"
                )
        print(f"{b:<8} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="One or more model .pkl paths")
    ap.add_argument("--events", nargs="+",
                    default=["MLB_DEBUT", "ESTABLISHED_MLB"])
    args = ap.parse_args()

    # Generate / locate a validation CSV per model
    val_csvs: dict[str, Path] = {}
    for m in args.models:
        mp = Path(m)
        label = mp.stem
        out_csv = Path(f"_bucket_val_{label}.csv")
        if not out_csv.exists():
            print(f"[gen] {out_csv} from {m}")
            r = subprocess.run(
                [sys.executable, "-u", "-m",
                 "prospects.classifier.validation_predictions",
                 "--model", m, "--out", str(out_csv)],
                check=True,
            )
        else:
            print(f"[use] {out_csv} (cached)")
        val_csvs[label] = out_csv

    # Load each CSV
    rows_by_model = {}
    for label, p in val_csvs.items():
        with open(p, encoding="utf-8") as fh:
            rows_by_model[label] = list(csv.DictReader(fh))

    # For each event, evaluate per bucket per model, then print side-by-side.
    for event in args.events:
        p_col = f"p_{event}"
        real_col = f"realized_{event}"
        results_by_model = {}
        for label, rows in rows_by_model.items():
            if rows and p_col not in rows[0]:
                continue  # model doesn't have this event
            results_by_model[label] = _evaluate_event(rows, event, p_col, real_col)
        if results_by_model:
            _print_event_compare(event, results_by_model)


if __name__ == "__main__":
    main()
