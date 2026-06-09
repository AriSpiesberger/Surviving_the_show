"""Per-event AP on an OOF long CSV, reported two ways: full val and the
entry-year>=2013 slice (where FG/TWTC scouting grades actually exist).

Works on any *_long.csv carrying p_<event>/eligible_<event>/realized_<event>
+ entry_year (hazard val_long or an XGB-scored val).

Usage:
    python -m scripts_v17.train.eval_oof_dual results/training/v2.0b_oof_val_long.csv
    python -m scripts_v17.train.eval_oof_dual <a.csv> <b.csv>   # compare two
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
WEIGHTS = {"TOP_100_PROSPECT": 1.0, "MLB_DEBUT": 2.0,
           "ESTABLISHED_MLB": 1.0, "STAR_PLUS_ELITE": 1.0}
MAX_ENTRY = 2020


def per_event(df: pd.DataFrame):
    rows, wap, wt = [], 0.0, 0.0
    for ev in EVENTS:
        sub = df[df.get(f"eligible_{ev}", 0) == 1]
        if sub.empty:
            continue
        y = sub[f"realized_{ev}"].astype(int).values
        p = sub[f"p_{ev}"].astype(float).values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ap = float(average_precision_score(y, p))
        auc = float(roc_auc_score(y, p))
        rows.append((ev, y.mean(), ap, ap / y.mean(), auc, len(y)))
        wap += WEIGHTS[ev] * ap
        wt += WEIGHTS[ev]
    return rows, (wap / wt if wt else float("nan"))


def report(name, df):
    df = df[df.entry_year <= MAX_ENTRY]
    print(f"\n### {name}  ({len(df):,} rows)")
    for label, d in [("FULL val", df),
                     ("entry>=2013", df[df.entry_year >= 2013])]:
        rows, w = per_event(d)
        print(f"  -- {label} ({d.player_id.nunique():,} players) --")
        print(f"     {'event':<20}{'base%':>7}{'AP':>7}{'lift':>6}{'AUC':>6}{'n':>8}")
        for ev, base, ap, lift, auc, n in rows:
            print(f"     {ev:<20}{base*100:>6.1f}{ap:>7.3f}{lift:>6.1f}{auc:>6.3f}{n:>8,}")
        print(f"     weighted-AP = {w:.4f}")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: eval_oof_dual <long.csv> [<long2.csv>]")
    for path in sys.argv[1:]:
        report(path.split("/")[-1].split("\\")[-1], pd.read_csv(path))


if __name__ == "__main__":
    main()
