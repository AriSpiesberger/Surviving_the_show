"""Walk-forward sheet for the 2021-entry cohort at snap=2022 (year 1)
and snap=2023 (year 2). Pulls from the existing v1.17 backtest CSVs.

Reports per-event:
  base%, AU-PR, AU-PR_lift, lift@2, lift@5, AUC, top-2% slab realized rate.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

IN_DIR = "backtests/v17"
OUT_DIR = "results/walkforward_2021"
EVENTS = [
    ("MLB_DEBUT", "p_MLB_DEBUT", "realized_MLB_DEBUT"),
    ("TOP_100_PROSPECT", "p_TOP_100_PROSPECT", "realized_TOP_100_PROSPECT"),
]
SNAPS = [
    (2022, "year 1"),
    (2023, "year 2"),
    (2024, "year 3"),
]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    base = float(y.mean()) if len(y) else float("nan")
    out = {"n": int(len(y)), "pos": int(y.sum()), "base%": base * 100}
    if y.sum() == 0 or y.sum() == len(y):
        out.update({"AU-PR": float("nan"), "AU-PR_lift": float("nan"),
                    "AUC": float("nan"), "Brier": float("nan"),
                    "lift@2%": float("nan"), "rate@2%": float("nan"),
                    "lift@5%": float("nan"), "rate@5%": float("nan")})
        return out
    ap = float(average_precision_score(y, p))
    auc = float(roc_auc_score(y, p))
    br = float(brier_score_loss(y, p))
    n = len(p)
    def topk(pct):
        k = max(1, int(round(pct / 100 * n)))
        idx = np.argsort(-p)[:k]
        rate = float(y[idx].mean())
        return rate, (rate / base if base else float("nan"))
    r2, l2 = topk(2)
    r5, l5 = topk(5)
    out.update({"AU-PR": ap, "AU-PR_lift": ap / base if base else float("nan"),
                "AUC": auc, "Brier": br,
                "lift@2%": l2, "rate@2%": r2 * 100,
                "lift@5%": l5, "rate@5%": r5 * 100})
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    summary_rows = []
    for snap, label in SNAPS:
        fname = f"{IN_DIR}/backtest_v1.17_2021entry_snap{snap}_nonR1.csv"
        df = pd.read_csv(fname)
        header = (f"{'='*68}\n"
                  f"2021 entry cohort — snap={snap} ({label} post-draft) "
                  f"— n={len(df):,}\n{'='*68}")
        print(header); lines.append(header)
        for ev_name, p_col, y_col in EVENTS:
            if p_col not in df.columns or y_col not in df.columns:
                continue
            y = df[y_col].values.astype(int)
            p = df[p_col].values
            m = _metrics(y, p)
            txt = (f"\n  {ev_name}\n"
                   f"  {'n':>6} {'pos':>5} {'base%':>7} {'AU-PR':>7} "
                   f"{'AU-PR_lift':>11} {'AUC':>7} {'lift@2':>7} "
                   f"{'rate@2%':>8} {'lift@5':>7} {'rate@5%':>8} {'Brier':>7}\n"
                   f"  {m['n']:>6d} {m['pos']:>5d} {m['base%']:>6.2f} "
                   f"{m['AU-PR']:>7.3f} {m['AU-PR_lift']:>11.2f} "
                   f"{m['AUC']:>7.3f} {m['lift@2%']:>7.2f} {m['rate@2%']:>7.2f} "
                   f"{m['lift@5%']:>7.2f} {m['rate@5%']:>7.2f} {m['Brier']:>7.4f}")
            print(txt); lines.append(txt)
            summary_rows.append({"snap": snap, "event": ev_name,
                                  "label": label, **m})

        # Per-yip slabs for MLB_DEBUT
        if "MLB_DEBUT" in [e[0] for e in EVENTS]:
            df_ranked = df.sort_values("p_MLB_DEBUT", ascending=False).copy()
            df_ranked["pctile_within_snap"] = (
                (np.arange(len(df_ranked)) + 0.5) / len(df_ranked) * 100)
            slabs = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2),
                     (2, 3), (3, 4), (4, 5), (5, 10), (10, 20),
                     (20, 50), (50, 100)]
            txt = (f"\n  --- MLB_DEBUT realized rate by percentile slab "
                   f"(top of list first) ---")
            print(txt); lines.append(txt)
            print(f"  {'slab':>10} {'n':>5} {'tp':>4} {'rate%':>6} "
                  f"{'lift':>6} {'score_lo':>9}")
            lines.append(f"  {'slab':>10} {'n':>5} {'tp':>4} {'rate%':>6} "
                          f"{'lift':>6} {'score_lo':>9}")
            base = float(df.realized_MLB_DEBUT.mean())
            for lo, hi in slabs:
                m_in = ((df_ranked.pctile_within_snap >= lo) &
                        (df_ranked.pctile_within_snap < hi))
                s = df_ranked[m_in]
                if len(s) == 0:
                    continue
                rate = float(s.realized_MLB_DEBUT.mean())
                lift = rate / base if base else float("nan")
                label_str = f"{lo}-{hi}%"
                txt = (f"  {label_str:>10} {len(s):>5d} "
                       f"{int(s.realized_MLB_DEBUT.sum()):>4d} "
                       f"{rate*100:>5.1f} {lift:>5.2f}x "
                       f"{s.p_MLB_DEBUT.min():>+9.3f}")
                print(txt); lines.append(txt)

        # Top 2% picks with realized
        n = len(df)
        k2 = max(1, int(round(0.02 * n)))
        top = df.sort_values("p_MLB_DEBUT", ascending=False).head(k2)
        cols = [c for c in ("name", "bucket", "current_org", "cur_level",
                              "p_MLB_DEBUT", "realized_MLB_DEBUT")
                if c in top.columns]
        out_top = f"{OUT_DIR}/top2pct_snap{snap}.csv"
        top[cols].to_csv(out_top, index=False)
        msg = (f"\n  saved top-2% picks ({k2} players) -> {out_top}\n"
               f"  among top 2%, realized_MLB_DEBUT = "
               f"{int(top.realized_MLB_DEBUT.sum())}/{k2} "
               f"({top.realized_MLB_DEBUT.mean()*100:.1f}%)")
        print(msg); lines.append(msg)

    # Write consolidated report + summary CSV
    with open(f"{OUT_DIR}/report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    pd.DataFrame(summary_rows).to_csv(
        f"{OUT_DIR}/walkforward_summary.csv", index=False)
    print(f"\nWrote {OUT_DIR}/report.txt + walkforward_summary.csv")


if __name__ == "__main__":
    main()
