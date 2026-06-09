"""Full data audit — surface WEIRD missingness (cohort/era/level-correlated),
the kind that silently biases training. Read-only.

    python -m scripts.local.data_audit
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd

DB = "prospects_snapshot.db"


def nn(s: pd.Series) -> float:
    """Non-null AND non-empty-string rate."""
    if s.dtype == object:
        return s.replace("", np.nan).notna().mean()
    return s.notna().mean()


def main():
    c = sqlite3.connect(DB)
    pro = pd.read_sql("SELECT * FROM prospects", c)
    ss = pd.read_sql("SELECT * FROM season_stats", c)
    co = pd.read_sql("SELECT * FROM career_outcomes", c)
    c.close()

    # ---- 1. prospects: overall missingness ----
    print(f"{'='*70}\nPROSPECTS (n={len(pro):,}) — columns under 90% populated\n{'='*70}")
    rows = sorted(((col, nn(pro[col])) for col in pro.columns), key=lambda x: x[1])
    for col, p in rows:
        if p < 0.90:
            print(f"  {col:<26}{p:6.1%}")

    # ---- 2. prospects: drafted vs IFA divergence ----
    print(f"\n--- prospects: drafted vs IFA (|gap|>25%) ---")
    d, i = pro[pro.is_international == 0], pro[pro.is_international == 1]
    for col in pro.columns:
        if col in ("player_id", "is_international"):
            continue
        gd, gi = nn(d[col]), nn(i[col])
        if abs(gd - gi) > 0.25:
            print(f"  {col:<26} drafted={gd:5.0%}  IFA={gi:5.0%}")

    # ---- 3. prospects: recent vs old draftees (era gaps) ----
    print(f"\n--- prospects (drafted): era gaps (pre-2015 vs 2021+, |gap|>25%) ---")
    old = pro[(pro.draft_year.notna()) & (pro.draft_year < 2015)]
    new = pro[(pro.draft_year.notna()) & (pro.draft_year >= 2021)]
    for col in pro.columns:
        if col in ("player_id",):
            continue
        go, gnv = nn(old[col]), nn(new[col])
        if abs(go - gnv) > 0.25:
            print(f"  {col:<26} pre2015={go:5.0%}  2021+={gnv:5.0%}")

    # ---- 4. season_stats: stat coverage by level ----
    ss["lvl"] = ss["level"].fillna("?").str.upper().str.replace("A-", " A-", regex=False)
    print(f"\n{'='*70}\nSEASON_STATS (n={len(ss):,}) — key stat coverage by level\n{'='*70}")
    levels = ["RK", "A", "A+", "AA", "AAA", "MLB"]
    stat_cols = [c for c in ["pa", "avg", "obp", "slg", "woba", "iso", "k_pct",
                             "bb_pct", "babip", "ip", "era", "fip", "whip", "k9",
                             "bb9", "velo_avg"] if c in ss.columns]
    sub = ss[ss.level.str.upper().isin(levels)] if "level" in ss else ss
    cov = sub.assign(L=sub.level.str.upper()).groupby("L")[stat_cols].apply(
        lambda g: g.apply(nn))
    print(cov.reindex([l for l in levels if l in cov.index]).round(2).to_string())

    # ---- 5. season_stats: coverage by era (find stats that only exist recently) ----
    print(f"\n--- season_stats: stat coverage by era (pre-2015 vs 2020+) ---")
    so = ss[ss.season_year < 2015]
    sn = ss[ss.season_year >= 2020]
    for col in stat_cols:
        a, b = nn(so[col]), nn(sn[col])
        if abs(a - b) > 0.20:
            print(f"  {col:<12} pre2015={a:5.0%}  2020+={b:5.0%}")

    # ---- 6. career_outcomes ----
    print(f"\n{'='*70}\nCAREER_OUTCOMES (n={len(co):,}) — columns under 95%\n{'='*70}")
    for col in co.columns:
        p = nn(co[col])
        if p < 0.95:
            print(f"  {col:<26}{p:6.1%}")

    # ---- 7. sentinel scan (0-as-missing, common in stats) ----
    print(f"\n--- season_stats: suspicious exact-zero rates (possible 0=missing) ---")
    for col in ["velo_avg", "babip", "woba", "fip"]:
        if col in ss.columns:
            s = pd.to_numeric(ss[col], errors="coerce")
            z = (s == 0).mean()
            if z > 0.05:
                print(f"  {col:<12} exact-zero={z:.0%}  (vs null={s.isna().mean():.0%})")


if __name__ == "__main__":
    main()
