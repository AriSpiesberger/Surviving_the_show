"""Thesis 3 — do card prices trend or reverse, and do last season's stats
predict next year's card return?

(a) Price momentum. At each formation month t sort cards on their trailing
    6-month log return, skip a month, hold 6 months. Formation dates are six
    months apart so holding windows don't overlap. Returns are net of the
    card-market index. Only cards priced >= MIN_PRICE at t and whose price
    actually moved in the formation window (drops stale flat lines).

(b) Fundamentals. Each November, for players who have not debuted: does
    season-Y performance (league-percentile wOBA for hitters, FIP for
    pitchers), age relative to level, or level itself rank next season's card
    return? Everything on the right-hand side is known at the buy date.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from load import DB, tables
from panel import player_panel, wide

MIN_PRICE = 10.0


def momentum(form: int = 6, hold: int = 6, skip: int = 1) -> pd.DataFrame:
    m = player_panel()
    w = wide(m.assign(x=m.logp - m.mkt_level), "x").sort_index(axis=1)
    px = np.exp(wide(m)).sort_index(axis=1)
    rows = []
    for t in pd.period_range("2022-06", "2025-12", freq="M")[::6]:
        a, b, c = t - form, t + skip, t + skip + hold
        if not {a, t, b, c} <= set(w.columns):
            continue
        d = pd.DataFrame({"form": w[t] - w[a], "fwd": w[c] - w[b], "p": px[t]}).dropna()
        d = d[(d.p >= MIN_PRICE) & (d.form != 0)]
        d["q"] = pd.qcut(d.form, 5, labels=[1, 2, 3, 4, 5])
        d["t"] = str(t)
        rows.append(d)
    return pd.concat(rows)


def fundamentals() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    s = pd.read_sql(
        "select player_id, season_year as year, level, age_during_season as age, pa, ip, "
        "pct_woba, pct_fip, pct_k_pct, pct_k9 from season_stats "
        "where level in ('AAA','AA','A+','A') and season_year between 2022 and 2025 and (pa>=100 or ip>=30)", con)
    s["lv"] = s.level.map({"AAA": 0, "AA": 1, "A+": 2, "A": 3})
    s = s.sort_values("lv").drop_duplicates(["player_id", "year"])
    # percentile where higher = better for both groups
    s["perf"] = np.where(s.pa >= 100, s.pct_woba, 100 - s.pct_fip)
    s["age_vs_level"] = s.age - s.groupby(["year", "level"]).age.transform("median")
    m = player_panel()
    x = wide(m.assign(x=m.logp - m.mkt_level), "x").sort_index(axis=1).ffill(axis=1, limit=3)
    px = np.exp(wide(m)).sort_index(axis=1).ffill(axis=1, limit=3)
    deb = tables()["people"].set_index("player_id").debut_date
    out = []
    for y in (2022, 2023, 2024, 2025):
        buy, end = pd.Period(f"{y}-11", "M"), min(pd.Period(f"{y + 1}-10", "M"), pd.Period("2026-08", "M"))
        prev = pd.Period(f"{y}-05", "M")
        d = s[s.year == y].set_index("player_id")
        d = d.join(pd.DataFrame({"fwd": x[end] - x[buy], "past": x[buy] - x[prev], "p0": px[buy]}), how="inner")
        d = d[d.index.map(deb).isna() | (d.index.map(deb) > buy.to_timestamp())]
        out.append(d.dropna(subset=["fwd", "p0"]))
    return pd.concat(out)


def spearman_by_year(d: pd.DataFrame, col: str) -> str:
    parts = []
    for y, g in d.groupby("year"):
        g = g.dropna(subset=[col, "fwd"])
        r, p = stats.spearmanr(g[col], g.fwd)
        parts.append(f"{y}: {r:+.2f} (p={p:.2f}, n={len(g)})")
    return "  ".join(parts)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    mo = momentum()
    print(f"== price momentum: 6m formation, skip 1, hold 6m, vs market, price >= ${MIN_PRICE:.0f} ==")
    print(mo.groupby("q", observed=True).agg(n=("fwd", "size"), form=("form", "mean"), fwd=("fwd", "mean"),
                                             fwd_med=("fwd", "median"), up=("fwd", lambda v: (v > 0).mean())).round(3))
    spread = mo.groupby(["t", "q"], observed=True).fwd.mean().unstack()
    spread["5-1"] = spread[5] - spread[1]
    print(spread.round(3))
    print("mean 5-1 spread %.3f, t=%.2f over %d dates" % (spread["5-1"].mean(), spread["5-1"].mean() / spread["5-1"].sem(), len(spread)))

    f = fundamentals()
    f = f[f.p0 >= 5]
    print(f"\n== fundamentals -> next-season card return vs market (not-debuted, price >= $5, n={len(f)}) ==")
    for col in ("perf", "age_vs_level", "lv", "past", "p0"):
        print(f"{col:>13}: {spearman_by_year(f, col)}")
    f["perf_q"] = f.groupby("year").perf.transform(lambda v: pd.qcut(v, 4, labels=False, duplicates="drop"))
    f["young"] = f.age_vs_level < -0.75
    print(f.groupby("perf_q").fwd.agg(["size", "mean", "median"]).round(3))
    print(f.groupby(["young", "perf_q"]).fwd.agg(["size", "mean", "median"]).round(3))
