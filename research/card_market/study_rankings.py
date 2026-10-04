"""Thesis 2 — Top-100 list changes as a card-price event.

Preseason Top-100 lists (Baseball America, MLB Pipeline) are stamped Jan 1 of
the list year and published late January. For each list year Y a player is
    new      on the Top 100 in Y, not in Y-1
    stayed   on it both years
    dropped  on it in Y-1, not in Y (and had not debuted before Y)
We track the raw 1st Chrome auto from the summer before (run-up) through the
year after publication, net of the card-market index.

If the list were news, `new` cards would jump Dec -> Feb. If the market has
already traded the breakout season, the move sits in the run-up instead and
nothing is left after publication.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from load import DB, tables
from panel import player_panel, wide
from study_debut import summarize

SOURCES = ("Baseball America", "MLB Pipeline")


def events() -> pd.DataFrame:
    r = pd.read_sql("select player_id, as_of, source, overall_rank from rankings_history "
                    "where overall_rank between 1 and 100", sqlite3.connect(DB))
    r = r[r.source.isin(SOURCES)]
    r["year"] = pd.to_datetime(r.as_of).dt.year
    on = r.groupby(["player_id", "year"]).overall_rank.min().reset_index()
    cur = on.assign(key=1)
    prev = on.assign(year=on.year + 1).rename(columns={"overall_rank": "prev_rank"})
    e = cur.merge(prev, on=["player_id", "year"], how="outer")
    e["status"] = np.select([e.prev_rank.isna(), e.overall_rank.isna()], ["new", "dropped"], "stayed")
    deb = tables()["people"].set_index("player_id").debut_date
    e["debut"] = e.player_id.map(deb)
    # a "dropped" player who already graduated to MLB isn't a demotion
    e = e[~((e.status == "dropped") & (e.debut < pd.to_datetime(e.year.astype(str) + "-01-01")))]
    return e[e.year.between(2022, 2026)]


def paths(e: pd.DataFrame) -> pd.DataFrame:
    m = player_panel()
    m = m.assign(x=m.logp - m.mkt_level)
    wx, wr = wide(m, "x").ffill(axis=1, limit=2), wide(m, "logp").ffill(axis=1, limit=2)
    pts = {"jul_prev": (-1, 7), "dec_prev": (-1, 12), "feb": (0, 2), "apr": (0, 4), "jul": (0, 7), "dec": (0, 12)}
    rows = []
    for pid, y, st, rk in zip(e.player_id, e.year, e.status, e.overall_rank):
        if pid not in wx.index:
            continue
        row = {"player_id": pid, "year": y, "status": st, "rank": rk}
        for name, (dy, mo) in pts.items():
            p = pd.Period(f"{y + dy}-{mo:02d}", "M")
            row["x_" + name] = wx.at[pid, p] if p in wx.columns else np.nan
            row["r_" + name] = wr.at[pid, p] if p in wr.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


WINDOWS = [("jul_prev", "dec_prev", "run-up Jul→Dec before"), ("dec_prev", "feb", "publication Dec→Feb"),
           ("feb", "jul", "after Feb→Jul"), ("feb", "dec", "after Feb→Dec"), ("dec_prev", "dec", "full year Dec→Dec")]

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    p = paths(events())
    print(p.groupby(["status", "year"]).size().unstack(fill_value=0), "\n")
    for pre, lab in (("x_", "vs market"), ("r_", "raw")):
        print(f"== {lab} (log return) ==")
        for st in ("new", "stayed", "dropped"):
            g = p[p.status == st]
            t = pd.DataFrame({name: summarize((g[pre + b] - g[pre + a]).dropna()) for a, b, name in WINDOWS}).T
            print(f"-- {st} --\n{t.round(3).to_string()}")
        print()
    g = p[p.status == "new"].assign(price_dec=lambda d: np.exp(d.r_dec_prev))
    g["tier"] = pd.cut(g["rank"], [0, 25, 50, 100], labels=["1-25", "26-50", "51-100"])
    print("new entrants by rank tier, vs market, Dec→Feb and Feb→Dec:")
    print(g.groupby("tier", observed=True).apply(lambda d: pd.Series({
        "n": len(d), "pub": (d.x_feb - d.x_dec_prev).mean(), "after": (d.x_dec - d.x_feb).mean(),
        "med_price": d.price_dec.median()}), include_groups=False).round(3))
