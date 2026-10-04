"""Thesis 1b — a no-hindsight version of "buy before debut, sell at debut".

Each November (season Y just ended) buy the raw base 1st Bowman Chrome auto of
every player who has not debuted and whose top level in season Y is known.
Exit one month after his MLB debut if he debuts during season Y+1, otherwise at
the end of season Y+1. Nothing here uses the model: it measures what the
structural trade earns by level and price, so the model's job (picking within
the basket) can be judged against it.

Costs (edit COSTS): eBay final value fee on the sale, a fixed per-order fee,
inbound shipping and sales tax on the purchase. Buyer pays outbound shipping.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from load import DB, tables
from panel import player_panel, wide
from study_debut import boot

COSTS = dict(sell_fee=0.1325, sell_fixed=0.40, buy_ship=1.50, buy_tax=0.07)
LAST_MONTH = pd.Period("2026-08", "M")  # last month with full price coverage
LEVELS = ["AAA", "AA", "A+", "A"]


def top_level(year: int) -> pd.Series:
    """Highest level with real playing time in `year` (50 PA or 15 IP)."""
    s = pd.read_sql(
        "select player_id, level, pa, ip from season_stats where season_year=? "
        "and level in ('AAA','AA','A+','A') and (pa>=50 or ip>=15)",
        sqlite3.connect(DB), params=(year,))
    s["rank"] = s.level.map({l: i for i, l in enumerate(LEVELS)})
    return s.sort_values("rank").drop_duplicates("player_id").set_index("player_id").level


def net(p0: pd.Series, p1: pd.Series) -> pd.Series:
    c = COSTS
    cost = p0 * (1 + c["buy_tax"]) + c["buy_ship"]
    proceeds = p1 * (1 - c["sell_fee"]) - c["sell_fixed"]
    return proceeds / cost - 1


def basket(year: int) -> pd.DataFrame:
    m = player_panel()
    px = np.exp(wide(m)).sort_index(axis=1).ffill(axis=1, limit=3)
    buy = pd.Period(f"{year}-11", "M")
    end = min(pd.Period(f"{year + 1}-10", "M"), LAST_MONTH)
    ppl = tables()["people"].set_index("player_id")
    lv = top_level(year)
    b = pd.DataFrame({"p0": px[buy]}).dropna().join(lv, how="inner")
    b = b.join(ppl[["debut_date", "is_pitcher", "year_top_100", "name"]])
    b = b[b.debut_date.isna() | (b.debut_date > buy.to_timestamp())]
    dm = b.debut_date.dt.to_period("M")
    b["debuted"] = dm.notna() & (dm <= end)
    exit_m = (dm + 1).where(b.debuted, end).map(lambda p: min(p, LAST_MONTH))
    b["p_exit"] = [px.at[pid, em] if em in px.columns else np.nan for pid, em in exit_m.items()]
    b["p_hold"] = px[end].reindex(b.index)  # same basket, no sell-at-debut rule
    b = b.dropna(subset=["p_exit", "p_hold"])
    b["gross"] = b.p_exit / b.p0 - 1
    b["net"] = net(b.p0, b.p_exit)
    b["net_hold"] = net(b.p0, b.p_hold)
    b["year"] = year
    b["bucket"] = pd.cut(b.p0, [0, 5, 10, 25, 60, 1e6], labels=["<$5", "$5-10", "$10-25", "$25-60", "$60+"])
    return b


def report(b: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    c = COSTS

    def agg(g):
        cost = (g.p0 * (1 + c["buy_tax"]) + c["buy_ship"]).sum()
        proceeds = (g.p_exit * (1 - c["sell_fee"]) - c["sell_fixed"]).sum()
        lo, hi = boot(g.net)
        return pd.Series({"n": len(g), "debut_rate": g.debuted.mean(), "med_p0": g.p0.median(),
                          "gross_mean": g.gross.mean(), "net_mean": g.net.mean(), "net_lo": lo, "net_hi": hi,
                          "net_median": g.net.median(), "win_rate": (g.net > 0).mean(),
                          "net_$wtd": proceeds / cost - 1, "net_hold_mean": g.net_hold.mean()})
    return pd.DataFrame({k: agg(g) for k, g in b.groupby(by if len(by) > 1 else by[0], observed=True)}).T


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    b = pd.concat([basket(y) for y in (2022, 2023, 2024, 2025)])
    print(len(b), "positions\n")
    for by in (["year"], ["level"], ["bucket"], ["level", "bucket"], ["debuted"]):
        print(report(b, by).round(3).to_string(), "\n")
    b.to_csv("raw/basket_positions.csv")
