"""Player x month price panel for 1st Bowman Chrome autos, plus a market index.

One series per player: the raw (ungraded) base 1st Chrome auto. A player with
several first autos (Bowman Chrome + Draft Chrome, 1st Edition...) keeps the
card with the longest price history.

The SportsCardsPro monthly price is a last-sale-style estimate, so a thinly
traded card shows a flat line between sales. `stale` marks months whose price
is unchanged from the prior month; studies can drop or keep them.

Market index: equal-weight mean of monthly log price changes across all cards
priced in both months, cumulated. Returns "vs market" subtract its change.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from load import tables


@lru_cache
def player_panel(kind: str = "first_auto", grade: str = "raw") -> pd.DataFrame:
    t = tables()
    cards = t["cards"]
    cards = cards[(cards.kind == kind) & cards.player_id.notna()]
    m = t["monthly"][["id", "month", grade]].dropna().rename(columns={grade: "price"})
    m = m.merge(cards[["id", "player_id", "set", "card_year"]], on="id")
    # one card per player: longest history
    n = m.groupby(["player_id", "id"]).size().rename("n").reset_index()
    keep = n.sort_values("n", ascending=False).drop_duplicates("player_id").id
    m = m[m.id.isin(keep)].sort_values(["player_id", "month"])
    m["logp"] = np.log(m.price)
    g = m.groupby("player_id")
    gap = g.month.diff().map(lambda d: d.n if pd.notna(d) else np.nan)
    m["ret"] = g.logp.diff().where(gap == 1)
    m["stale"] = m.ret == 0
    idx = market_index(m)
    m = m.merge(idx, on="month", how="left")
    m["xret"] = m.ret - m.mkt_ret
    return m.reset_index(drop=True)


def market_index(m: pd.DataFrame) -> pd.DataFrame:
    idx = m.groupby("month").ret.agg(mkt_ret="mean", n_cards="count").reset_index()
    idx["mkt_ret"] = idx.mkt_ret.fillna(0)
    idx["mkt_level"] = idx.mkt_ret.cumsum()
    return idx


def wide(m: pd.DataFrame, col: str = "logp") -> pd.DataFrame:
    return m.pivot(index="player_id", columns="month", values=col)


def fwd_return(m: pd.DataFrame, start: str, end: str, min_price: float = 0.0) -> pd.DataFrame:
    """Buy-and-hold log return from month `start` to month `end` per player, raw
    and vs market. Requires a price in both months."""
    w = wide(m)
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    if s not in w.columns or e not in w.columns:
        return pd.DataFrame(columns=["p0", "p1", "ret", "xret"])
    out = pd.DataFrame({"p0": np.exp(w[s]), "p1": np.exp(w[e])}).dropna()
    out = out[out.p0 >= min_price]
    out["ret"] = np.log(out.p1 / out.p0)
    idx = m.drop_duplicates("month").set_index("month").mkt_level
    out["xret"] = out.ret - (idx[e] - idx[s])
    return out


if __name__ == "__main__":
    m = player_panel()
    print(m.player_id.nunique(), "players;", len(m), "player-months")
    print("stale share:", round(m.stale.mean(), 3))
    idx = m.drop_duplicates("month").set_index("month")[["mkt_ret", "n_cards", "mkt_level"]]
    print(idx.round(3).to_string())
