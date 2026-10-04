"""Thesis 1c — day-level timing of the debut pop, from individual sales.

SportsCardsPro keeps the most recent ~30 raw sales per card, so this only
covers recent debuts (those whose sale history still reaches back before the
call-up). Each raw sale of a player's first auto is expressed relative to his
own baseline: the median raw sale in the 15-90 days before debut. Sales are
then pooled by days-from-debut.

The debut date is the first MLB game; the call-up is usually announced a day
or two earlier, so day -2..-1 already carries news.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from load import tables

BINS = [-90, -30, -14, -7, -3, 0, 2, 4, 8, 15, 31, 61, 121]
LABELS = ["-90..-31", "-30..-15", "-14..-8", "-7..-4", "-3..-1", "0..1", "2..3", "4..7", "8..14", "15..30", "31..60",
          "61..120"]


def sales_rel() -> pd.DataFrame:
    t = tables()
    c = t["cards"]
    c = c[(c.kind == "first_auto") & c.player_id.notna()]
    s = t["sales"]
    s = s[s.grade == "raw"].merge(c[["id", "player_id"]], on="id")
    s = s.merge(t["people"][["player_id", "debut_date", "is_pitcher", "year_top_100"]], on="player_id")
    s = s[s.debut_date.notna()]
    s["day"] = (s.date - s.debut_date).dt.days
    s = s[s.day.between(-90, 120)]
    base = s[s.day.between(-90, -15)].groupby("id").price.agg(base="median", n_base="size")
    s = s.merge(base, on="id")
    s = s[s.n_base >= 3]
    s["rel"] = np.log(s.price / s.base)
    s["bin"] = pd.cut(s.day, BINS, right=False, labels=LABELS)
    return s


def by_bin(s: pd.DataFrame) -> pd.DataFrame:
    # average within card first so one heavily traded card doesn't dominate
    card = s.groupby(["id", "bin"], observed=True).rel.median().reset_index()
    g = card.groupby("bin", observed=True).rel
    out = g.agg(cards="size", mean="mean", median="median")
    out["se"] = g.sem()
    out["sales"] = s.groupby("bin", observed=True).size()
    out["mean_pct"] = np.expm1(out["mean"])
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    s = sales_rel()
    print(s.id.nunique(), "cards,", s.player_id.nunique(), "players,", len(s), "raw sales; debuts",
          s.debut_date.min().date(), "to", s.debut_date.max().date(), "\n")
    print(by_bin(s).round(3).to_string())
    for name, mask in (("base price < $15", s.base < 15), ("base price >= $15", s.base >= 15)):
        print(f"\n== {name} ==")
        print(by_bin(s[mask])[["cards", "sales", "mean", "se", "median"]].round(3).to_string())
    # sales volume per day: when is the market deepest?
    vol = s.groupby("bin", observed=True).size() / pd.Series(np.diff(BINS), index=LABELS)
    print("\nraw sales per day (all cards pooled):")
    print(vol.round(1).to_string())
