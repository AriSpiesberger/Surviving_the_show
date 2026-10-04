"""Thesis 1 — what a 1st Bowman Chrome auto does around the MLB debut.

Event time k = calendar month minus debut month. For each debuting player we
track the cumulative log price change of his raw base 1st Chrome auto, raw and
net of the equal-weight card-market index, and average across players.

Outputs the event-time path and the holding-window returns a trader cares
about (buy k months before debut, sell at / after debut), with player-level
bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from load import tables
from panel import player_panel

RNG = np.random.default_rng(7)


def event_panel(lo: int = -12, hi: int = 18) -> pd.DataFrame:
    m = player_panel()
    p = tables()["people"][["player_id", "debut_date", "is_pitcher", "year_top_100", "draft_round"]]
    m = m.merge(p[p.debut_date.notna()], on="player_id")
    m["k"] = (m.month - m.debut_date.dt.to_period("M")).map(lambda d: d.n)
    return m[m.k.between(lo, hi)]


def window(ev: pd.DataFrame, k0: int, k1: int, col: str = "logp") -> pd.Series:
    """Per-player log return from event month k0 to k1 (needs a price at both)."""
    w = ev.pivot(index="player_id", columns="k", values=col)
    if k0 not in w or k1 not in w:
        return pd.Series(dtype=float)
    return (w[k1] - w[k0]).dropna()


def xwindow(ev: pd.DataFrame, k0: int, k1: int) -> pd.Series:
    """Same, net of the market index over the same calendar months."""
    ev = ev.assign(x=ev.logp - ev.mkt_level)
    return window(ev, k0, k1, "x")


def boot(x: pd.Series, n: int = 4000) -> tuple[float, float]:
    if len(x) < 5:
        return (np.nan, np.nan)
    v = x.to_numpy()
    means = v[RNG.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(x: pd.Series) -> dict:
    lo, hi = boot(x)
    return {"n": len(x), "mean": x.mean(), "ci_lo": lo, "ci_hi": hi, "median": x.median(),
            "pct_up": (x > 0).mean(), "mean_pct": np.expm1(x.mean())}


def windows_table(ev: pd.DataFrame, pairs, fn=xwindow) -> pd.DataFrame:
    return pd.DataFrame({f"{a:+d}→{b:+d}": summarize(fn(ev, a, b)) for a, b in pairs}).T


PAIRS = [(-12, -1), (-6, -1), (-3, -1), (-1, 0), (-1, 1), (0, 1), (0, 3), (0, 6), (0, 12), (1, 12), (-6, 6), (-12, 12)]

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    ev = event_panel()
    ev = ev[ev.debut_date >= "2022-01-01"]
    print(ev.player_id.nunique(), "debuting players with a priced 1st auto\n")
    print("== vs market (log return, bootstrap 95% CI) ==")
    print(windows_table(ev, PAIRS).round(3).to_string())
    print("\n== raw ==")
    print(windows_table(ev, PAIRS, window).round(3).to_string())
    for name, mask in [("hitters", ev.is_pitcher == 0), ("pitchers", ev.is_pitcher == 1),
                       ("ever top-100", ev.year_top_100.notna()), ("never top-100", ev.year_top_100.isna())]:
        print(f"\n== {name}, vs market ==")
        print(windows_table(ev[mask], [(-6, -1), (-1, 1), (0, 6), (0, 12)]).round(3).to_string())
