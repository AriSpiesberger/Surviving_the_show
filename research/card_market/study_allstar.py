"""Thesis 5 — the All-Star lottery: buy prospects by P(All-Star), hold.

Probabilities from `light_model_as.py` (walk-forward P(first All-Star within
four seasons)). Each November 2021-2023 buy every priced, not-yet-debuted
prospect's first auto and hold to mid-2026 (median of Jun-Aug 2026 prices).
Net of the same costs as the other studies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import study_basket as sb
from load import HERE, tables
from panel import player_panel, wide
from study_debut import boot


def positions() -> pd.DataFrame:
    pr = pd.read_csv(HERE / "raw" / "light_preds_as.csv")
    m = player_panel()
    px = np.exp(wide(m)).sort_index(axis=1).ffill(axis=1, limit=3)
    end = px[[pd.Period(f"2026-{k:02d}", "M") for k in (6, 7, 8)]].median(axis=1)
    rows = []
    for Y in (2021, 2022, 2023):
        d = pr[pr.S == Y].set_index("player_id")
        d = d.join(pd.DataFrame({"p0": px[pd.Period(f"{Y}-11", "M")], "p1": end}), how="inner").dropna(subset=["p0", "p1"])
        d["AS"] = d.as_first > Y
        d["Y"] = Y
        rows.append(d)
    d = pd.concat(rows)
    d["name"] = d.index.map(tables()["people"].set_index("player_id").name)
    d["mult"] = d.p1 / d.p0
    d["net"] = sb.net(d.p0, d.p1)
    return d


def agg(g: pd.DataFrame) -> pd.Series:
    lo, hi = boot(g.net)
    c = sb.COSTS
    cost = (g.p0 * (1 + c["buy_tax"]) + c["buy_ship"]).sum()
    proceeds = (g.p1 * (1 - c["sell_fee"]) - c["sell_fixed"]).sum()
    return pd.Series({"n": len(g), "all_stars": int(g.AS.sum()), "p0_med": g.p0.median(), "mult_mean": g.mult.mean(),
                      "mult_med": g.mult.median(), "net": g.net.mean(), "net_lo": lo, "net_hi": hi,
                      "win": (g.net > 0).mean(), "net_$wtd": proceeds / cost - 1})


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    d = positions()
    print(len(d), "positions;", int(d.AS.sum()), "later All-Stars\n")
    band = pd.cut(d.p_as4, [0, 0.005, 0.02, 0.05, 0.1, 1], include_lowest=True)
    print(pd.DataFrame({str(k): agg(g) for k, g in d.groupby(band, observed=True)}).T.round(3).to_string(), "\n")
    for Y, g in d.groupby("Y"):
        a = agg(g.nlargest(25, "p_as4"))
        print(f"{Y} top 25 by P(All-Star): {a.all_stars}/25 All-Stars, median ${a.p0_med:.0f}, "
              f"net {a.net:+.2f} [{a.net_lo:+.2f}, {a.net_hi:+.2f}]")
    d["resid"] = np.nan
    for Y, g in d.groupby("Y"):
        lp = np.log(g.p_as4.clip(1e-5))
        b = np.polyfit(lp, np.log(g.p0), 1)
        d.loc[d.Y == Y, "resid"] = np.log(g.p0) - np.polyval(b, lp)
        print(f"{Y}: log price on log P(All-Star) slope {b[0]:.2f}, corr {np.corrcoef(lp, np.log(g.p0))[0, 1]:.2f}")
    u = d[d.p_as4 >= 0.005]
    t = u.groupby("Y").resid.transform(lambda v: pd.qcut(v, 3, labels=["cheap", "mid", "rich"]))
    print("\nP(All-Star) >= 0.5%, by price relative to P(All-Star):")
    print(pd.DataFrame({k: agg(g) for k, g in u.groupby(t, observed=True)}).T.round(3).to_string())
    print("\nlater All-Stars:")
    print(d[d.AS].sort_values(["Y", "p_as4"], ascending=[True, False])[["Y", "name", "p_as4", "p0", "p1", "mult", "as_first"]].round(3).to_string())
