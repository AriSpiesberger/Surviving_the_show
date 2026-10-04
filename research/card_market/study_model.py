"""Thesis 4 — what do the model's probabilities know that card prices don't?

Probabilities come from `light_model.py` (walk-forward, leak-fixed features):
at season-end Y, P(debut in Y+1), P(debut by Y+2), P(new Top-100 in the Y+1
lists). Cards are bought in November Y at the SportsCardsPro monthly price.

(a) Calibration on the card universe and the trade by probability band:
    sell one month after debut, else at the end of season Y+1, net of costs.
(b) How much of each probability is already in the price (rank correlation
    with price), and whether the part that isn't predicts returns:
    regress log price on the probabilities within year, take the residual
    ("cheap for its probabilities"), sort on it.
(c) Top-100 probability vs the November -> February list-season move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import study_basket as sb
from load import HERE
from panel import player_panel, wide
from study_debut import boot


def positions() -> pd.DataFrame:
    pr = pd.read_csv(HERE / "raw" / "light_preds.csv")
    m = player_panel()
    px = np.exp(wide(m)).sort_index(axis=1).ffill(axis=1, limit=3)
    lev = m.drop_duplicates("month").set_index("month").mkt_level
    out = []
    for y in (2022, 2023, 2024, 2025):
        b = sb.basket(y)  # Nov buy, not debuted, exit rule, costs
        d = pr[pr.S == y].set_index("player_id")
        b = b.join(d[["p_deb1", "p_deb2", "p_top100", "elig_top100", "start"]], how="inner")
        nov, feb, end = pd.Period(f"{y}-11", "M"), pd.Period(f"{y + 1}-02", "M"), min(pd.Period(f"{y + 1}-10", "M"), sb.LAST_MONTH)
        b["x_feb"] = np.log(px[feb].reindex(b.index) / b.p0) - (lev[feb] - lev[nov])
        b["x_hold"] = np.log(b.p_hold / b.p0) - (lev[end] - lev[nov])
        out.append(b)
    return pd.concat(out)


def band_table(b: pd.DataFrame, col: str, edges, min_price: float = 0.0) -> pd.DataFrame:
    b = b[b.p0 >= min_price]
    g = b.groupby(pd.cut(b[col], edges, include_lowest=True), observed=True)

    def agg(d):
        lo, hi = boot(d.net)
        cost = (d.p0 * (1 + sb.COSTS["buy_tax"]) + sb.COSTS["buy_ship"]).sum()
        proceeds = (d.p_exit * (1 - sb.COSTS["sell_fee"]) - sb.COSTS["sell_fixed"]).sum()
        return pd.Series({"n": len(d), "p_mean": d[col].mean(), "debut": d.debuted.mean(), "p0_med": d.p0.median(),
                          "gross": d.gross.mean(), "net": d.net.mean(), "net_lo": lo, "net_hi": hi,
                          "net_med": d.net.median(), "win": (d.net > 0).mean(), "net_$wtd": proceeds / cost - 1})
    return pd.DataFrame({str(k): agg(d) for k, d in g}).T


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    b = positions()
    print(len(b), "positions with a model probability and a priced first auto\n")

    edges = [0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0]
    print("== (a) by P(debut next season); buy Nov, sell debut+1mo else season end ==")
    print(band_table(b, "p_deb1", edges).round(3).to_string())
    print("\n-- same, price >= $10 --")
    print(band_table(b, "p_deb1", edges, 10).round(3).to_string())
    print("\n-- by year, P >= 0.5 --")
    hi = b[b.p_deb1 >= 0.5]
    print(hi.groupby("year").agg(n=("net", "size"), debut=("debuted", "mean"), gross=("gross", "mean"),
                                 net=("net", "mean"), win=("net", lambda v: (v > 0).mean())).round(3))

    print("\n== (b) is the probability already in the price? spearman with Nov price, by year ==")
    for c in ("p_deb1", "p_deb2", "p_top100"):
        print(f"{c:>9}:", "  ".join(f"{y}: {stats.spearmanr(g[c], g.p0)[0]:+.2f}" for y, g in b.groupby("year")))
    b["resid"] = np.nan
    for y, g in b.groupby("year"):
        X = np.column_stack([np.ones(len(g)), logit(g.p_deb1), logit(g.p_deb2), logit(g.p_top100)])
        beta, *_ = np.linalg.lstsq(X, np.log(g.p0), rcond=None)
        b.loc[g.index[g.year == y] if False else (b.year == y), "resid"] = np.log(g.p0).to_numpy() - X @ beta
        r2 = 1 - ((np.log(g.p0) - X @ beta) ** 2).sum() / ((np.log(g.p0) - np.log(g.p0).mean()) ** 2).sum()
        print(f"  {y}: log price ~ logit probabilities, R2 = {r2:.2f} (n={len(g)})")
    b["cheap_q"] = b.groupby("year").resid.transform(lambda v: pd.qcut(v, 4, labels=["cheapest", "q2", "q3", "richest"]))
    print("\nforward return vs market by 'cheap for its probabilities' quartile (price >= $5):")
    u = b[b.p0 >= 5]
    print(u.groupby("cheap_q", observed=True).agg(n=("x_hold", "size"), p0=("p0", "median"), p_deb1=("p_deb1", "mean"),
                                                  debut=("debuted", "mean"), x_hold=("x_hold", "mean"),
                                                  net=("net", "mean"), win=("net", lambda v: (v > 0).mean())).round(3))
    print("spearman resid vs x_hold by year:",
          "  ".join(f"{y}: {stats.spearmanr(g.resid, g.x_hold, nan_policy='omit')[0]:+.2f}" for y, g in u.groupby("year")))
    hp = u[u.p_deb1 >= 0.3]
    print("\nwithin P(debut) >= 0.3: cheap half vs rich half")
    hp = hp.assign(half=np.where(hp.resid < hp.groupby("year").resid.transform("median"), "cheap", "rich"))
    print(hp.groupby("half").agg(n=("net", "size"), p0=("p0", "median"), debut=("debuted", "mean"), gross=("gross", "mean"),
                                 net=("net", "mean"), win=("net", lambda v: (v > 0).mean())).round(3))

    print("\n== (c) P(new Top-100) at season end vs Nov->Feb move (vs market), not already Top-100 ==")
    t = b[b.elig_top100 & (b.p0 >= 5)]
    t = t.assign(band=pd.cut(t.p_top100, [0, 0.02, 0.1, 0.3, 1.0], include_lowest=True))
    print(t.groupby("band", observed=True).agg(n=("x_feb", "size"), p0=("p0", "median"), x_feb=("x_feb", "mean"),
                                               x_feb_med=("x_feb", "median"), x_hold=("x_hold", "mean")).round(3))
    b.to_csv(HERE / "raw" / "model_positions.csv")
