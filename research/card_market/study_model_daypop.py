"""Thesis 3c — §3b's model rules with a days 0-3 exit instead of the month after debut.

Same positions as study_model.py (buy the raw first auto in November Y for every not-yet-debuted
player with a walk-forward probability; light_model.py, trained from 1990 on the 1988-2004 DB).
Only the exit of players who debut changes:
  monthly  (study_model)  the monthly price one month after the debut
  day 0-3  (this study)   the monthly price the month BEFORE the debut x the measured days 0-3
                          pop for the card's price band (study_callup.pop_by_band, from sales)
Non-debuters exit at the end of season Y+1 in both. Net of the same costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import study_basket as sb
from panel import player_panel, wide
from study_callup import BAND_LBL, BANDS, pop_by_band
from study_debut import boot
from study_model import logit, positions


def rule_table(b: pd.DataFrame) -> pd.DataFrame:
    rules = {"all": np.ones(len(b), bool), "P>=0.3": b.p_deb1 >= 0.3, "P>=0.5": b.p_deb1 >= 0.5,
             "P>=0.5 cheap": (b.p_deb1 >= 0.5) & b.cheap, "P>=0.5 rich": (b.p_deb1 >= 0.5) & ~b.cheap,
             "P>=0.7": b.p_deb1 >= 0.7, "P>=0.7 cheap": (b.p_deb1 >= 0.7) & b.cheap,
             "P>=0.7 rich": (b.p_deb1 >= 0.7) & ~b.cheap}
    out = {}
    for k, m in rules.items():
        d = b[m]
        row = {"n": len(d), "debut": d.debuted.mean(), "med_p0": d.p0.median()}
        for col in ("net", "net_d03"):
            lo, hi = boot(d[col]) if len(d) >= 5 else (np.nan, np.nan)
            row.update({col: d[col].mean(), f"{col}_lo": lo, f"{col}_hi": hi})
        cost = (d.p0 * (1 + sb.COSTS["buy_tax"]) + sb.COSTS["buy_ship"]).sum()
        row["d03_$wtd"] = (d.p_d03 * (1 - sb.COSTS["sell_fee"]) - sb.COSTS["sell_fixed"]).sum() / cost - 1
        row["d03_median"] = d.net_d03.median()
        out[k] = row
    return pd.DataFrame(out).T


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pop = pop_by_band()
    b = positions().rename_axis("player_id").reset_index()   # players repeat across years
    px = np.exp(wide(player_panel())).sort_index(axis=1).ffill(axis=1, limit=3)
    dm = b.debut_date.dt.to_period("M")
    pre = []
    for pid, m, deb in zip(b.player_id, dm, b.debuted):
        mm = m - 1 if deb else None
        pre.append(px.at[pid, mm] if deb and mm in px.columns else np.nan)
    b["p_pre"] = pre
    band = pd.cut(b.p_pre, BANDS, labels=BAND_LBL, right=False)
    b["p_d03"] = np.where(b.debuted, b.p_pre * band.map(pop).astype(float), b.p_exit)
    b = b.dropna(subset=["p_d03"])
    b["net_d03"] = sb.net(b.p0, b.p_d03)
    # "cheap for its probabilities": residual of the SEPTEMBER log price on the probabilities, within year
    b["resid"] = np.nan
    for y in b.year.unique():
        g = b[b.year == y]
        sep = pd.Series(px[pd.Period(f"{y}-09", "M")].reindex(g.player_id).to_numpy(), index=g.index)
        ok = sep.notna() & (sep > 0)
        X = np.column_stack([np.ones(ok.sum()), logit(g.p_deb1[ok]), logit(g.p_deb2[ok]), logit(g.p_top100[ok])])
        beta, *_ = np.linalg.lstsq(X, np.log(sep[ok]), rcond=None)
        b.loc[g.index[ok.to_numpy()], "resid"] = np.log(sep[ok].to_numpy()) - X @ beta
    b = b.dropna(subset=["resid"])
    b["cheap"] = b.resid < b.groupby("year").resid.transform("median")
    print(len(b), "positions (Nov 2022-2025 buys, priced first auto, walk-forward probabilities)\n")
    print("== rules: monthly exit (net) vs days 0-3 exit (net_d03) ==")
    print(rule_table(b).round(3).to_string())
    print("\n-- price >= $10 --")
    print(rule_table(b[b.p0 >= 10]).round(3).to_string())
    print("\n-- by year, P >= 0.5 cheap half (day 0-3 exit) --")
    h = b[(b.p_deb1 >= 0.5) & b.cheap]
    print(h.groupby("year").agg(n=("net_d03", "size"), debut=("debuted", "mean"), net_d03=("net_d03", "mean"),
                                median=("net_d03", "median")).round(3).to_string())
    top = h.assign(gain=(h.p_d03 * (1 - sb.COSTS["sell_fee"]) - sb.COSTS["sell_fixed"])
                   - (h.p0 * (1 + sb.COSTS["buy_tax"]) + sb.COSTS["buy_ship"]))
    print("\nshare of the P>=0.5 cheap-half dollar profit from the top 5 positions:",
          round(top.gain.nlargest(5).sum() / top.gain.sum(), 2) if top.gain.sum() > 0 else "n/a (no profit)")
