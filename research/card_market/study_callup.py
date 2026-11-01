"""Thesis 7 — trade the short-horizon call-up model (callup.py) with a debut-pop exit.

Signal: walk-forward P(debut in the next 3-30 days) for every not-yet-debuted AA/AAA player,
every Monday April-September 2022-2025 (raw/callup_preds.csv). Universe: players with a priced
raw first Bowman Chrome auto. One position per player per season: the first week his
probability clears the threshold.

Entry: the card's monthly raw price in the signal month (forward-filled up to 3 months).
Exit:
  * debut within the hold window -> entry price x the measured days 0-3 pop for the card's
    price band (study_debut_daily's sales, each card vs its own median sale 15-90 days before
    debut; the price band is the entry price)
  * otherwise -> the monthly price at the end of the hold window
Net of costs (study_basket.COSTS). The pop multiplier is an average over recent debuts applied to
every hit, so treat hit-level returns as expected values, not realised sales.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import study_basket as sb
from load import DB, HERE, tables
from panel import player_panel, wide
from study_debut import boot
from study_debut_daily import sales_rel

BANDS = [0, 10, 15, 30, 1e9]
BAND_LBL = ["<$10", "$10-15", "$15-30", "$30+"]


def pop_by_band() -> pd.Series:
    s = sales_rel()
    s = s[s.day.between(0, 3)]
    card = s.groupby("id").agg(rel=("rel", "median"), base=("base", "first"))
    card["band"] = pd.cut(card.base, BANDS, labels=BAND_LBL, right=False)
    g = card.groupby("band", observed=True).rel
    out = pd.DataFrame({"cards": g.size(), "mult": np.exp(g.mean())})
    print("days 0-3 pop by baseline price band (from individual sales):")
    print(out.round(2).to_string(), "\n")
    return out["mult"]


def positions(window: int, thr: float, pop: pd.Series) -> pd.DataFrame:
    pr = pd.read_csv(HERE / "raw" / "callup_preds.csv", parse_dates=["t", "debut"])
    pr = pr[pr.season >= 2022]
    xm = pd.read_sql("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL", sqlite3.connect(DB))
    xm["mlbam"] = pd.to_numeric(xm.mlbam_id, errors="coerce")
    pr = pr.merge(xm[["player_id", "mlbam"]].drop_duplicates("mlbam"), on="mlbam")
    px = np.exp(wide(player_panel())).sort_index(axis=1).ffill(axis=1, limit=3)
    sig = pr[pr.p_call30 >= thr].sort_values("t").drop_duplicates(["player_id", "season"])
    sig = sig[sig.player_id.isin(px.index)]
    rows = []
    for r in sig.itertuples():
        m0 = pd.Period(r.t, "M")
        if m0 not in px.columns:
            continue
        p0 = px.at[r.player_id, m0]
        if not np.isfinite(p0):
            continue
        end = r.t + pd.Timedelta(days=window)
        hit = pd.notna(r.debut) and (r.t + pd.Timedelta(days=2) < r.debut <= end)
        band = pd.cut([p0], BANDS, labels=BAND_LBL, right=False)[0]
        if hit:
            p1 = p0 * pop.get(band, np.nan)
        else:
            m1 = min(pd.Period(end, "M"), sb.LAST_MONTH)
            p1 = px.at[r.player_id, m1] if m1 in px.columns else np.nan
        rows.append((r.player_id, r.season, r.t, r.p_call30, p0, band, hit, p1))
    d = pd.DataFrame(rows, columns=["player_id", "season", "t", "p", "p0", "band", "hit", "p1"]).dropna(subset=["p1"])
    d["net"] = sb.net(d.p0, d.p1)
    return d


def summarize(d: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    def one(x):
        lo, hi = boot(x.net) if len(x) >= 5 else (np.nan, np.nan)
        return pd.Series({"n": len(x), "hit": x.hit.mean(), "med_price": x.p0.median(), "net": x.net.mean(),
                          "lo": lo, "hi": hi, "win": (x.net > 0).mean()})
    return d.groupby(by, observed=True).apply(one) if by else one(d).to_frame().T


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pop = pop_by_band()
    for window in (30, 60):
        print(f"===== hold {window} days =====")
        for thr in (0.05, 0.10, 0.20, 0.30):
            d = positions(window, thr, pop)
            s = summarize(d)
            print(f"P >= {thr:.2f}: " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.iloc[0].items()))
            if thr in (0.10, 0.20):
                print(summarize(d, "band").round(3).to_string())
                print(summarize(d, "season").round(3).to_string(), "\n")
