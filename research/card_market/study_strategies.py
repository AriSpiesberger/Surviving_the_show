"""Strategy search around the §3c debut rule (2026-10-05).

One engine for "buy in month M, hold until an event or a deadline":
  universe   not-yet-debuted players with a walk-forward season-end probability for season Y
             (light_model.py; raw/light_preds.csv), a priced first auto in the entry month, and
             no debut before the entry month
  entry      the monthly price (raw or PSA 10) in the entry month (Y-11 .. Y+1-06)
  exit       debut by the deadline -> price the month before debut x the measured days 0-3 pop
             for that grade and price band (individual sales vs each card's pre-debut baseline);
             otherwise the monthly price at the deadline (October of Y+1)
  costs      study_basket.COSTS (13.25% + $0.40 on the sale; 7% tax + $1.50 on the buy)

Every variant reports n, debut rate, median entry price, mean net with a bootstrap CI, median,
dollar-weighted net and the per-year means; nothing is tuned on the years it is scored on
except the threshold grid, which is printed in full.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd

import study_basket as sb
from load import DB, HERE, tables
from panel import player_panel, wide
from study_debut import boot

BANDS = [0, 10, 15, 30, 1e9]
BAND_LBL = ["<$10", "$10-15", "$15-30", "$30+"]
YEARS = (2022, 2023, 2024, 2025)


@lru_cache
def pop(grade: str = "raw") -> pd.Series:
    """days 0-3 sale price vs the card's own median sale 15-90 days before debut, by band"""
    t = tables()
    c = t["cards"]
    c = c[(c.kind == "first_auto") & c.player_id.notna()]
    s = t["sales"]
    s = s[s.grade == grade].merge(c[["id", "player_id"]], on="id")
    s = s.merge(t["people"][["player_id", "debut_date"]], on="player_id")
    s = s[s.debut_date.notna()]
    s["day"] = (s.date - s.debut_date).dt.days
    base = s[s.day.between(-90, -15)].groupby("id").price.agg(base="median", n="size")
    s = s.merge(base[base.n >= 3], on="id")
    w = s[s.day.between(0, 3)]
    card = w.groupby("id").agg(rel=("price", lambda v: 0), base=("base", "first"))
    card["rel"] = w.groupby("id").apply(lambda g: np.log(g.price / g.base).median())
    card["band"] = pd.cut(card.base, BANDS, labels=BAND_LBL, right=False)
    g = card.groupby("band", observed=True).rel
    out = pd.DataFrame({"cards": g.size(), "mult": np.exp(g.mean())})
    # a band needs >= 5 cards for its own multiplier; otherwise the pooled MEDIAN card (robust to the
    # single-card outliers a thin grade produces)
    pooled = np.exp(card.rel.median())
    m = out["mult"].where(out["cards"] >= 5).reindex(BAND_LBL).fillna(pooled)
    out["used"] = m.reindex(out.index)
    return m, out


@lru_cache
def prices(grade: str = "raw") -> pd.DataFrame:
    return np.exp(wide(player_panel(grade=grade))).sort_index(axis=1).ffill(axis=1, limit=3)


@lru_cache
def preds() -> pd.DataFrame:
    p = pd.read_csv(HERE / "raw" / "light_preds.csv")
    return p[p.S.isin(YEARS)]


@lru_cache
def forty_man_adds() -> pd.DataFrame:
    """November 'selected the contract' (40-man protection) by player and year (raw/transactions_nov)"""
    import json
    rows = []
    for l in (HERE / "raw" / "transactions_nov.jsonl").open(encoding="utf-8"):
        t = json.loads(l)
        if t.get("type") == "SE" and str(t.get("date", ""))[5:7] == "11":
            rows.append((t.get("pid"), int(str(t["date"])[:4])))
    d = pd.DataFrame(rows, columns=["mlbam", "year"]).dropna().drop_duplicates()
    xm = pd.read_sql("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL", sqlite3.connect(DB))
    xm["mlbam"] = pd.to_numeric(xm.mlbam_id, errors="coerce")
    return d.merge(xm[["player_id", "mlbam"]], on="mlbam")[["player_id", "year"]]


def positions(entry_offset: int = 0, grade: str = "raw", hold_years: int = 1) -> pd.DataFrame:
    """entry_offset: months after November of season Y (0 = Nov, 1 = Dec, ..., 5 = Apr);
    hold_years: the deadline is October of Y + hold_years"""
    px = prices(grade)
    mult, _ = pop(grade)
    ppl = tables()["people"].set_index("player_id")
    out = []
    for y in YEARS:
        buy = pd.Period(f"{y}-11", "M") + entry_offset
        end = min(pd.Period(f"{y + hold_years}-10", "M"), sb.LAST_MONTH)
        if buy not in px.columns or buy > end or (hold_years > 1 and y + hold_years > 2026):
            continue
        d = preds()[preds().S == y].set_index("player_id")
        d = d.join(pd.DataFrame({"p0": px[buy]}), how="inner").dropna(subset=["p0"])
        d = d.join(ppl[["debut_date"]])
        d = d[d.debut_date.isna() | (d.debut_date > buy.to_timestamp(how="end"))]
        dm = d.debut_date.dt.to_period("M")
        d["debuted"] = dm.notna() & (dm <= end)
        pre = [px.at[pid, m - 1] if deb and (m - 1) in px.columns else np.nan
               for pid, m, deb in zip(d.index, dm, d.debuted)]
        d["p_pre"] = pre
        band = pd.cut(d.p_pre, BANDS, labels=BAND_LBL, right=False)
        d["p_exit"] = np.where(d.debuted, d.p_pre * band.map(mult).astype(float), px[end].reindex(d.index))
        d["year"] = y
        out.append(d.dropna(subset=["p_exit"]))
    b = pd.concat(out).rename_axis("player_id").reset_index()
    b["net"] = sb.net(b.p0, b.p_exit)
    adds = forty_man_adds()
    b["added40"] = b.set_index(["player_id", "year"]).index.isin(adds.set_index(["player_id", "year"]).index)
    return b


def summary(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return {"n": 0}
    lo, hi = boot(d.net) if len(d) >= 5 else (np.nan, np.nan)
    cost = (d.p0 * (1 + sb.COSTS["buy_tax"]) + sb.COSTS["buy_ship"]).sum()
    proceeds = (d.p_exit * (1 - sb.COSTS["sell_fee"]) - sb.COSTS["sell_fixed"]).sum()
    yr = d.groupby("year").net.mean()
    return {"n": len(d), "debut": d.debuted.mean(), "med_p0": d.p0.median(), "net": d.net.mean(), "lo": lo, "hi": hi,
            "median": d.net.median(), "$wtd": proceeds / cost - 1,
            "yrs+": int((yr > 0).sum()), **{f"y{y % 100}": yr.get(y, np.nan) for y in YEARS}}


def table(rows: dict) -> pd.DataFrame:
    return pd.DataFrame({k: summary(v) for k, v in rows.items()}).T


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for g in ("raw", "psa10"):
        m, detail = pop(g)
        print(f"days 0-3 pop, {g}:\n{detail.round(2).to_string()}\n")

    print("===== A. entry month, raw, P >= 0.7 / 0.5 =====")
    rows = {}
    for off, name in enumerate(["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]):
        b = positions(off)
        rows[f"{name} P>=0.7"] = b[b.p_deb1 >= 0.7]
        rows[f"{name} P>=0.5"] = b[b.p_deb1 >= 0.5]
    print(table(rows).round(3).to_string(), "\n")

    print("===== B. threshold x price band, raw, November =====")
    b = positions(0)
    rows = {}
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        for lo_, hi_, lab in ((0, 10, "<$10"), (10, 30, "$10-30"), (30, 1e9, "$30+"), (10, 1e9, "$10+")):
            rows[f"P>={thr} {lab}"] = b[(b.p_deb1 >= thr) & (b.p0 >= lo_) & (b.p0 < hi_)]
    print(table(rows).round(3).to_string(), "\n")

    print("===== C. PSA 10 copies, November =====")
    bp = positions(0, "psa10")
    print(table({f"PSA10 P>={t}": bp[bp.p_deb1 >= t] for t in (0.5, 0.6, 0.7, 0.8)}).round(3).to_string(), "\n")

    print("===== D. 40-man add confirmation, December entry (after the Nov adds) =====")
    bd = positions(1)
    print(table({"P>=0.3 & added": bd[(bd.p_deb1 >= 0.3) & bd.added40], "P>=0.5 & added": bd[(bd.p_deb1 >= 0.5) & bd.added40],
                 "P>=0.5 not added": bd[(bd.p_deb1 >= 0.5) & ~bd.added40], "P>=0.7 & added": bd[(bd.p_deb1 >= 0.7) & bd.added40],
                 "P>=0.7 not added": bd[(bd.p_deb1 >= 0.7) & ~bd.added40], "all added": bd[bd.added40]}).round(3).to_string())
