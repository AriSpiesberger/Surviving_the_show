"""Ceiling events: what do cards do around a first All-Star selection or establishment, and can the
model's P(established) / P(star) be bought ahead of them? (2026-10-25)

A. Event study, monthly SportsCardsPro prices (raw and PSA 10 first autos), vs the market index:
   - first All-Star (career_outcomes.year_all_star_once = Y): rosters are announced early July, so
     windows Jan->Jun Y (run-up), Jun->Aug Y (selection), Aug Y->Jun Y+1 (after)
   - established (year_established_mlb = Y, a season-end fact): Jun->Oct Y, Oct Y->Oct Y+1
B. Strategy: buy in November of snap year S (2021-2024) the first auto of held-out validation
   players (runs/current/scratch/val_rows_v3.csv: the v3 model scored on players it never trained
   on), hold to October S+2 (or the last price), net of study_basket.COSTS; sort on the model's
   P(star within 3y) and P(established within 3y).

    python study_ceiling.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import study_basket as sb
from load import HERE, REPO, tables
from panel import player_panel, wide
from study_debut import boot

pd.set_option("display.width", 220)
LAST = pd.Period("2026-10", "M")


def fresh_labels() -> pd.DataFrame:
    """first All-Star season (raw/allstars.csv, Stats API) and established year recomputed from MLB
    season_stats through 2026 with outcomes.py's rule (cum. 500 PA or 200 IP, or a 3rd season with
    >= 100 PA / 20 IP): career_outcomes stops both at 2021."""
    import sqlite3
    from load import DB
    con = sqlite3.connect(DB)
    ppl = pd.read_sql("select player_id, mlbam_id from prospects where mlbam_id is not null", con)
    ppl["mlbam"] = pd.to_numeric(ppl.mlbam_id, errors="coerce")
    a = pd.read_csv(HERE / "raw" / "allstars.csv").groupby("mlbam").season.min().rename("first_allstar")
    m = pd.read_sql("select player_id, season_year y, sum(coalesce(pa,0)) pa, sum(coalesce(ip,0)) ip from season_stats "
                    "where level='MLB' group by player_id, season_year", con).sort_values(["player_id", "y"])
    m["cpa"], m["cip"] = m.groupby("player_id").pa.cumsum(), m.groupby("player_id").ip.cumsum()
    m["meaningful"] = (m.pa >= 100) | (m.ip >= 20)
    m["n_meaningful"] = m.groupby("player_id").meaningful.cumsum()
    est = m[(m.cpa >= 500) | (m.cip >= 200) | (m.n_meaningful >= 3)].groupby("player_id").y.min().rename("est_year")
    out = ppl.set_index("player_id").join(est)
    out["first_allstar"] = out.mlbam.map(a)
    return out[["est_year", "first_allstar"]]


def px(grade):
    m = player_panel(grade=grade)
    w = wide(m).sort_index(axis=1).ffill(axis=1, limit=3)
    idx = m.drop_duplicates("month").set_index("month").mkt_level.sort_index()
    return w, idx


def window(w, idx, pid, a, b):
    if pid not in w.index or a not in w.columns or b not in w.columns:
        return np.nan, np.nan
    r = w.at[pid, b] - w.at[pid, a]
    return r, r - (idx.get(b, np.nan) - idx.get(a, np.nan))


def event_study():
    co = fresh_labels().rename(columns={"first_allstar": "year_all_star_once", "est_year": "year_established_mlb"})
    for grade in ("raw", "psa10"):
        w, idx = px(grade)
        out = []
        for ev, col, wins in (("first All-Star", "year_all_star_once",
                               [("Jan->Jun", 1, 6, 0), ("Jun->Aug", 6, 8, 0), ("Aug->Jun+1", 8, 6, 1), ("Jan->Dec", 1, 12, 0)]),
                              ("established", "year_established_mlb",
                               [("Jun->Oct", 6, 10, 0), ("Oct->Oct+1", 10, 10, 1), ("Oct-12->Oct", 10, 10, -1)])):
            ys = co[col].dropna()
            ys = ys[(ys >= 2022) & (ys <= 2026)]
            for name, m0, m1, dy in wins:
                rs = []
                for pid, y in ys.items():
                    y = int(y)
                    a = pd.Period(f"{y - 1 if dy == -1 else y}-{m0:02d}", "M")
                    b = pd.Period(f"{y + (1 if dy == 1 else 0)}-{m1:02d}", "M")
                    if b > LAST:
                        continue
                    r, x = window(w, idx, pid, a, b)
                    if not np.isnan(r):
                        rs.append((r, x))
                if len(rs) >= 5:
                    r = np.array(rs)
                    lo, hi = boot(pd.Series(r[:, 1]))
                    out.append({"grade": grade, "event": ev, "window": name, "n": len(r), "raw ret": np.exp(r[:, 0].mean()) - 1,
                                "vs mkt": np.exp(r[:, 1].mean()) - 1, "lo": np.exp(lo) - 1, "hi": np.exp(hi) - 1,
                                "up share": (r[:, 0] > 0).mean()})
        print(pd.DataFrame(out).round(3).to_string(index=False), "\n")


def strategy(hold_years: int = 2):
    v = pd.read_csv(REPO / "runs" / "current" / "scratch" / "val_rows_v3.csv")
    co = tables()["people"].set_index("player_id")[["debut_date"]].join(
        fresh_labels().rename(columns={"first_allstar": "year_all_star_once", "est_year": "year_established_mlb"}))
    res = {}
    for grade in ("raw", "psa10"):
        w, _ = px(grade)
        P = np.exp(w)
        rows = []
        for S in (2021, 2022, 2023, 2024):
            buy = pd.Period(f"{S}-11", "M")
            end = min(pd.Period(f"{S + hold_years}-10", "M"), LAST)
            if buy not in P.columns:
                continue
            g = v[v.snap_year == S].set_index("player_id")
            g = g.join(pd.DataFrame({"p0": P[buy], "p1": P[end]}), how="inner").dropna(subset=["p0", "p1"])
            g = g.join(co[["debut_date", "year_established_mlb", "year_all_star_once"]])
            g["debuted_before"] = g.debut_date.notna() & (g.debut_date < buy.to_timestamp())
            g["S"], g["held_to"] = S, str(end)
            rows.append(g)
        b = pd.concat(rows)
        b = b[b.p0 >= 5]
        b["net"] = sb.net(b.p0, b.p1)
        b["became_est"] = b.year_established_mlb.notna() & (b.year_established_mlb <= b.S + hold_years)
        b["became_star"] = b.year_all_star_once.notna() & (b.year_all_star_once <= b.S + hold_years)
        res[grade] = b

        def summ(d):
            if len(d) < 5:
                return {"n": len(d)}
            lo, hi = boot(d.net)
            yr = d.groupby("S").net.mean()
            return {"n": len(d), "med_p0": d.p0.median(), "est rate": d.became_est.mean(), "star rate": d.became_star.mean(),
                    "net": d.net.mean(), "lo": lo, "hi": hi, "median": d.net.median(),
                    "$wtd": (d.p1 * (1 - sb.COSTS["sell_fee"]) - sb.COSTS["sell_fixed"]).sum()
                    / (d.p0 * (1 + sb.COSTS["buy_tax"]) + sb.COSTS["buy_ship"]).sum() - 1,
                    **{f"y{s % 100}": yr.get(s, np.nan) for s in (2021, 2022, 2023, 2024)}}
        print(f"===== {grade}: buy Nov S, hold to Oct S+{hold_years}; held-out validation players, price >= $5 (n={len(b)}) =====")
        tab = {"all": b}
        for col, nm in (("xp_STAR_PLUS_ELITE_h3", "P(star 3y)"), ("xp_ESTABLISHED_MLB_h3", "P(est 3y)")):
            q = b.groupby("S")[col].rank(pct=True)
            tab[f"{nm} top 10%"] = b[q >= 0.9]
            tab[f"{nm} top 25%"] = b[q >= 0.75]
            tab[f"{nm} bottom 50%"] = b[q < 0.5]
        tab["became star (hindsight)"] = b[b.became_star]
        tab["became established (hindsight)"] = b[b.became_est]
        tab["neither (hindsight)"] = b[~b.became_est & ~b.became_star]
        print(pd.DataFrame({k: summ(d) for k, d in tab.items()}).T.round(3).to_string(), "\n")
    return res


if __name__ == "__main__":
    event_study()
    for h in (1, 2):
        strategy(h)
