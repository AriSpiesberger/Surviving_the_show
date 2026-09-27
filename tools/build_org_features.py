"""Parent-organization development features, point-in-time (2026-09-24).

Clubs differ in how fast they push prospects to the majors. For a player at snapshot S in parent
org P, highest level L:

  org_rate      share of org P's not-yet-debuted players at level L in seasons S-7..S-3 who debuted
                within 3 years (all resolved by S), shrunk toward the league rate for L (k=30)
  org_rate_rel  org_rate / league rate at L
  org_above     org P's not-yet-debuted players at levels above L in season S (congestion)

Affiliate -> parent via reference/transactions/team_parents.csv (season, abbreviation, level).
Uses only seasons <= S and debuts <= S.

    python tools/build_org_features.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LV = {"RK": 1, "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6}
K = 30.0


def main():
    con = sqlite3.connect(ROOT / "prospects_snapshot.db")
    ss = pd.read_sql("SELECT player_id, season_year AS y, level, org, pa, ip FROM season_stats WHERE level != 'MLB'", con)
    oc = pd.read_sql("SELECT player_id, mlb_debut_year AS debut FROM career_outcomes", con)
    con.close()
    tp = pd.read_csv(ROOT / "reference" / "transactions" / "team_parents.csv").drop_duplicates(["season", "abbr", "level"])
    ss = ss.merge(tp.rename(columns={"season": "y", "abbr": "org"})[["y", "org", "level", "parent"]],
                  on=["y", "org", "level"], how="left")
    ss["lv"] = ss.level.map(LV)
    ss["vol"] = ss.pa.fillna(0) + 3 * ss.ip.fillna(0)
    # one row per (player, season): primary parent (most volume), highest level
    prim = ss.sort_values("vol").groupby(["player_id", "y"]).tail(1)[["player_id", "y", "parent"]]
    top = ss.groupby(["player_id", "y"]).lv.max().rename("L").reset_index()
    ps = prim.merge(top, on=["player_id", "y"]).merge(oc, on="player_id", how="left")
    ps = ps[ps.debut.isna() | (ps.debut > ps.y)]                     # not yet debuted in season y
    ps["deb3"] = (ps.debut > ps.y) & (ps.debut <= ps.y + 3)
    print(f"[org] {len(ps):,} pre-debut player-seasons; parent known {ps.parent.notna().mean():.1%}")

    g = ps.dropna(subset=["parent"]).groupby(["parent", "L", "y"]).agg(n=("deb3", "size"), d=("deb3", "sum")).reset_index()
    lg = ps.groupby(["L", "y"]).agg(n=("deb3", "size"), d=("deb3", "sum")).reset_index()
    out = []
    years = range(2008, 2027)
    for S in years:
        w = g[(g.y >= S - 7) & (g.y <= S - 3)].groupby(["parent", "L"])[["n", "d"]].sum().reset_index()
        wl = lg[(lg.y >= S - 7) & (lg.y <= S - 3)].groupby("L")[["n", "d"]].sum()
        base = (wl.d / wl.n).to_dict()
        w["base"] = w.L.map(base)
        w["org_rate"] = (w.d + K * w.base) / (w.n + K)
        w["org_rate_rel"] = w.org_rate / w.base
        cur = ps[ps.y == S]
        above = cur.dropna(subset=["parent"]).groupby("parent").L.apply(lambda s: s.to_numpy()).to_dict()
        c = cur.merge(w[["parent", "L", "org_rate", "org_rate_rel"]], on=["parent", "L"], how="left")
        c["org_above"] = [float((above.get(p, np.array([])) > l).sum()) if pd.notna(p) else np.nan
                          for p, l in zip(c.parent, c.L)]
        c["snap_year"] = S
        out.append(c[["player_id", "snap_year", "org_rate", "org_rate_rel", "org_above"]])
    f = pd.concat(out, ignore_index=True)
    dest = ROOT / "reference" / "transactions" / "org_features.csv"
    f.to_csv(dest, index=False)
    print(f"[org] wrote {dest}: {len(f):,} rows; org_rate coverage {f.org_rate.notna().mean():.1%}")


if __name__ == "__main__":
    main()
