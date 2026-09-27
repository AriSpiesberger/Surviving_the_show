"""Point-in-time transaction features per (player, snapshot year) (2026-09-24).

Source: reference/transactions/raw (tools/fetch_transactions.py) + team_levels.json. Every
feature for snapshot S uses only transactions dated <= S-09-30 — the date the sheet is scored
(the November 40-man adds of year S are therefore features of S+1, never of S). Rows before the
coverage start year are NaN, not 0: absence of records there means "not recorded", not "nothing
happened" (the populate-only-where-complete rule).

Features (tx_*):
  on40          on the 40-man roster at the cutoff (last 40-man event: SE/OPT/CU/CLW vs OUT/REL/DFA)
  ever40        ever on a 40-man roster before the cutoff
  yrs40         years since first 40-man event (0 if never)
  il_S          IL placements in season S;  il_car: career IL placements before the cutoff
  il60_car      60-day IL placements (major injuries);  tj: "Tommy John" mentioned in an IL note
  promo_S       in-season assignments to a higher level in S;  demo_S: to a lower level
  last_promo_m  month of the latest promotion in S (0 if none)
  asg_S         assignments in S (churn)
  nri_S         non-roster invitation to spring training in S;  nri_car: career count
  trade_S, trade_car   traded;  released_car: released at any point before the cutoff
  waiver_car    claimed off waivers
  any           any transaction on record before the cutoff (coverage indicator)

    python tools/build_tx_features.py --start-year 2010
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reference" / "transactions" / "raw"
LEVELS = {"RK": 1, "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}
ADD40, DROP40 = {"SE", "OPT", "CU", "CLW"}, {"OUT", "REL", "DFA"}


def load_tx() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RAW / "*" / "*.json")):
        for t in json.loads(Path(f).read_text(encoding="utf-8")):
            p = t.get("person") or {}
            if not p.get("id"):
                continue
            rows.append((t.get("id"), p["id"], t.get("date") or t.get("effectiveDate"), t.get("typeCode"),
                         (t.get("description") or "").lower(),
                         (t.get("fromTeam") or {}).get("id"), (t.get("toTeam") or {}).get("id")))
    df = pd.DataFrame(rows, columns=["tid", "mlbam", "date", "code", "desc", "from_team", "to_team"])
    df = df.drop_duplicates(["tid", "mlbam"]).dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    return df.sort_values(["mlbam", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2010, help="first snapshot year with complete coverage")
    ap.add_argument("--db", default=str(ROOT / "prospects_snapshot.db"))
    ap.add_argument("--out", default=str(ROOT / "reference" / "transactions" / "tx_features.csv"))
    a = ap.parse_args()
    tx = load_tx()
    lv = {k: v for k, v in json.loads((ROOT / "reference" / "transactions" / "team_levels.json").read_text()).items()}
    lvl = lambda y, t: LEVELS.get(lv.get(f"{y}:{int(t)}")) if pd.notna(t) else None
    tx["from_lv"] = [lvl(y, t) for y, t in zip(tx.year, tx.from_team)]
    tx["to_lv"] = [lvl(y, t) for y, t in zip(tx.year, tx.to_team)]
    d = tx.desc
    tx["is_il"] = d.str.contains(r"placed .* on the .*(?:disabled|injured) list", regex=True)
    tx["is_il60"] = tx.is_il & d.str.contains("60-day")
    tx["is_tj"] = tx.is_il & d.str.contains("tommy john")
    tx["is_nri"] = d.str.contains(r"invited .*spring training|non-roster invitee|invitation to spring", regex=True)
    tx["is_trade"] = tx.code.eq("TR")
    tx["is_rel"] = tx.code.eq("REL")
    tx["is_clw"] = tx.code.eq("CLW")
    inseason = (tx.month >= 4) & (tx.month <= 9)
    tx["is_promo"] = tx.code.eq("ASG") & inseason & (tx.to_lv > tx.from_lv)
    tx["is_demo"] = tx.code.eq("ASG") & inseason & (tx.to_lv < tx.from_lv)
    tx["is_asg"] = tx.code.eq("ASG") & inseason
    print(f"[tx] {len(tx):,} transactions for {tx.mlbam.nunique():,} players, {tx.date.min().date()} .. {tx.date.max().date()}")

    con = sqlite3.connect(a.db)
    pmap = pd.read_sql("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL", con)
    con.close()
    pmap["mlbam"] = pd.to_numeric(pmap.mlbam_id, errors="coerce")
    pmap = pmap.dropna(subset=["mlbam"]).astype({"mlbam": int}).drop_duplicates("mlbam")
    tx = tx.merge(pmap[["mlbam", "player_id"]], on="mlbam", how="inner")

    # Vectorised point-in-time build. Each event is bucketed into the first snapshot year that may
    # see it: dated <= Sep 30 of year Y -> snapshot Y, later in the year -> snapshot Y+1.
    tx["B"] = tx.year + (tx.month >= 10).astype(int)
    tx["in_S"] = tx.month <= 9                      # dated in season B itself (not the prior offseason)
    flags = ["is_il", "is_il60", "is_tj", "is_nri", "is_trade", "is_rel", "is_clw"]
    per = tx.groupby(["player_id", "B"])[flags].sum()
    s_only = tx[tx.in_S].groupby(["player_id", "B"])[["is_il", "is_promo", "is_demo", "is_asg", "is_nri", "is_trade"]].sum()
    lastpromo = tx[tx.in_S & tx.is_promo].groupby(["player_id", "B"]).month.max().rename("last_promo_m")
    ev = tx[tx.code.isin(ADD40 | DROP40)].sort_values(["player_id", "date"])
    state40 = ev.groupby(["player_id", "B"]).code.last().isin(ADD40).astype(float).rename("on40")
    first40 = tx[tx.code.isin(ADD40)].groupby("player_id").B.min().rename("first40")
    years = list(range(2005, 2027))
    grid = pd.MultiIndex.from_product([tx.player_id.unique(), years], names=["player_id", "B"])
    g = per.reindex(grid, fill_value=0)
    car = g.groupby(level=0).cumsum()                          # career-to-date counts at each snapshot
    f = pd.DataFrame(index=grid)
    f["tx_on40"] = state40.reindex(grid).groupby(level=0).ffill().fillna(0.0)
    f = f.join(first40, on="player_id")
    snap = np.asarray(grid.get_level_values("B"))
    f["tx_ever40"] = (f.first40 <= snap).astype(float)
    f["tx_yrs40"] = np.where(f.first40 <= snap, snap - f.first40, 0.0)
    si = s_only.reindex(grid, fill_value=0)
    f["tx_il_S"], f["tx_il_car"] = si.is_il.values, car.is_il.values
    f["tx_il60_car"], f["tx_tj"] = car.is_il60.values, (car.is_tj.values > 0).astype(float)
    f["tx_promo_S"], f["tx_demo_S"], f["tx_asg_S"] = si.is_promo.values, si.is_demo.values, si.is_asg.values
    f["tx_last_promo_m"] = lastpromo.reindex(grid).fillna(0.0).values
    f["tx_nri_S"], f["tx_nri_car"] = si.is_nri.values, car.is_nri.values
    f["tx_trade_S"], f["tx_trade_car"] = si.is_trade.values, car.is_trade.values
    f["tx_released_car"] = (car.is_rel.values > 0).astype(float)
    f["tx_waiver_car"] = car.is_clw.values
    anyev = tx.groupby(["player_id", "B"]).size().reindex(grid, fill_value=0).groupby(level=0).cumsum()
    f["tx_any"] = (anyev.values > 0).astype(float)
    f = f.drop(columns="first40").reset_index().rename(columns={"B": "snap_year"})
    f = f[(f.snap_year >= a.start_year) & (f.tx_any > 0)]
    # Leak guard: once a player has debuted, his 40-man moves are MLB-roster activity — the
    # MLB-side information the pipeline keeps out of every feature. Blank all tx_* from the
    # debut year on (the sheet only scores pre-debut players anyway).
    con = sqlite3.connect(a.db)
    deb = pd.read_sql("SELECT player_id, mlb_debut_year FROM career_outcomes", con)
    con.close()
    f = f.merge(deb, on="player_id", how="left")
    post = f.mlb_debut_year.notna() & (f.snap_year >= f.mlb_debut_year)
    f = f[~post].drop(columns="mlb_debut_year")
    print(f"[tx] dropped {int(post.sum()):,} post-debut (player, snap) rows")
    f.to_csv(a.out, index=False)
    print(f"[tx] wrote {a.out}: {len(f):,} (player, snap) rows, {f.player_id.nunique():,} players, "
          f"snaps {a.start_year}..2026")


if __name__ == "__main__":
    main()
