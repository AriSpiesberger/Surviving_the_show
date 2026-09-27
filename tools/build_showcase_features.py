"""Showcase features: Futures Game selections and winter / Arizona Fall League play (2026-09-24).

  fut_S, fut_car   selected to the All-Star Futures Game in season S (played in July, before the
                   Sep-30 cutoff) / career count to date
  afl_n            Arizona Fall League stints before S (the AFL season labelled W runs Oct-Nov of
                   W, after W's cutoff, so it counts from snapshot W+1)
  afl_last         played in the AFL the previous fall (W = S-1)
  afl_ops_last     OPS in that stint (hitters); afl_era_last for pitchers
  win_n            any winter-league stint (AFL, LVBP, LIDOM, LMP, PWL, ABL, ...) before S
  win_pa_car, win_ip_car   winter-league volume to date

Per-year row counts are printed as a coverage audit; the caller decides the start year.

    python tools/build_showcase_features.py
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "reference" / "showcase"
API = "https://statsapi.mlb.com/api/v1"


def get(url, tries=4):
    for i in range(tries):
        try:
            return requests.get(url, timeout=90).json()
        except Exception:
            time.sleep(2 * (i + 1))
    return {}


def futures_rosters(years):
    rows = []
    for y in years:
        f = CACHE / "futures" / f"{y}.json"
        if f.exists():
            ids = json.loads(f.read_text())
        else:
            sch = get(f"{API}/schedule?sportId=1,21,22,51&season={y}&startDate={y}-06-25&endDate={y}-07-25&gameTypes=A,E,S")
            games = [g["gamePk"] for d in sch.get("dates", []) for g in d["games"]
                     if "futures" in (g["teams"]["away"]["team"]["name"] + g["teams"]["home"]["team"]["name"]).lower()]
            ids = []
            for pk in games:
                box = get(f"{API}/game/{pk}/boxscore")
                for side in ("away", "home"):
                    ids += [int(k[2:]) for k in (box.get("teams", {}).get(side, {}).get("players") or {})]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(sorted(set(ids))))
        rows += [(y, i) for i in ids]
    return pd.DataFrame(rows, columns=["season", "mlbam"])


def winter_stats(years):
    rows = []
    for y in years:
        f = CACHE / "winter" / f"{y}.json"
        if f.exists():
            recs = json.loads(f.read_text())
        else:
            recs = []
            for grp in ("hitting", "pitching"):
                r = get(f"{API}/stats?stats=season&group={grp}&season={y}&sportId=17&playerPool=ALL&limit=20000")
                for s in (r.get("stats") or [{}])[0].get("splits", []):
                    st = s.get("stat", {})
                    recs.append({"mlbam": (s.get("player") or {}).get("id"), "league": (s.get("league") or {}).get("name"),
                                 "grp": grp, "pa": st.get("plateAppearances"), "ops": st.get("ops"),
                                 "ip": st.get("inningsPitched"), "era": st.get("era")})
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(recs))
        rows += [dict(r, season=y) for r in recs]
    d = pd.DataFrame(rows)
    for c in ("pa", "ops", "ip", "era"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["mlbam"]).astype({"mlbam": int})


def main():
    years = range(2005, 2027)
    fut = futures_rosters(years)
    win = winter_stats(range(2005, 2026))
    print("[show] Futures Game selections per year:", fut.groupby("season").size().to_dict())
    print("[show] AFL players per year:", win[win.league == "AFL"].groupby("season").mlbam.nunique().to_dict())
    print("[show] winter-league players per year:", win.groupby("season").mlbam.nunique().to_dict())

    con = sqlite3.connect(ROOT / "prospects_snapshot.db")
    pm = pd.read_sql("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL", con)
    con.close()
    pm["mlbam"] = pd.to_numeric(pm.mlbam_id, errors="coerce")
    pm = pm.dropna(subset=["mlbam"]).astype({"mlbam": int}).drop_duplicates("mlbam")[["mlbam", "player_id"]]
    fut = fut.merge(pm, on="mlbam")
    win = win.merge(pm, on="mlbam")
    afl = win[win.league == "AFL"]
    pids = sorted(set(fut.player_id) | set(win.player_id))
    grid = pd.MultiIndex.from_product([pids, list(years)], names=["player_id", "snap_year"]).to_frame(index=False)
    fy = fut.groupby(["player_id", "season"]).size().rename("n").reset_index()
    g = grid.merge(fy.rename(columns={"season": "snap_year"}), on=["player_id", "snap_year"], how="left").fillna({"n": 0})
    grid["fut_S"] = g.n.values
    grid["fut_car"] = g.groupby("player_id").n.cumsum().values
    # winter season W becomes visible at snapshot W+1
    aflw = afl.groupby(["player_id", "season"]).agg(ops=("ops", "max"), era=("era", "min")).reset_index()
    aflw["snap_year"] = aflw.season + 1
    ww = win.groupby(["player_id", "season"]).agg(pa=("pa", "sum"), ip=("ip", "sum")).reset_index()
    ww["snap_year"] = ww.season + 1
    a1 = grid[["player_id", "snap_year"]].merge(aflw[["player_id", "snap_year", "ops", "era"]], how="left")
    grid["afl_last"] = (a1.ops.notna() | a1.era.notna()).astype(float).values
    grid["afl_ops_last"], grid["afl_era_last"] = a1.ops.values, a1.era.values
    grid["afl_n"] = grid.groupby("player_id").afl_last.cumsum().values
    w1 = grid[["player_id", "snap_year"]].merge(ww[["player_id", "snap_year", "pa", "ip"]], how="left")
    stint = (w1.pa.notna() | w1.ip.notna()).astype(float)
    grid["win_n"] = stint.groupby(w1.player_id).cumsum().values
    grid["win_pa_car"] = w1.pa.fillna(0).groupby(w1.player_id).cumsum().values
    grid["win_ip_car"] = w1.ip.fillna(0).groupby(w1.player_id).cumsum().values
    out = grid[(grid[["fut_car", "afl_n", "win_n"]].sum(axis=1) > 0)]
    dest = CACHE / "showcase_features.csv"
    out.to_csv(dest, index=False)
    print(f"[show] wrote {dest}: {len(out):,} rows, {out.player_id.nunique():,} players")


if __name__ == "__main__":
    main()
