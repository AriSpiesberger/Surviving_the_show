"""Minor-league honours as point-in-time features (2026-09-24).

Pulls every MiLB-level award / all-star series from the MLB Stats API (recipients carry MLBAM
ids), audits each series for continuity, and keeps only series present in >= 90% of the
seasons 2010-2025 (the populate-only-where-complete rule; e.g. the AFL all-prospect team stops
in 2018 and is dropped). An honour counts for snapshot S only if dated before S-09-20: late-
September post-season teams are published after the sheet is scored and count from S+1.

Features (aw_*), per (player, snapshot year), counting honours before the cutoff:
  aw_allstar      all-star team selections (league mid/post-season, BA level all-stars, ...)
  aw_prospect     "top prospect" type awards
  aw_poy          player / pitcher of the year, MVP, rookie of the year
  aw_weekly       player / pitcher of the week or month (hot-streak markers)
  aw_any          any honour;  aw_*_S: the same, dated in season S only

    python tools/build_award_features.py
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "reference" / "awards"
MLB_LEAGUES = {103, 104}


def milb_awards():
    r = requests.get("https://statsapi.mlb.com/api/v1/awards", timeout=60).json().get("awards", [])
    keep = []
    for a in r:
        name = (a.get("name") or "").lower()
        sport = (a.get("sport") or {}).get("id")
        league = (a.get("league") or {}).get("id")
        if league in MLB_LEAGUES or "world baseball" in name or "hall of fame" in name:
            continue
        if sport in (11, 12, 13, 14, 15, 16) or (league and league not in MLB_LEAGUES and sport is None):
            keep.append(a)
    return keep


def recipients(a):
    f = CACHE / "raw" / f"{a['id']}.json"
    if f.exists():
        return a, json.loads(f.read_text(encoding="utf-8"))
    for i in range(4):
        try:
            r = requests.get(f"https://statsapi.mlb.com/api/v1/awards/{a['id']}/recipients", timeout=60)
            rec = r.json().get("awards", [])
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(rec), encoding="utf-8")
            return a, rec
        except Exception:
            time.sleep(2 * (i + 1))
    return a, []


def kind(name):
    n = name.lower()
    if any(k in n for k in ("manager", "coach", "executive", "umpire", "trainer", "broadcast", "general manager")):
        return "drop"
    if "prospect" in n or "rising stars" in n:
        return "prospect"
    if "of the week" in n or "of the month" in n:
        return "weekly"
    if ("player of the year" in n or "pitcher of the year" in n or "mvp" in n or "most valuable" in n
            or "rookie of the year" in n):
        return "poy"
    if "all-star" in n or "all star" in n:
        return "allstar"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--db", default=str(ROOT / "prospects_snapshot.db"))
    ap.add_argument("--out", default=str(ROOT / "reference" / "awards" / "award_features.csv"))
    a = ap.parse_args()
    awards = milb_awards()
    print(f"[aw] {len(awards)} MiLB award series; fetching recipients", flush=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(recipients, awards))
    rows = []
    for aw, rec in res:
        for r in rec:
            p = r.get("player") or {}
            if p.get("id") and r.get("season"):
                rows.append((aw["id"], aw.get("name", ""), int(r["season"]), r.get("date"), int(p["id"])))
    df = pd.DataFrame(rows, columns=["award", "name", "season", "date", "mlbam"])
    # continuity audit: keep series present in >= 90% of 2010-2025 seasons
    span = set(range(2010, 2026))
    cov = df.groupby("award").season.apply(lambda s: len(set(s) & span) / len(span))
    good = set(cov[cov >= 0.9].index)
    audit = (df.groupby(["award", "name"]).season.agg(["min", "max", "nunique"]).reset_index()
             .assign(coverage=lambda t: t.award.map(cov).round(2), kept=lambda t: t.award.isin(good)))
    audit.to_csv(CACHE / "award_series_audit.csv", index=False)
    df = df[df.award.isin(good)].copy()
    df["kind"] = df.name.map(kind)
    df = df[df.kind != "drop"]
    df["date"] = pd.to_datetime(df.date, errors="coerce").fillna(pd.to_datetime(df.season.astype(str) + "-09-25"))
    print(f"[aw] kept {len(good)} continuous series of {df.award.nunique() if len(df) else 0}; "
          f"{len(df):,} honours; kinds {df.kind.value_counts().to_dict()}", flush=True)

    con = sqlite3.connect(a.db)
    pm = pd.read_sql("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL", con)
    con.close()
    pm["mlbam"] = pd.to_numeric(pm.mlbam_id, errors="coerce")
    pm = pm.dropna(subset=["mlbam"]).astype({"mlbam": int}).drop_duplicates("mlbam")
    df = df.merge(pm[["mlbam", "player_id"]], on="mlbam", how="inner")
    out = []
    for pid, g in df.groupby("player_id"):
        for S in range(a.start_year, 2027):
            h = g[g.date < pd.Timestamp(f"{S}-09-20")]
            if h.empty:
                continue
            s = h[h.season == S]
            out.append((pid, S, float((h.kind == "allstar").sum()), float((h.kind == "prospect").sum()),
                        float((h.kind == "poy").sum()), float((h.kind == "weekly").sum()), float(len(h)),
                        float((s.kind == "allstar").sum()), float((s.kind == "weekly").sum()), float(len(s))))
    f = pd.DataFrame(out, columns=["player_id", "snap_year", "aw_allstar", "aw_prospect", "aw_poy", "aw_weekly",
                                   "aw_any", "aw_allstar_S", "aw_weekly_S", "aw_any_S"])
    f.to_csv(a.out, index=False)
    print(f"[aw] wrote {a.out}: {len(f):,} rows, {f.player_id.nunique():,} players")


if __name__ == "__main__":
    main()
