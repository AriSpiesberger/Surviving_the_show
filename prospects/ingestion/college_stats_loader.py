"""
prospects/ingestion/college_stats_loader.py
=============================================

Load NCAA Division-I season stats from the ncaa_bbStats data cache
(player_stats_cache/batting+pitching_qualified.csv) and write them as
SeasonStats rows with level="NCAA-D1".

Matching strategy:
  1. mlbamid (numeric) -> prospects.mlbam_id  (preferred, exact)
  2. (last_normalized, first_normalized) -> prospect.name  (fallback)

Coverage: 2021-2025 only — older college seasons aren't in the cache.

Usage:
    python -m prospects.ingestion.college_stats_loader [--db prospects.db]
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata
from typing import Optional

import pandas as pd

from prospects.schema import SeasonStats
from prospects.storage import ProspectDB


NCAA_DATA = r"C:\Users\arisp\anaconda3\Lib\site-packages\data\player_stats_cache"


def _norm(s) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s).lower()).strip()


def _split(full: str) -> tuple[str, str]:
    parts = _norm(full).split()
    sfx = {"jr", "sr", "ii", "iii", "iv"}
    while parts and parts[-1] in sfx:
        parts.pop()
    if len(parts) <= 1:
        return ("", parts[0] if parts else "")
    return parts[0], parts[-1]


def _build_lookups(db: ProspectDB):
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT player_id, mlbam_id, name FROM prospects"
        ).fetchall()
    by_mlbam = {}
    by_name: dict[tuple, list] = {}
    for r in rows:
        if r["mlbam_id"]:
            try:
                by_mlbam[int(r["mlbam_id"])] = r["player_id"]
            except (TypeError, ValueError):
                pass
        f, l = _split(r["name"])
        by_name.setdefault((l, f), []).append(r["player_id"])
    return by_mlbam, by_name


def _resolve(row, by_mlbam, by_name) -> Optional[str]:
    mlbam = row.get("mlbamid")
    if mlbam is not None and not pd.isna(mlbam):
        pid = by_mlbam.get(int(mlbam))
        if pid:
            return pid
    name = row.get("nameascii") or row.get("name") or ""
    f, l = _split(name)
    cands = by_name.get((l, f), [])
    if len(cands) == 1:
        return cands[0]
    return None


def _make_batting_row(r, player_id, year, team_name) -> SeasonStats:
    return SeasonStats(
        player_id=player_id,
        season_year=int(year),
        level="NCAA-D1",
        org=team_name,
        age_during_season=float(r["age"]) if r.get("age") and not pd.isna(r["age"]) else None,
        pa=int(r.get("pa") or 0),
        avg=float(r.get("avg")) if r.get("avg") and not pd.isna(r["avg"]) else None,
        obp=float(r.get("obp")) if r.get("obp") and not pd.isna(r["obp"]) else None,
        slg=float(r.get("slg")) if r.get("slg") and not pd.isna(r["slg"]) else None,
        woba=float(r.get("woba")) if r.get("woba") and not pd.isna(r["woba"]) else None,
        iso=float(r.get("iso")) if r.get("iso") and not pd.isna(r["iso"]) else None,
        k_pct=float(r.get("k%")) if r.get("k%") and not pd.isna(r["k%"]) else None,
        bb_pct=float(r.get("bb%")) if r.get("bb%") and not pd.isna(r["bb%"]) else None,
        babip=float(r.get("babip")) if r.get("babip") and not pd.isna(r["babip"]) else None,
        home_runs=int(r.get("hr") or 0),
        stolen_bases=int(r.get("sb") or 0),
    )


def _make_pitching_row(r, player_id, year, team_name) -> SeasonStats:
    return SeasonStats(
        player_id=player_id,
        season_year=int(year),
        level="NCAA-D1",
        org=team_name,
        age_during_season=float(r["age"]) if r.get("age") and not pd.isna(r["age"]) else None,
        primary_position="P",
        ip=float(r.get("ip") or 0.0),
        era=float(r.get("era")) if r.get("era") and not pd.isna(r["era"]) else None,
        fip=float(r.get("fip")) if r.get("fip") and not pd.isna(r["fip"]) else None,
        whip=float(r.get("whip")) if r.get("whip") and not pd.isna(r["whip"]) else None,
        k9=float(r.get("k/9")) if r.get("k/9") and not pd.isna(r["k/9"]) else None,
        bb9=float(r.get("bb/9")) if r.get("bb/9") and not pd.isna(r["bb/9"]) else None,
        hr9=float(r.get("hr/9")) if r.get("hr/9") and not pd.isna(r["hr/9"]) else None,
    )


def run(db_path: str = "prospects.db", verbose: bool = True) -> None:
    db = ProspectDB(db_path)
    by_mlbam, by_name = _build_lookups(db)
    if verbose:
        print(f"[college] {len(by_mlbam):,} prospects with mlbam_id, "
              f"{len(by_name):,} (last,first) keys")

    n_bat_matched = 0
    n_bat_total = 0
    bat = pd.read_csv(os.path.join(NCAA_DATA, "batting", "batting_qualified.csv"))
    for r in bat.to_dict("records"):
        n_bat_total += 1
        pid = _resolve(r, by_mlbam, by_name)
        if pid is None:
            continue
        s = _make_batting_row(r, pid, r["year"], r.get("team name") or r.get("team", ""))
        db.upsert_season_stats(s)
        n_bat_matched += 1

    n_pit_matched = 0
    n_pit_total = 0
    pit = pd.read_csv(os.path.join(NCAA_DATA, "pitching", "pitching_qualified.csv"))
    for r in pit.to_dict("records"):
        n_pit_total += 1
        pid = _resolve(r, by_mlbam, by_name)
        if pid is None:
            continue
        s = _make_pitching_row(r, pid, r["year"], r.get("team name") or r.get("team", ""))
        db.upsert_season_stats(s)
        n_pit_matched += 1

    if verbose:
        print(f"[college] batting: matched {n_bat_matched:,} / {n_bat_total:,}  "
              f"({n_bat_matched/max(n_bat_total,1):.1%})")
        print(f"[college] pitching: matched {n_pit_matched:,} / {n_pit_total:,}  "
              f"({n_pit_matched/max(n_pit_total,1):.1%})")
        print(f"[college] total NCAA-D1 rows: {n_bat_matched + n_pit_matched:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    args = parser.parse_args()
    run(args.db)
