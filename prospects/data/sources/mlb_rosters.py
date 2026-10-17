"""
prospects/data/sources/mlb_rosters.py
=====================================

Every club's 40-man roster on November 30 of each season, from the MLB Stats
API roster endpoint (which accepts a historical date).

Nov 30 falls after the Rule 5 protection deadline, so the snapshot records
which not-yet-debuted prospects their club chose to protect. In the card-market
research (research/card_market/FINDINGS.md, section 1d) November additions
debuted the next season 70% of the time against 20% for other AA/AAA players,
and the flag carried odds x4.3 on top of the model's own probability.

For the current season, before Nov 30, the snapshot is taken today instead.
The cache is written after every season, and a rerun only pulls seasons it
lacks plus the current one, so an interrupted pull resumes.

    python -m prospects.data.sources.mlb_rosters            # fill / refresh the cache
"""

from __future__ import annotations

import csv
import datetime as dt

from prospects.config import ROSTER40_NOV
from prospects.data.sources.milb import _get_json

FIRST_SEASON = 2008  # earliest season the endpoint returns full 40-man rosters
FIELDS = ["season", "date", "team_id", "mlbam_id", "status"]
_TEAMS = "https://statsapi.mlb.com/api/v1/teams?sportId=1&season={season}"
_ROSTER = "https://statsapi.mlb.com/api/v1/teams/{team}/roster?rosterType=40Man&date={date}"


def snapshot_date(season: int, today: dt.date | None = None) -> dt.date:
    today = today or dt.date.today()
    return min(dt.date(season, 11, 30), today)


def pull_season(season: int) -> list[dict]:
    date = snapshot_date(season)
    rows = []
    for t in (_get_json(_TEAMS.format(season=season)) or {}).get("teams", []):
        roster = _get_json(_ROSTER.format(team=t["id"], date=date.isoformat())) or {}
        for p in roster.get("roster", []):
            rows.append({"season": season, "date": date.isoformat(), "team_id": t["id"],
                         "mlbam_id": p["person"]["id"],
                         "status": (p.get("status") or {}).get("code", "")})
    return rows


def read_cache() -> list[dict]:
    if not ROSTER40_NOV.exists():
        return []
    with ROSTER40_NOV.open() as f:
        return [dict(r, season=int(r["season"])) for r in csv.DictReader(f)]


def write_cache(rows: list[dict]) -> None:
    ROSTER40_NOV.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROSTER40_NOV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (int(r["season"]), int(r["team_id"]), int(r["mlbam_id"]))))
    tmp.replace(ROSTER40_NOV)


def refresh(first: int = FIRST_SEASON, last: int | None = None, verbose: bool = True) -> None:
    """Pull every season in first..last the cache lacks, plus the current
    season (its snapshot date moves until Nov 30)."""
    today = dt.date.today()
    last = last or today.year
    rows = read_cache()
    have = {r["season"] for r in rows}
    for season in range(first, last + 1):
        if season in have and season != today.year:
            continue
        got = pull_season(season)
        rows = [r for r in rows if r["season"] != season] + got
        write_cache(rows)
        if verbose:
            print(f"  {season}: {len(got)} roster rows ({snapshot_date(season)})")


if __name__ == "__main__":
    refresh()
    print(f"cache -> {ROSTER40_NOV}")
