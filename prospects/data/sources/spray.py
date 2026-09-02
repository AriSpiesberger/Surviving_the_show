"""
prospects/data/sources/spray.py
===============================

Batted-ball direction and contact quality, scraped from game feeds.

Why this exists: pull%, spray angle and contact quality are normally treated
as Statcast-era, MLB-only metrics. They are not. Every game feed — at every
level, back to 2007 — carries a `hitData` block on batted-ball play events:

    {"trajectory": "line_drive", "hardness": "medium", "location": "8",
     "coordinates": {"coordX": 116.02, "coordY": 103.93}}

`startSpeed` and the `breaks` object are empty below MLB (no pitch tracking),
but the *hit* coordinates are populated regardless, because they come from the
human stringer rather than from Hawkeye. That makes spray angle and pull rate
computable across the entire minor-league history, and `hardness` is a
human-graded contact-quality tag at levels where exit velocity does not exist.

Cost: this is a per-game scrape, roughly 280 KB per game via the playByPlay
endpoint (the full feed/live carries the same content at ~500 KB). One season
across MLB and all full-season levels is on the order of 6,800 games. Runs are
therefore resumable — every parsed gamePk is recorded, and a rerun skips what
it already has.

Usage:
    from prospects.core.storage import ProspectDB
    from prospects.data.sources.spray import pull_spray_season

    db = ProspectDB("prospects.db")
    pull_spray_season(db, 2024, "AA")
    pull_spray_season(db, 2024, "AA", start="2024-06-01", end="2024-06-30")
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections import defaultdict
from typing import Iterable, Optional

import requests

from prospects.core.schema import BattedBallProfile
from prospects.core.storage import ProspectDB
from prospects.data.sources.milb import (
    LEVEL_TO_SPORT_ID,
    USER_AGENT,
    _load_mlbam_map,
    _resolve_prospect_id,
)

STATSAPI = "https://statsapi.mlb.com/api/v1"


# ============================================================================
# FIELD GEOMETRY
#
# Gameday hit coordinates live on a fixed pixel grid with home plate near
# (125.42, 203.5) and the outfield toward decreasing Y. This is the
# long-standing convention behind every public spray chart built on this feed.
#
# Verified against the feed's own `location` (fielder position) tag:
#     location 7 (LF)  ->  -39.5 deg
#     location 8 (CF)  ->   -5.4 deg
#     location 9 (RF)  ->  +35.9 deg
# ============================================================================

HOME_X = 125.42
HOME_Y = 203.5

# Angle (degrees) partitioning the field into thirds. -45 is the left-field
# line, +45 the right-field line.
THIRD_ANGLE = 15.0

_TRAJECTORY_BUCKET = {
    "ground_ball": "gb",
    "line_drive": "ld",
    "fly_ball": "fb",
    "popup": "pu",
    "pop_up": "pu",
}


def spray_angle(coord_x: float, coord_y: float) -> Optional[float]:
    """Horizontal angle off home plate, in degrees.

    Negative is toward left field, positive toward right field, 0 dead center.
    Returns None for coordinates that fall behind the plate.
    """
    dx = coord_x - HOME_X
    dy = HOME_Y - coord_y
    if dy <= 0:
        # Behind home plate — a foul pop or a bad coordinate. Not usable.
        return None
    return math.degrees(math.atan2(dx, dy))


def pull_side_angle(angle: Optional[float],
                    bats: Optional[str]) -> Optional[float]:
    """Re-express a spray angle from the hitter's point of view.

    Positive is toward the hitter's pull side, negative toward the opposite
    field, regardless of handedness. A right-handed hitter pulls to left
    (a negative raw angle), so the sign flips for him.
    """
    if angle is None or not bats:
        return None
    b = str(bats).strip().upper()
    if b == "R":
        return -angle
    if b == "L":
        return angle
    return None    # switch hitter with no per-PA stance resolved


# ============================================================================
# RESUMABLE LEDGER
# ============================================================================

_LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS spray_games_parsed (
    game_pk INTEGER PRIMARY KEY,
    season_year INTEGER,
    level TEXT,
    parsed_at TEXT
);
"""


def _ensure_ledger(db: ProspectDB) -> None:
    with sqlite3.connect(str(db.db_path)) as conn:
        conn.executescript(_LEDGER_SQL)


def _already_parsed(db: ProspectDB, season: int, level: str) -> set:
    with sqlite3.connect(str(db.db_path)) as conn:
        rows = conn.execute(
            "SELECT game_pk FROM spray_games_parsed "
            "WHERE season_year = ? AND level = ?", (season, level)).fetchall()
    return {r[0] for r in rows}


def _mark_parsed(db: ProspectDB, pks: Iterable[int],
                 season: int, level: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with sqlite3.connect(str(db.db_path)) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO spray_games_parsed "
            "(game_pk, season_year, level, parsed_at) VALUES (?, ?, ?, ?)",
            [(pk, season, level, stamp) for pk in pks])


# ============================================================================
# PULL
# ============================================================================

def _get(url: str, timeout: int = 60, retries: int = 2) -> Optional[dict]:
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT},
                             timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return None


def game_pks(season: int, sport_id: int, start: Optional[str] = None,
             end: Optional[str] = None) -> list:
    """Final regular-season game IDs for a season/level."""
    url = (f"{STATSAPI}/schedule?sportId={sport_id}&season={season}"
           f"&gameType=R")
    if start and end:
        url += f"&startDate={start}&endDate={end}"
    data = _get(url, timeout=90)
    if not data:
        return []
    out = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                pk = g.get("gamePk")
                if pk:
                    out.append(pk)
    return sorted(set(out))


def pull_spray_season(
    db: ProspectDB,
    season: int,
    level: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    verbose: bool = True,
    sleep_between_games: float = 0.12,
    max_games: Optional[int] = None,
    resume: bool = True,
) -> int:
    """Scrape batted-ball direction for one season/level.

    Aggregates every batted ball into a per-player season row and upserts it.
    Safe to interrupt: parsed games are recorded and skipped on a rerun.

    Returns the number of player-season rows written.
    """
    if level not in LEVEL_TO_SPORT_ID:
        raise ValueError(f"Unknown level: {level}")
    sport_id = LEVEL_TO_SPORT_ID[level]

    _ensure_ledger(db)
    _load_mlbam_map(db)
    bats_map = _load_bats(db)

    pks = game_pks(season, sport_id, start, end)
    if resume:
        done = _already_parsed(db, season, level)
        pks = [p for p in pks if p not in done]
    if max_games:
        pks = pks[:max_games]

    if verbose:
        print(f"[spray] {level} {season}: {len(pks)} games to parse")
    if not pks:
        return 0

    acc = defaultdict(lambda: {
        "n": 0, "pull": 0, "center": 0, "oppo": 0,
        "hard": 0, "medium": 0, "soft": 0,
        "gb": 0, "fb": 0, "ld": 0, "pu": 0,
        "angles": [],
    })

    parsed = []
    for i, pk in enumerate(pks):
        data = _get(f"{STATSAPI}/game/{pk}/playByPlay")
        if data is None:
            continue
        _accumulate_game(data, acc, bats_map)
        parsed.append(pk)

        if sleep_between_games:
            time.sleep(sleep_between_games)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pks)} games, {len(acc)} batters so far")
        # Checkpoint periodically so a long run is never fully lost.
        if len(parsed) >= 250:
            _mark_parsed(db, parsed, season, level)
            parsed = []

    if parsed:
        _mark_parsed(db, parsed, season, level)

    records = _to_records(acc, season, level)
    n = db.upsert_batted_ball_profile(records)
    if verbose:
        print(f"[spray] {level} {season}: {n} player rows written")
    return n


def _load_bats(db: ProspectDB) -> dict:
    """mlbam_id -> bats handedness, for the players we know."""
    with sqlite3.connect(str(db.db_path)) as conn:
        rows = conn.execute(
            "SELECT mlbam_id, bats FROM prospects "
            "WHERE mlbam_id IS NOT NULL AND bats IS NOT NULL").fetchall()
    return {str(r[0]): r[1] for r in rows}


def _accumulate_game(data: dict, acc: dict, bats_map: dict) -> None:
    for play in data.get("allPlays", []):
        matchup = play.get("matchup") or {}
        batter = (matchup.get("batter") or {}).get("id")
        if not batter:
            continue
        key = str(batter)

        # The stance actually used in this PA. For a switch hitter this is the
        # only correct source — his registered `bats` value is just "S".
        stance = (matchup.get("batSide") or {}).get("code")
        if not stance or stance == "S":
            stance = bats_map.get(key)

        for ev in play.get("playEvents", []):
            hd = ev.get("hitData")
            if not hd:
                continue
            a = acc[key]
            a["n"] += 1

            traj = _TRAJECTORY_BUCKET.get(hd.get("trajectory") or "")
            if traj:
                a[traj] += 1

            hardness = (hd.get("hardness") or "").lower()
            if hardness in ("hard", "medium", "soft"):
                a[hardness] += 1

            coords = hd.get("coordinates") or {}
            cx, cy = coords.get("coordX"), coords.get("coordY")
            if cx is None or cy is None:
                continue
            try:
                ang = spray_angle(float(cx), float(cy))
            except (TypeError, ValueError):
                continue
            if ang is None:
                continue
            a["angles"].append(ang)

            pa_ang = pull_side_angle(ang, stance)
            if pa_ang is None:
                continue
            if pa_ang > THIRD_ANGLE:
                a["pull"] += 1
            elif pa_ang < -THIRD_ANGLE:
                a["oppo"] += 1
            else:
                a["center"] += 1


def _to_records(acc: dict, season: int, level: str) -> list:
    out = []
    for mlbam, a in acc.items():
        player_id = _resolve_prospect_id(mlbam, "", "", db=None)
        if player_id is None:
            continue
        angles = a["angles"]
        mean = sd = None
        if angles:
            mean = sum(angles) / len(angles)
            if len(angles) > 1:
                var = sum((x - mean) ** 2 for x in angles) / (len(angles) - 1)
                sd = math.sqrt(var)
        out.append(BattedBallProfile(
            player_id=player_id,
            season_year=season,
            level=level,
            batted_balls=a["n"],
            pull_n=a["pull"], center_n=a["center"], oppo_n=a["oppo"],
            spray_angle_mean=mean, spray_angle_sd=sd,
            hard_n=a["hard"], medium_n=a["medium"], soft_n=a["soft"],
            gb_n=a["gb"], fb_n=a["fb"], ld_n=a["ld"], pu_n=a["pu"],
        ))
    return out
