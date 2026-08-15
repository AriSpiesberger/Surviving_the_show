"""
prospects/data/sources/milb_advanced.py
=======================================

Pulls the advanced-stat layers that the per-team season endpoint in
`milb.py` does not carry, for MLB and every MiLB level:

    pull_platoon_splits(...)  - vs-LHP / vs-RHP lines (statSplits, sitCodes vl/vr)
    pull_fielding(...)        - innings and rate stats by position
    compute_park_factors(...) - home/road run environment per team

None of these need Statcast — they work at every level and back to the
mid-2000s, which is what makes them usable as *prospect* features rather
than MLB-only context.

Usage:
    from prospects.core.storage import ProspectDB
    from prospects.data.sources.milb_advanced import (
        pull_platoon_splits, pull_fielding, compute_park_factors)

    db = ProspectDB("prospects.db")
    pull_platoon_splits(db, 2024, "AA")
    pull_fielding(db, 2024, "AA")
    compute_park_factors(db, 2024, "AA")
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from prospects.core.schema import FieldingStats, ParkFactor, PlatoonSplit
from prospects.core.storage import ProspectDB
from prospects.data.sources.milb import (
    LEVEL_TO_SPORT_ID,
    USER_AGENT,
    _load_mlbam_map,
    _parse_ip,
    _resolve_prospect_id,
    _to_float,
    _to_int,
)


STATSAPI = "https://statsapi.mlb.com/api/v1"
STITCH = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"


def _get(url: str, timeout: int = 45) -> Optional[dict]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


# ============================================================================
# PLATOON SPLITS
# ============================================================================

_SIT_CODE_TO_SIDE = {"vl": "L", "vr": "R"}


def pull_platoon_splits(
    db: ProspectDB,
    season: int,
    level: str,
    group: str = "hitting",
    verbose: bool = True,
    page_size: int = 1000,
    sleep_between_pages: float = 0.2,
) -> int:
    """Pull vs-LHP and vs-RHP lines for one season/level.

    For hitters, `side` is the pitcher's handedness. For pitchers, it is the
    batter's handedness. A hitter who cannot handle same-side pitching is the
    classic case of a prospect whose overall line overstates him — this is
    the data that catches it.
    """
    if level not in LEVEL_TO_SPORT_ID:
        raise ValueError(f"Unknown level: {level}")
    sport_id = LEVEL_TO_SPORT_ID[level]
    is_pitcher = group == "pitching"

    _load_mlbam_map(db)

    records: list[PlatoonSplit] = []
    for sit, side in _SIT_CODE_TO_SIDE.items():
        offset = 0
        while True:
            url = (
                f"{STATSAPI}/stats?stats=statSplits&sitCodes={sit}"
                f"&group={group}&season={season}&sportId={sport_id}"
                f"&gameType=R&playerPool=ALL&limit={page_size}&offset={offset}"
            )
            data = _get(url)
            if not data:
                break
            blocks = data.get("stats") or []
            splits = []
            for b in blocks:
                splits.extend(b.get("splits") or [])
            if not splits:
                break

            for sp in splits:
                rec = _parse_split(sp, season, level, side, is_pitcher, db)
                if rec is not None:
                    records.append(rec)

            if len(splits) < page_size:
                break
            offset += page_size
            if sleep_between_pages:
                time.sleep(sleep_between_pages)

    n = db.upsert_platoon_splits(records)
    if verbose:
        print(f"[splits] {level} {season} {group}: {n} rows")
    return n


def _parse_split(sp: dict, season: int, level: str, side: str,
                 is_pitcher: bool, db: ProspectDB) -> Optional[PlatoonSplit]:
    player = sp.get("player") or {}
    raw_mlbam = str(player.get("id") or "")
    if not raw_mlbam:
        return None
    full = player.get("fullName") or ""
    parts = full.split()
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""
    pos = (sp.get("position") or {}).get("abbreviation", "")

    player_id = _resolve_prospect_id(
        raw_mlbam, last, first, db=None, full_name=full, position=pos)
    if player_id is None:
        return None

    st = sp.get("stat") or {}
    return PlatoonSplit(
        player_id=player_id,
        season_year=season,
        level=level,
        side=side,
        is_pitcher=is_pitcher,
        pa=_to_int(st.get("plateAppearances")) or 0,
        ab=_to_int(st.get("atBats")),
        hits=_to_int(st.get("hits")),
        doubles=_to_int(st.get("doubles")),
        triples=_to_int(st.get("triples")),
        home_runs=_to_int(st.get("homeRuns")),
        bb=_to_int(st.get("baseOnBalls")),
        ibb=_to_int(st.get("intentionalWalks")),
        hbp=_to_int(st.get("hitByPitch")),
        sf=_to_int(st.get("sacFlies")),
        so=_to_int(st.get("strikeOuts")),
        avg=_to_float(st.get("avg")),
        obp=_to_float(st.get("obp")),
        slg=_to_float(st.get("slg")),
        ops=_to_float(st.get("ops")),
        babip=_to_float(st.get("babip")),
        ground_outs=_to_int(st.get("groundOuts")),
        air_outs=_to_int(st.get("airOuts")),
    )


# ============================================================================
# FIELDING
# ============================================================================

def pull_fielding(
    db: ProspectDB,
    season: int,
    level: str,
    verbose: bool = True,
    page_size: int = 1000,
    sleep_between_pages: float = 0.2,
) -> int:
    """Pull fielding lines (one row per player-position) for a season/level.

    No DRS/UZR/OAA exists below MLB, so what this buys is *positional*: which
    spots a player actually holds, how many innings, and basic reliability.
    Position scarcity is the signal — innings at short or behind the plate
    mean something a corner-outfield line does not.
    """
    if level not in LEVEL_TO_SPORT_ID:
        raise ValueError(f"Unknown level: {level}")
    sport_id = LEVEL_TO_SPORT_ID[level]
    _load_mlbam_map(db)

    records: list[FieldingStats] = []
    offset = 0
    while True:
        url = (
            f"{STITCH}?stitch_env=prod&season={season}&sportId={sport_id}"
            f"&stats=season&group=fielding&gameType=R"
            f"&limit={page_size}&offset={offset}&playerPool=ALL"
        )
        data = _get(url)
        if not data:
            break
        rows = data.get("stats") or []
        if not rows:
            break

        for r in rows:
            rec = _parse_fielding(r, season, level)
            if rec is not None:
                records.append(rec)

        if len(rows) < page_size:
            break
        offset += page_size
        if sleep_between_pages:
            time.sleep(sleep_between_pages)

    n = db.upsert_fielding_stats(records)
    if verbose:
        print(f"[fielding] {level} {season}: {n} rows")
    return n


def _parse_fielding(r: dict, season: int, level: str) -> Optional[FieldingStats]:
    raw_mlbam = str(r.get("playerId") or "")
    if not raw_mlbam:
        return None
    pos = r.get("positionAbbrev") or r.get("primaryPositionAbbrev") or ""
    if not pos:
        return None
    player_id = _resolve_prospect_id(
        raw_mlbam,
        r.get("playerLastName") or "",
        r.get("playerFirstName") or "",
        db=None,
        full_name=r.get("playerFullName") or "",
        position=pos,
    )
    if player_id is None:
        return None

    return FieldingStats(
        player_id=player_id,
        season_year=season,
        level=level,
        position=pos,
        games=_to_int(r.get("games")) or _to_int(r.get("gamesPlayed")),
        games_started=_to_int(r.get("gamesStarted")),
        innings=_parse_ip(r.get("innings")) if r.get("innings") else None,
        chances=_to_int(r.get("chances")),
        putouts=_to_int(r.get("putOuts")),
        assists=_to_int(r.get("assists")),
        errors=_to_int(r.get("errors")),
        throwing_errors=_to_int(r.get("throwingErrors")),
        double_plays=_to_int(r.get("doublePlays")),
        fielding_pct=_to_float(r.get("fielding")),
        range_factor_per9=_to_float(r.get("rangeFactorPer9Inn")),
    )


# ============================================================================
# PARK FACTORS
# ============================================================================

# Regression strength: a park factor computed off a half-season of games is
# mostly noise. `PF_REGRESSION_GAMES` is the number of home games at which the
# raw factor gets equal weight with a neutral 1.00.
PF_REGRESSION_GAMES = 120


def compute_park_factors(
    db: ProspectDB,
    season: int,
    level: str,
    verbose: bool = True,
) -> int:
    """Compute and store park factors for every team at one season/level.

    Method: the standard "halved" home/road factor. For each team, compare
    what happened in its home games to what happened in its road games — both
    sides involve that team's own hitters and pitchers, so team quality
    largely cancels and what is left is the park.

        raw  = (home_stat / home_games) / (road_stat / road_games)
        pf   = 1 + (raw - 1) / 2          # halved: a team plays half its
                                          # games at home
        pf   = regressed toward 1.00 by sample size

    A .300 hitter in the Cal League and a .300 hitter in the Florida State
    League are not the same hitter; this is the correction for that.
    """
    if level not in LEVEL_TO_SPORT_ID:
        raise ValueError(f"Unknown level: {level}")
    sport_id = LEVEL_TO_SPORT_ID[level]

    teams = _team_list(season, sport_id)
    if not teams:
        if verbose:
            print(f"[park] {level} {season}: no teams")
        return 0

    records: list[ParkFactor] = []
    for team_id, abbrev in teams.items():
        home = _team_home_away(season, sport_id, team_id, "home")
        away = _team_home_away(season, sport_id, team_id, "away")
        if not home or not away:
            continue
        pf = _park_factor_from_splits(home, away)
        if pf is None:
            continue
        records.append(ParkFactor(
            team_id=str(team_id),
            season_year=season,
            level=level,
            org=abbrev,
            **pf,
        ))
        time.sleep(0.15)

    n = db.upsert_park_factors(records)
    if verbose:
        print(f"[park] {level} {season}: {n} teams")
    return n


def _team_list(season: int, sport_id: int) -> dict[int, str]:
    data = _get(f"{STATSAPI}/teams?sportId={sport_id}&season={season}")
    if not data:
        return {}
    return {t["id"]: t.get("abbreviation") or t.get("teamName", "")
            for t in data.get("teams", []) if t.get("id")}


def _team_home_away(season: int, sport_id: int, team_id: int,
                    which: str) -> Optional[dict]:
    """Team totals in home or away games, both halves of the inning.

    Uses the team's hitting line plus the line its pitchers allowed, so the
    park factor reflects the full run environment rather than one side of it.
    """
    sit = "h" if which == "home" else "a"
    out: dict[str, float] = {}
    got = False
    for group in ("hitting", "pitching"):
        url = (
            f"{STATSAPI}/teams/{team_id}/stats?stats=statSplits&sitCodes={sit}"
            f"&group={group}&season={season}&sportId={sport_id}&gameType=R"
        )
        data = _get(url)
        if not data:
            continue
        splits = []
        for b in data.get("stats") or []:
            splits.extend(b.get("splits") or [])
        if not splits:
            continue
        st = splits[0].get("stat") or {}
        got = True
        for key, col in (("runs", "runs"), ("homeRuns", "hr"), ("hits", "hits"),
                         ("doubles", "doubles"), ("triples", "triples"),
                         ("strikeOuts", "so"), ("baseOnBalls", "bb")):
            v = _to_float(st.get(key))
            if v is not None:
                out[col] = out.get(col, 0.0) + v
        g = _to_int(st.get("gamesPlayed"))
        if g:
            # Both groups report the same game count; don't double it.
            out["games"] = max(out.get("games", 0.0), float(g))
    return out if got and out.get("games") else None


def _park_factor_from_splits(home: dict, away: dict) -> Optional[dict]:
    hg, ag = home.get("games"), away.get("games")
    if not hg or not ag:
        return None

    def factor(col: str) -> Optional[float]:
        h, a = home.get(col), away.get(col)
        if h is None or a is None or a <= 0:
            return None
        raw = (h / hg) / (a / ag)
        halved = 1.0 + (raw - 1.0) / 2.0
        # Regress toward neutral by sample size.
        w = hg / (hg + PF_REGRESSION_GAMES)
        return 1.0 + (halved - 1.0) * w

    return {
        "pf_runs": factor("runs"),
        "pf_hr": factor("hr"),
        "pf_hits": factor("hits"),
        "pf_doubles": factor("doubles"),
        "pf_triples": factor("triples"),
        "pf_so": factor("so"),
        "pf_bb": factor("bb"),
        "home_games": int(hg),
        "road_games": int(ag),
    }
