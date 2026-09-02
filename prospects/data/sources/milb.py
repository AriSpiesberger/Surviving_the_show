"""
prospects/data/sources/milb.py
====================================

Pulls MiLB (minor league baseball) stats by directly calling the MLB Stats API
endpoint that the armstjc/milb-data-repository uses.

This is the OFFICIAL source - same API the league uses internally - and it's free.
Covers 2005-present at every level (AAA, AA, A+, A, A-, RK).

Usage:
    from prospects.data.sources.milb import pull_milb_season
    pull_milb_season(db, season=2024, level="AAA")
    pull_milb_season(db, season=2024, level="AA")
    # ... etc

For bulk historical pull, see run_bulk_pull.py.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, fields
from typing import Optional

import requests

from prospects.core.schema import SeasonStats
from prospects.core.storage import ProspectDB


MILB_CSV_PATH = "milb_season_stats.csv"  # tee-output sidecar for crash recovery


class _CsvAppender:
    """Append SeasonStats rows to a single CSV. Writes header on first use."""

    def __init__(self, path: str):
        self.path = path
        self._fieldnames = [f.name for f in fields(SeasonStats)] + [
            "stats_type", "pulled_at"
        ]
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        # Open in append mode; keep file handle for the life of the appender.
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, s: SeasonStats, stats_type: str) -> None:
        row = asdict(s)
        row["stats_type"] = stats_type
        row["pulled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._writer.writerow(row)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


# MLB Stats API uses sportId codes for each level
LEVEL_TO_SPORT_ID = {
    "MLB": 1,
    "AAA": 11,
    "AA": 12,
    "A+": 13,
    "A": 14,
    "A-": 15,
    "RK": 16,
    "WIN": 17,
}

USER_AGENT = (
    "Mozilla/5.0 (compatible; ProspectClassifier/1.0; "
    "research project; contact: example@example.com)"
)

# Transient faults from the MLB endpoints (connection resets, read timeouts,
# brief DNS failures) are common on a long pull. Retry them with exponential
# backoff + jitter so a flaky moment never silently drops a team or a level.
_RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _get_json(url: str, timeout: int = 30, retries: int = 6,
              verbose: bool = False) -> Optional[dict]:
    """GET a JSON endpoint with retry/backoff.

    Returns parsed JSON, or None for a definitive empty result (e.g. HTTP 404
    or a non-retryable 4xx). Raises ``requests.RequestException`` only after
    exhausting retries on *transient* faults, so callers can distinguish
    "endpoint says no data" from "we failed to reach it" — the latter must not
    be treated as an empty level.
    """
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (408, 425, 429, 500, 502, 503, 504):
                return None  # 404 and other 4xx: genuinely no data, don't retry
            last = f"HTTP {r.status_code}"
        except _RETRYABLE as e:
            last = e
        except json.JSONDecodeError as e:
            last = e
        if attempt < retries - 1:
            wait = min(2.0 ** attempt + random.uniform(0, 0.75), 30.0)
            if verbose:
                print(f"    retry {attempt + 1}/{retries} in {wait:.1f}s "
                      f"({str(last)[:70]})")
            time.sleep(wait)
    raise requests.exceptions.RequestException(
        f"GET failed after {retries} tries ({str(last)[:80]}): {url[:90]}")


_MLBAM_TO_PROSPECT: Optional[dict] = None
_NAME_TO_PROSPECTS: Optional[dict] = None


def _norm_name(s) -> str:
    import re, unicodedata
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", " ", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def _load_mlbam_map(db: ProspectDB) -> dict:
    """Build mlbam_id -> synthetic prospect_id and (last,first) -> [prospect_id]."""
    global _MLBAM_TO_PROSPECT, _NAME_TO_PROSPECTS
    if _MLBAM_TO_PROSPECT is not None:
        return _MLBAM_TO_PROSPECT
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT player_id, mlbam_id, name FROM prospects"
        ).fetchall()
    _MLBAM_TO_PROSPECT = {}
    _NAME_TO_PROSPECTS = {}
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    for r in rows:
        if r["mlbam_id"]:
            _MLBAM_TO_PROSPECT[r["mlbam_id"]] = r["player_id"]
        parts = _norm_name(r["name"]).split()
        while parts and parts[-1] in suffixes:
            parts.pop()
        if len(parts) >= 2:
            key = (parts[-1], parts[0])
            _NAME_TO_PROSPECTS.setdefault(key, []).append(r["player_id"])
    return _MLBAM_TO_PROSPECT


PERMISSIVE_IFA_MODE = False  # set True before a pull to create IFA stubs
_PERMISSIVE_DB: Optional["ProspectDB"] = None  # set when pull_milb_season starts in permissive mode


def _resolve_prospect_id(
    mlbam_id: str,
    last: str,
    first: str,
    db: Optional[ProspectDB] = None,
    full_name: str = "",
    position: str = "",
) -> Optional[str]:
    """Translate an MLB Stats API player to one of our synthetic prospect IDs.

    STRICT MLBAM MATCHING ONLY (v1.10+). The previous name-fallback path
    caused ~8% of 2005+ MiLB stats to be written to the wrong player_id
    when multiple prospects shared a (last, first) name — see the Thomas
    White / Brandon White / Daniel Cabrera cases. Every player returned
    by the MLB Stats API has a deterministic mlbam_id (= person.id), so
    the name-fallback never produces a more accurate match; it only
    produces collisions.

    Returns None if the player isn't in our prospect universe (skip the
    row), unless permissive mode is on, in which case an IFA stub is
    created keyed on the mlbam_id (which is unique per real player).
    """
    if mlbam_id and _MLBAM_TO_PROSPECT and mlbam_id in _MLBAM_TO_PROSPECT:
        return _MLBAM_TO_PROSPECT[mlbam_id]

    if not PERMISSIVE_IFA_MODE or db is None or not mlbam_id:
        return None

    # Create an IFA stub. Use mlbam_id as the canonical player_id to keep
    # things simple and avoid synthetic-id collisions.
    from prospects.core.schema import Pedigree, Prospect
    name = full_name or f"{first} {last}".strip()
    pos = (position or "").upper()
    is_pitcher = pos in {"P", "RHP", "LHP", "SP", "RP"}
    stub_id = f"ifa_{mlbam_id}"
    p = Prospect(
        player_id=stub_id,
        name=name or f"mlbam_{mlbam_id}",
        is_pitcher=is_pitcher,
        primary_position=pos or "UNK",
        pedigree=Pedigree(is_international=True),
    )
    db.upsert_prospect(p)
    db.set_mlbam_id(stub_id, str(mlbam_id))
    _MLBAM_TO_PROSPECT[mlbam_id] = stub_id
    return stub_id


def pull_milb_season(
    db: ProspectDB,
    season: int,
    level: str,
    stats_type: str = "batting",
    verbose: bool = True,
    sleep_between_teams: float = 0.2,
    csv_appender: Optional[_CsvAppender] = None,
) -> int:
    """
    Pull all player stats for one MiLB season at one level.

    Hits bdfed.stitch.mlbinfra.com per team, aggregates, writes SeasonStats rows.

    Args:
        season: e.g. 2024
        level: "AAA", "AA", "A+", "A", "A-", "RK"
        stats_type: "batting" or "pitching"

    Returns:
        Number of player-season records written.
    """
    if level not in LEVEL_TO_SPORT_ID:
        raise ValueError(f"Unknown level: {level}. Valid: {list(LEVEL_TO_SPORT_ID)}")
    if stats_type not in ("batting", "pitching"):
        raise ValueError(f"stats_type must be 'batting' or 'pitching'")

    sport_id = LEVEL_TO_SPORT_ID[level]

    # Pre-load the mlbam->prospect_id map so we can rewrite IDs as we go.
    _load_mlbam_map(db)
    # Permissive mode needs the DB handle inside _resolve_prospect_id.
    global _PERMISSIVE_DB
    _PERMISSIVE_DB = db if PERMISSIVE_IFA_MODE else None

    if verbose:
        print(f"[milb] Pulling {stats_type} for {level} {season}...")

    # First, get the list of teams at this level/season
    team_ids = _get_team_ids(season, sport_id, verbose=verbose)
    if not team_ids:
        if verbose:
            print(f"[milb] No teams found for {level} {season}")
        return 0

    if verbose:
        print(f"[milb] Found {len(team_ids)} teams. Fetching per-team stats...")

    def _fetch(tid: int) -> int:
        return _pull_team_stats(
            db, season, level, sport_id, tid, stats_type,
            verbose=verbose, csv_appender=csv_appender,
        )

    total_records = 0
    failed: list[int] = []
    for i, team_id in enumerate(team_ids):
        try:
            total_records += _fetch(team_id)
        except Exception as e:
            failed.append(team_id)
            if verbose:
                print(f"  team {team_id} failed inline retries "
                      f"({type(e).__name__}); queued for sweep")

        if sleep_between_teams > 0:
            time.sleep(sleep_between_teams)

        if verbose and (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(team_ids)} teams done, {total_records} records so far")

    # Second-chance sweep: teams that exhausted their inline backoff get one
    # more pass after a cooldown. Anything still failing is reported LOUDLY —
    # never silently dropped — so incomplete data is visible, not assumed.
    if failed:
        if verbose:
            print(f"  sweeping {len(failed)} failed team(s) after cooldown...")
        time.sleep(5.0)
        still: list[int] = []
        for team_id in failed:
            try:
                total_records += _fetch(team_id)
            except Exception:
                still.append(team_id)
            time.sleep(max(sleep_between_teams, 0.5))
        if still:
            print(f"  [milb] WARNING {level} {season} {stats_type}: "
                  f"{len(still)}/{len(team_ids)} teams STILL failed after "
                  f"retries+sweep: {still} — DATA INCOMPLETE, re-run this "
                  f"level to backfill")

    if verbose:
        print(f"[milb] {level} {season} {stats_type}: {total_records} records total")
    return total_records


def _get_team_ids(season: int, sport_id: int, verbose: bool = False) -> list[int]:
    """Get list of team IDs at a given level for a given season."""
    url = (
        f"https://statsapi.mlb.com/api/v1/teams"
        f"?sportId={sport_id}&season={season}"
    )
    # Raises on persistent failure — a network fault must not look like an
    # empty level ("No teams found"), which would silently drop the whole pull.
    data = _get_json(url, verbose=verbose)
    if not data:
        return []
    return [t["id"] for t in data.get("teams", []) if "id" in t]


def _pull_team_stats(
    db: ProspectDB,
    season: int,
    level: str,
    sport_id: int,
    team_id: int,
    stats_type: str,
    verbose: bool = False,
    csv_appender: Optional[_CsvAppender] = None,
) -> int:
    """Pull stats for one team-season-level and write SeasonStats rows."""
    group = "hitting" if stats_type == "batting" else "pitching"
    url = (
        f"https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"
        f"?stitch_env=prod&season={season}&sportId={sport_id}"
        f"&teamId={team_id}&stats=season&group={group}&gameType=R"
        f"&limit=100&offset=0&playerPool=ALL"
    )

    # Raises on persistent failure so the caller can retry-sweep this team
    # rather than counting it as zero records.
    data = _get_json(url, verbose=verbose)
    if not data:
        return 0

    if data.get("totalSplits", 0) == 0:
        return 0

    count = 0
    for player in data.get("stats", []):
        try:
            s = _parse_player_stats(player, season, level, stats_type, team_id)
            if s:
                db.upsert_season_stats(s)
                if csv_appender is not None:
                    csv_appender.write(s, stats_type)
                count += 1
        except Exception as e:
            if verbose:
                print(f"    player parse ERROR: {e}")
    return count


def _parse_player_stats(
    player: dict, season: int, level: str, stats_type: str, team_id: int
) -> Optional[SeasonStats]:
    """Convert an MLB Stats API player record into a SeasonStats."""
    raw_mlbam = str(player.get("playerId", ""))
    if not raw_mlbam:
        return None
    last = player.get("playerLastName") or ""
    first = player.get("playerFirstName") or ""
    full = player.get("playerFullName") or ""
    pos = player.get("primaryPositionAbbrev", "")
    # If permissive mode is on, _resolve_prospect_id creates an IFA stub
    # for unknown players. Otherwise it returns None and we skip the row.
    db_for_resolve = _PERMISSIVE_DB
    player_id = _resolve_prospect_id(
        raw_mlbam, last, first,
        db=db_for_resolve, full_name=full, position=pos,
    )
    if player_id is None:
        return None

    org = player.get("teamAbbrev", "") or str(team_id)
    position = player.get("primaryPositionAbbrev", "")

    if stats_type == "batting":
        return SeasonStats(
            player_id=player_id,
            season_year=season,
            level=level,
            org=org,
            primary_position=position,
            league=player.get("leagueName"),
            league_id=_to_int(player.get("leagueId")),
            pa=int(player.get("plateAppearances", 0) or 0),
            home_runs=int(player.get("homeRuns", 0) or 0),
            stolen_bases=int(player.get("stolenBases", 0) or 0),
            avg=_to_float(player.get("avg")),
            obp=_to_float(player.get("obp")),
            slg=_to_float(player.get("slg")),
            babip=_to_float(player.get("babip")),
            # Compute K% and BB% from raw counts (more reliable than rate fields)
            k_pct=_safe_div(player.get("strikeOuts"), player.get("plateAppearances")),
            bb_pct=_safe_div(player.get("baseOnBalls"), player.get("plateAppearances")),
            iso=_compute_iso(player),

            # ---- Raw counting stats ----
            # Kept raw so wOBA/wRC+/xFIP can be recomputed with era-correct
            # weights without re-pulling. Present at every level back to 2005.
            ab=_to_int(player.get("atBats")),
            hits=_to_int(player.get("hits")),
            doubles=_to_int(player.get("doubles")),
            triples=_to_int(player.get("triples")),
            bb=_to_int(player.get("baseOnBalls")),
            ibb=_to_int(player.get("intentionalWalks")),
            hbp=_to_int(player.get("hitByPitch")),
            sf=_to_int(player.get("sacFlies")),
            sac_bunts=_to_int(player.get("sacBunts")),
            so=_to_int(player.get("strikeOuts")),
            total_bases=_to_int(player.get("totalBases")),
            runs=_to_int(player.get("runs")),
            rbi=_to_int(player.get("rbi")),
            caught_stealing=_to_int(player.get("caughtStealing")),
            gidp=_to_int(player.get("groundIntoDoublePlay")),
            gidp_opp=_to_int(player.get("gidpOpp")),
            left_on_base=_to_int(player.get("leftOnBase")),
            reached_on_error=_to_int(player.get("reachedOnError")),

            # ---- Batted-ball profile ----
            ground_outs=_to_int(player.get("groundOuts")),
            air_outs=_to_int(player.get("airOuts")),
            fly_outs=_to_int(player.get("flyOuts")),
            line_outs=_to_int(player.get("lineOuts")),
            pop_outs=_to_int(player.get("popOuts")),
            ground_hits=_to_int(player.get("groundHits")),
            fly_hits=_to_int(player.get("flyHits")),
            line_hits=_to_int(player.get("lineHits")),
            pop_hits=_to_int(player.get("popHits")),
            balls_in_play=_to_int(player.get("ballsInPlay")),

            # ---- Plate discipline ----
            pitches_seen=_to_int(player.get("numberOfPitches")),
            total_swings=_to_int(player.get("totalSwings")),
            swings_and_misses=_to_int(player.get("swingAndMisses")),
        )
    else:  # pitching
        ip_value = _parse_ip(player.get("inningsPitched"))
        return SeasonStats(
            player_id=player_id,
            season_year=season,
            level=level,
            org=org,
            primary_position=position or "P",
            league=player.get("leagueName"),
            league_id=_to_int(player.get("leagueId")),
            ip=ip_value,
            era=_to_float(player.get("era")),
            whip=_to_float(player.get("whip")),
            k9=_compute_per_9(player.get("strikeOuts"), ip_value),
            bb9=_compute_per_9(player.get("baseOnBalls"), ip_value),
            hr9=_compute_per_9(player.get("homeRuns"), ip_value),

            # ---- Raw counting stats ----
            p_batters_faced=_to_int(player.get("battersFaced")),
            p_ab=_to_int(player.get("atBats")),
            p_hits=_to_int(player.get("hits")),
            p_doubles=_to_int(player.get("doubles")),
            p_triples=_to_int(player.get("triples")),
            p_hr=_to_int(player.get("homeRuns")),
            p_bb=_to_int(player.get("baseOnBalls")),
            p_ibb=_to_int(player.get("intentionalWalks")),
            p_hbp=_to_int(player.get("hitBatsmen")),
            p_so=_to_int(player.get("strikeOuts")),
            p_sf=_to_int(player.get("sacFlies")),
            p_sac_bunts=_to_int(player.get("sacBunts")),
            p_earned_runs=_to_int(player.get("earnedRuns")),
            p_runs=_to_int(player.get("runs")),
            p_total_bases=_to_int(player.get("totalBases")),
            p_gidp=_to_int(player.get("groundIntoDoublePlay")),
            p_gidp_opp=_to_int(player.get("gidpOpp")),
            p_balks=_to_int(player.get("balks")),
            p_wild_pitches=_to_int(player.get("wildPitches")),
            p_pickoffs=_to_int(player.get("pickoffs")),
            p_outs=_to_int(player.get("outs")),

            # ---- Batted-ball profile allowed ----
            p_ground_outs=_to_int(player.get("groundOuts")),
            p_air_outs=_to_int(player.get("airOuts")),
            p_fly_outs=_to_int(player.get("flyOuts")),
            p_line_outs=_to_int(player.get("lineOuts")),
            p_pop_outs=_to_int(player.get("popOuts")),
            p_ground_hits=_to_int(player.get("groundHits")),
            p_fly_hits=_to_int(player.get("flyHits")),
            p_line_hits=_to_int(player.get("lineHits")),
            p_pop_hits=_to_int(player.get("popHits")),
            p_balls_in_play=_to_int(player.get("ballsInPlay")),

            # ---- Pitch-level ----
            p_pitches=_to_int(player.get("numberOfPitches")),
            p_strikes=_to_int(player.get("strikes")),
            p_total_swings=_to_int(player.get("totalSwings")),
            p_swings_and_misses=_to_int(player.get("swingAndMisses")),

            # ---- Role / usage ----
            p_games=_to_int(player.get("gamesPitched")),
            p_games_started=_to_int(player.get("gamesStarted")),
            p_complete_games=_to_int(player.get("completeGames")),
            p_saves=_to_int(player.get("saves")),
            p_holds=_to_int(player.get("holds")),
            p_babip_allowed=_to_float(player.get("babip")),
            p_avg_against=_to_float(player.get("avg")),
            p_obp_against=_to_float(player.get("obp")),
            p_slg_against=_to_float(player.get("slg")),
        )


def _to_int(v) -> Optional[int]:
    """Int or None. The API sends "-" / "" / ".---" for absent values."""
    if v is None or v == "" or v == "-" or v == ".---":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _to_float(v) -> Optional[float]:
    if v is None or v == "" or v == ".---":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_div(num, denom) -> Optional[float]:
    try:
        n = float(num) if num is not None else 0
        d = float(denom) if denom is not None else 0
        return n / d if d > 0 else None
    except (ValueError, TypeError):
        return None


def _compute_iso(player: dict) -> Optional[float]:
    """ISO = SLG - AVG."""
    slg = _to_float(player.get("slg"))
    avg = _to_float(player.get("avg"))
    if slg is None or avg is None:
        return None
    return round(slg - avg, 3)


def _compute_per_9(count, innings: float) -> Optional[float]:
    """Per-9 rate from raw count and IP."""
    if innings <= 0:
        return None
    try:
        c = float(count) if count is not None else 0
        return round(9 * c / innings, 2)
    except (ValueError, TypeError):
        return None


def _parse_ip(ip_str) -> float:
    """
    MiLB API often returns IP like '142.1' meaning 142⅓ innings.
    Convert to proper decimal.
    """
    if ip_str is None or ip_str == "":
        return 0.0
    try:
        ip_str = str(ip_str)
        if "." in ip_str:
            whole, frac = ip_str.split(".", 1)
            return float(whole) + (float(frac) / 3.0 if frac else 0)
        return float(ip_str)
    except (ValueError, TypeError):
        return 0.0


# ============================================================================
# DIAGNOSTIC HELPER
# ============================================================================

def quick_diagnostic(verbose: bool = True) -> dict:
    """
    Test the MLB Stats API endpoint. Run BEFORE bulk pulls.
    Returns dict with what worked.
    """
    results = {}

    # Test 1: team list endpoint
    try:
        url = "https://statsapi.mlb.com/api/v1/teams?sportId=12&season=2024"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        results["team_list"] = {
            "status": r.status_code,
            "team_count": len(r.json().get("teams", [])) if r.status_code == 200 else 0,
        }
        if verbose:
            print(f"[diag] team list (AA 2024): status={r.status_code}, "
                  f"teams={results['team_list']['team_count']}")
    except Exception as e:
        results["team_list"] = {"error": str(e)}
        if verbose:
            print(f"[diag] team list FAILED: {e}")

    # Test 2: a specific team's stats
    try:
        url = (
            "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"
            "?stitch_env=prod&season=2024&sportId=12"
            "&stats=season&group=hitting&gameType=R"
            "&limit=5&offset=0&playerPool=ALL"
        )
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            n_stats = data.get("totalSplits", 0)
            sample = data.get("stats", [])[:1]
            results["bdfed_stats"] = {
                "status": 200,
                "total_splits": n_stats,
                "sample_keys": list(sample[0].keys())[:20] if sample else [],
            }
            if verbose:
                print(f"[diag] bdfed (AA 2024): status=200, n={n_stats}")
                if sample:
                    print(f"  sample keys: {list(sample[0].keys())[:10]}...")
        else:
            results["bdfed_stats"] = {"status": r.status_code}
            if verbose:
                print(f"[diag] bdfed: status={r.status_code}")
    except Exception as e:
        results["bdfed_stats"] = {"error": str(e)}
        if verbose:
            print(f"[diag] bdfed FAILED: {e}")

    return results
