"""
prospects/data/backfills/repull_missing_raw.py
==============================================

Repairs season_stats rows that have a rate line (pa/avg/obp) but no raw
counting stats, by re-pulling only the team-seasons those rows belong to.

Why they exist: the raw columns were added after most of the panel had been
pulled, and a full `--phase all` re-pull still left ~7% of MiLB hitter rows
unfilled — 1,049 players with no filled row anywhere. The pull code is not at
fault; re-running one affected team-season by hand fills it correctly. The
misses are spread thin, roughly one player per team-season, which points at
transient per-player drops during the original sweep rather than a systematic
resolution failure.

That matters beyond tidiness. A row without raw counting stats gets no wOBA
and no cohort percentile, so it falls to MISSING in the panel — and the
players it happens to are not a random sample. Before the MLB backfill they
skewed hard toward never-debuted, which put label signal into a column that
should only carry performance.

Re-pulling only the affected (season, level, team) triples costs roughly a
third of a full sweep. The pull is idempotent: storage.py upserts with
COALESCE, so a re-pull fills the empty columns without disturbing anything
already stored.

Usage:
    python -m prospects.data.backfills.repull_missing_raw --dry-run
    python -m prospects.data.backfills.repull_missing_raw
    python -m prospects.data.backfills.repull_missing_raw --levels AA A+

Follow with woba_backfill and percentile_backfill — both derive from the
columns this repairs, and neither will pick up the new rows on its own.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

from prospects.config import REPO_ROOT
from prospects.core.storage import ProspectDB
from prospects.data.sources.milb import (
    LEVEL_TO_SPORT_ID, USER_AGENT, _load_mlbam_map, _parse_player_stats,
)

STITCH = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"

STATSAPI = "https://statsapi.mlb.com/api/v1"
DEFAULT_LEVELS = ["AAA", "AA", "A+", "A"]


def _missing(conn: sqlite3.Connection, levels: list[str]) -> dict:
    """(season, level) -> {org abbrevs with at least one unfilled row}."""
    marks = ",".join("?" for _ in levels)
    out: dict[tuple[int, str], set] = defaultdict(set)
    # A hitter row is identified by pa>0 and a pitcher row by ip>0; the two
    # merge into one row for two-way players, so check each side separately.
    for qual, raw in (("pa > 0", "ab"), ("ip > 0", "p_hits")):
        rows = conn.execute(
            f"SELECT DISTINCT season_year, level, org FROM season_stats "
            f"WHERE {qual} AND {raw} IS NULL AND org IS NOT NULL "
            f"AND level IN ({marks})", levels).fetchall()
        for yr, lv, org in rows:
            out[(yr, lv)].add(org)
    return out


def _fetch(job: tuple) -> tuple:
    """Fetch one team-season-group. Network only — no DB touched here.

    Fetches run on a thread pool but every write happens on the main thread:
    each upsert opens its own sqlite connection, and concurrent writers would
    just trade throughput for `database is locked`.

    Pages through `totalSplits` rather than trusting a single limit=100 call.
    The original pull did not, so a team-season with over 100 players in a
    group was silently truncated. Nothing observed today comes close, but the
    failure is invisible when it does happen, which is the worst kind.
    """
    season, level, sport_id, team_id, group = job
    rows, offset, total = [], 0, None
    while True:
        url = (f"{STITCH}?stitch_env=prod&season={season}&sportId={sport_id}"
               f"&teamId={team_id}&stats=season&group={group}&gameType=R"
               f"&limit=100&offset={offset}&playerPool=ALL")
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError):
            return job, rows, True
        if total is None:
            total = data.get("totalSplits", 0) or 0
        batch = data.get("stats", []) or []
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= total:
            break
    return job, rows, False


def _team_index(season: int, sport_id: int) -> dict:
    """abbreviation -> team_id for one (season, level)."""
    try:
        r = requests.get(f"{STATSAPI}/teams?sportId={sport_id}&season={season}",
                         headers={"User-Agent": USER_AGENT}, timeout=45)
        r.raise_for_status()
        teams = r.json().get("teams", [])
    except requests.RequestException:
        return {}
    idx = {}
    for t in teams:
        for key in (t.get("abbreviation"), t.get("teamCode"),
                    t.get("fileCode"), str(t.get("id"))):
            if key:
                idx.setdefault(str(key).upper(), t["id"])
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects.db"))
    ap.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent fetches. Writes stay on one thread.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = ProspectDB(args.db)
    _load_mlbam_map(db)
    conn = sqlite3.connect(args.db)
    todo = _missing(conn, args.levels)
    n_teams = sum(len(v) for v in todo.values())
    print(f"[repull] {len(todo)} (season, level) groups, "
          f"{n_teams} team-seasons to repair")
    if args.dry_run:
        for (yr, lv), orgs in sorted(todo.items())[:15]:
            print(f"  {yr} {lv}: {len(orgs)} teams — {', '.join(sorted(orgs)[:8])}")
        print(f"\n(dry run — {n_teams} team-seasons x 2 stat types would be pulled)")
        return

    # Resolve every org abbrev to a team id first: one team-list call per
    # (season, level) instead of one per team.
    jobs, unresolved = [], 0
    for (yr, lv), orgs in sorted(todo.items()):
        sport_id = LEVEL_TO_SPORT_ID.get(lv)
        if sport_id is None:
            continue
        idx = _team_index(yr, sport_id)
        for org in sorted(orgs):
            tid = idx.get(str(org).upper())
            if tid is None:
                unresolved += 1
                continue
            for group in ("hitting", "pitching"):
                jobs.append((yr, lv, sport_id, tid, group))
    print(f"[repull] {len(jobs):,} fetches queued "
          f"({unresolved:,} org abbrevs unmatched)")

    written = failed = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job, raw, err in pool.map(_fetch, jobs):
            done += 1
            if err:
                failed += 1
                continue
            season, level, _sid, team_id, group = job
            stats_type = "batting" if group == "hitting" else "pitching"
            parsed = []
            for p in raw:
                try:
                    s = _parse_player_stats(p, season, level, stats_type, team_id)
                except Exception:  # noqa: BLE001
                    continue
                if s:
                    parsed.append(s)
            written += db.upsert_season_stats_many(parsed)
            if done % 200 == 0:
                print(f"  {done:,}/{len(jobs):,} fetches, "
                      f"{written:,} rows written", flush=True)

    print(f"\n[repull] fetches            {done:,}")
    print(f"[repull] fetch failures     {failed:,}")
    print(f"[repull] org unmatched      {unresolved:,}")
    print(f"[repull] rows written       {written:,}")
    print("[repull] now re-run woba_backfill and percentile_backfill.")


if __name__ == "__main__":
    main()
