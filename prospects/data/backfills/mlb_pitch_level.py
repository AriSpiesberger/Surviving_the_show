"""
prospects/data/backfills/mlb_pitch_level.py
===========================================

Fills the batted-ball and pitch-level counters on MLB season_stats rows.

Those columns come from the MLB Stats API per-team stat endpoint, which the
MiLB pull already uses. MLB seasons, though, are sourced from Lahman, and
Lahman carries none of them — no groundOuts, no ballsInPlay, no swings, no
whiffs. The result was a systematic hole: pitch-level coverage runs ~82% of
panel hitter rows, and the missing 18% is *exactly* the MLB rows.

That shape is the problem. Any feature derived from these columns would be
absent precisely for seasons a player spent in the majors, so its missingness
would track having debuted — the far end of the event ladder, and the events
(ESTABLISHED_MLB, ALL_STAR, ELITE, STAR) most sensitive to it.

Only the counters listed below are written. Lahman's own totals are left
alone deliberately: Lahman sums a traded player's stints into one season line,
while this endpoint reports per team, so letting the normal upsert merge them
on (player_id, season_year, level) would quietly replace a full season with
one team's partial one. Counters are summed across a player's teams here
before anything is written, which is the same thing Lahman does.

Usage:
    python -m prospects.data.backfills.mlb_pitch_level --dry-run
    python -m prospects.data.backfills.mlb_pitch_level --start 2005 --end 2026

Follow with woba_backfill and percentile_backfill.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from prospects.config import REPO_ROOT
from prospects.core.storage import ProspectDB
from prospects.data.sources.milb import USER_AGENT, _load_mlbam_map, _resolve_prospect_id
from prospects.data.backfills.repull_missing_raw import _fetch

STATSAPI = "https://statsapi.mlb.com/api/v1"

# API field -> season_stats column. Counting stats only: every one of these is
# summable across a player's teams, which is what makes the multi-stint merge
# safe. Rate fields are deliberately absent — they cannot be summed, and
# Lahman already has them right.
HIT_COLS = {
    "groundOuts": "ground_outs", "airOuts": "air_outs", "flyOuts": "fly_outs",
    "lineOuts": "line_outs", "popOuts": "pop_outs", "groundHits": "ground_hits",
    "flyHits": "fly_hits", "lineHits": "line_hits", "popHits": "pop_hits",
    "ballsInPlay": "balls_in_play", "numberOfPitches": "pitches_seen",
    "totalSwings": "total_swings", "swingAndMisses": "swings_and_misses",
}
PIT_COLS = {
    "groundOuts": "p_ground_outs", "airOuts": "p_air_outs",
    "flyOuts": "p_fly_outs", "lineOuts": "p_line_outs", "popOuts": "p_pop_outs",
    "groundHits": "p_ground_hits", "flyHits": "p_fly_hits",
    "lineHits": "p_line_hits", "popHits": "p_pop_hits",
    "ballsInPlay": "p_balls_in_play", "numberOfPitches": "p_pitches",
    "strikes": "p_strikes", "totalSwings": "p_total_swings",
    "swingAndMisses": "p_swings_and_misses", "battersFaced": "p_batters_faced",
    "gamesStarted": "p_games_started",
}


def _to_int(v):
    if v is None or v == "" or v == "-" or v == ".---":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _teams(season: int) -> list[int]:
    r = requests.get(f"{STATSAPI}/teams?sportId=1&season={season}",
                     headers={"User-Agent": USER_AGENT}, timeout=45)
    r.raise_for_status()
    return [t["id"] for t in r.json().get("teams", [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects.db"))
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"[mlbpitch] no such db: {args.db}")
    db = ProspectDB(args.db)
    _load_mlbam_map(db)

    jobs = []
    for season in range(args.start, args.end + 1):
        try:
            tids = _teams(season)
        except requests.RequestException as e:
            print(f"  {season}: team list FAILED ({e})")
            continue
        for tid in tids:
            for group in ("hitting", "pitching"):
                jobs.append((season, "MLB", 1, tid, group))
    print(f"[mlbpitch] {len(jobs):,} fetches queued "
          f"({args.start}-{args.end})")
    if args.dry_run:
        print("(dry run)")
        return

    # (player_id, season, group) -> {column: summed value}
    acc: dict[tuple, dict] = collections.defaultdict(dict)
    failed = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job, raw, err in pool.map(_fetch, jobs):
            done += 1
            if err:
                failed += 1
                continue
            season, _lvl, _sid, _tid, group = job
            cols = HIT_COLS if group == "hitting" else PIT_COLS
            for p in raw:
                pid = _resolve_prospect_id(
                    str(p.get("playerId", "")),
                    p.get("playerLastName") or "", p.get("playerFirstName") or "")
                if pid is None:
                    continue
                bucket = acc[(pid, season, group)]
                for api_key, col in cols.items():
                    v = _to_int(p.get(api_key))
                    if v is None:
                        continue
                    # Sum across the player's teams — one season line, the way
                    # Lahman reports it.
                    bucket[col] = bucket.get(col, 0) + v
            if done % 200 == 0:
                print(f"  {done:,}/{len(jobs):,} fetches, "
                      f"{len(acc):,} player-seasons", flush=True)

    by_shape: dict[tuple, list] = collections.defaultdict(list)
    for (pid, season, _group), vals in acc.items():
        if not vals:
            continue
        names = tuple(sorted(vals))
        by_shape[names].append(tuple(vals[n] for n in names) + (pid, season))

    written = 0
    conn = sqlite3.connect(args.db, timeout=60.0)
    conn.execute("PRAGMA busy_timeout = 60000")
    with conn:
        for names, payload in by_shape.items():
            sets = ", ".join(f"{n} = ?" for n in names)
            cur = conn.executemany(
                f"UPDATE season_stats SET {sets} WHERE player_id = ? "
                f"AND season_year = ? AND level = 'MLB'", payload)
            written += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.close()

    print(f"\n[mlbpitch] fetches         {done:,}")
    print(f"[mlbpitch] fetch failures  {failed:,}")
    print(f"[mlbpitch] player-seasons  {len(acc):,}")
    print(f"[mlbpitch] rows updated    {written:,}")
    print("[mlbpitch] now re-run woba_backfill and percentile_backfill.")


if __name__ == "__main__":
    main()
