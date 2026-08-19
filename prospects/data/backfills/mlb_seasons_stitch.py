"""
prospects/data/backfills/mlb_seasons_stitch.py
==============================================

Pulls MLB season lines from the MLB Stats API, replacing the Lahman path as
the source of MLB rows.

Why: Lahman stops at 2021. season_stats therefore held no MLB row for any
season after that, so 1,422 players who reached the majors from 2022 on had
their entire big-league career invisible to the panel — five seasons of
post-debut performance missing for exactly the players closest to the events
the model is trying to predict.

Lahman also carries none of the raw counting detail: no batted-ball split, no
pitch counts, no swings or whiffs. This endpoint carries all of it, and for
MLB *and* MiLB alike, so the same derivations apply at every level.

The league-wide query (no teamId) returns ONE row per player-season with the
season already summed across teams, which is what makes this safe. The
per-team form does not: a traded player comes back as two partial lines, and
letting the (player_id, season_year, level) upsert merge those would quietly
replace a full season with whichever stint happened to be larger. Verified
against 2024: 742 hitters / 182,449 PA and 855 pitchers / 43,116 IP, one row
per player and no duplicates.

Usage:
    python -m prospects.data.backfills.mlb_seasons_stitch --start 2005 --end 2026
    python -m prospects.data.backfills.mlb_seasons_stitch --dry-run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from prospects.config import REPO_ROOT
from prospects.core.storage import ProspectDB
from prospects.data.sources.milb import (
    USER_AGENT, _load_mlbam_map, _parse_player_stats,
)

STITCH = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"


def fetch_season(season: int, group: str, timeout: int = 60) -> list[dict]:
    """Every MLB player's season line for one group, paged to completion."""
    out: list[dict] = []
    offset = 0
    total = None
    while True:
        url = (f"{STITCH}?stitch_env=prod&season={season}&sportId=1"
               f"&stats=season&group={group}&gameType=R"
               f"&limit=1000&offset={offset}&playerPool=ALL")
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT},
                             timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  {season} {group} FAILED at offset {offset}: {e}")
            return out
        if total is None:
            total = data.get("totalSplits", 0) or 0
        batch = data.get("stats", []) or []
        out.extend(batch)
        offset += len(batch)
        if not batch or offset >= total:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects.db"))
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"[mlb] no such db: {args.db}")
    db = ProspectDB(args.db)
    _load_mlbam_map(db)

    written = skipped = 0
    for season in range(args.start, args.end + 1):
        rows = []
        for group in ("hitting", "pitching"):
            raw = fetch_season(season, group)
            stats_type = "batting" if group == "hitting" else "pitching"
            for p in raw:
                try:
                    s = _parse_player_stats(p, season, "MLB", stats_type, 0)
                except Exception:  # noqa: BLE001
                    continue
                if s is None:
                    # Not in our prospect universe — the same strict-mlbam
                    # rule the MiLB pull uses. Nothing is invented here.
                    skipped += 1
                    continue
                rows.append(s)
            time.sleep(args.sleep)
        if not args.dry_run and rows:
            written += db.upsert_season_stats_many(rows)
        print(f"  {season}: {len(rows):,} rows for our prospects", flush=True)

    print(f"\n[mlb] rows upserted     {written:,}"
          f"{'  (dry run)' if args.dry_run else ''}")
    print(f"[mlb] players not ours  {skipped:,}")
    print("[mlb] now re-run woba_backfill and percentile_backfill.")


if __name__ == "__main__":
    main()
