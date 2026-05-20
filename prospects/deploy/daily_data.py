"""Daily MiLB data refresh — runs on Railway at 04:30 UTC (midnight ET).

Pulls current-season batting + pitching stats at every level into prospects.db.
Idempotent: re-running the same day overwrites the season's rows.

Usage (local):
    python -m prospects.deploy.daily_data
    python -m prospects.deploy.daily_data --season 2026 --db /data/prospects.db
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from prospects.ingestion.milb_stats import pull_milb_season, LEVEL_TO_SPORT_ID
from prospects.storage import ProspectDB


LEVELS = ["MLB", "AAA", "AA", "A+", "A", "A-", "RK", "WIN"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None,
                   help="Season year to pull (default: current year)")
    p.add_argument("--db", default=os.environ.get("PROSPECT_DB",
                                                  "/data/prospects.db"),
                   help="Path to prospects.db (default: env PROSPECT_DB "
                        "or /data/prospects.db)")
    p.add_argument("--levels", default=",".join(LEVELS),
                   help=f"Comma-separated levels (default: {','.join(LEVELS)})")
    args = p.parse_args()

    season = args.season or datetime.now(timezone.utc).year
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    unknown = [lv for lv in levels if lv not in LEVEL_TO_SPORT_ID]
    if unknown:
        print(f"ERROR: unknown levels: {unknown}", file=sys.stderr)
        return 2

    t0 = time.time()
    print(f"[daily_data] season={season}  db={args.db}  levels={levels}")
    print(f"[daily_data] start: {datetime.now(timezone.utc).isoformat()}")

    db = ProspectDB(args.db)
    total = 0
    failures: list[tuple[str, str, str]] = []  # (level, stats_type, err)

    for lv in levels:
        for stats_type in ("batting", "pitching"):
            try:
                n = pull_milb_season(db, season=season, level=lv,
                                     stats_type=stats_type, verbose=True)
                total += n
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"  [daily_data] {lv} {stats_type} FAILED: {msg}",
                      file=sys.stderr)
                failures.append((lv, stats_type, msg))

    dt = time.time() - t0
    print(f"\n[daily_data] DONE in {dt:.1f}s  rows={total:,}  "
          f"failures={len(failures)}")
    if failures:
        for lv, st, err in failures:
            print(f"  - {lv}/{st}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
