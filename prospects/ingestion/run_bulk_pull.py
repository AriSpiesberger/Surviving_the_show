"""
prospects/ingestion/run_bulk_pull.py
======================================

Orchestrator for the historical data pull. Run this once to bootstrap the
classifier's training corpus.

Recommended sequence:
    1. Diagnostics (verify each source works)
    2. Draft data 2005-2024 from pybaseball (defines our player universe)
    3. MLB outcomes for those players (training labels)
    4. MiLB stats by year-level (training features)
    5. NCAA stats for college draftees (extra training features)

Each phase is independent. If one fails, others still work.

Usage:
    python -m prospects.ingestion.run_bulk_pull --phase diagnostics
    python -m prospects.ingestion.run_bulk_pull --phase draft --start 2005 --end 2024
    python -m prospects.ingestion.run_bulk_pull --phase outcomes
    python -m prospects.ingestion.run_bulk_pull --phase milb --start 2005 --end 2024
    python -m prospects.ingestion.run_bulk_pull --phase ncaa
    python -m prospects.ingestion.run_bulk_pull --phase all  # full pull

Database path defaults to ./prospects.db. Override with --db.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from prospects.storage import ProspectDB


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def phase_diagnostics() -> None:
    banner("PHASE: DIAGNOSTICS")
    print("Checking each data source can be reached and returns data we expect...")

    try:
        from prospects.ingestion.pybaseball_loader import quick_diagnostic as pyb_diag
        print("\n--- pybaseball ---")
        pyb_diag(verbose=True)
    except ImportError as e:
        print(f"pybaseball not installed: {e}")

    try:
        from prospects.ingestion.milb_stats import quick_diagnostic as milb_diag
        print("\n--- MLB Stats API (MiLB) ---")
        milb_diag(verbose=True)
    except Exception as e:
        print(f"milb_stats failed: {e}")

    try:
        from prospects.ingestion.ncaa_loader import quick_diagnostic as ncaa_diag
        print("\n--- ncaa_bbStats ---")
        ncaa_diag(verbose=True)
    except Exception as e:
        print(f"ncaa_loader failed: {e}")


def phase_draft(db: ProspectDB, start: int, end: int) -> None:
    banner(f"PHASE: DRAFT DATA {start}-{end}")
    # Prefer the local ncaa_bbStats draft cache (covers 1965-2025, no scraping).
    # Fall back to pybaseball only if the cache is unavailable.
    try:
        from prospects.ingestion.mlb_draft_cache import pull_draft_from_cache
        pull_draft_from_cache(db, start_year=start, end_year=end, verbose=True)
    except FileNotFoundError as e:
        print(f"[draft] cache unavailable ({e}); falling back to pybaseball scrape")
        from prospects.ingestion.pybaseball_loader import pull_draft_data
        pull_draft_data(db, start_year=start, end_year=end, verbose=True)
    print(f"\nDraft pull complete. Total prospects in DB: {db.count_prospects()}")


def phase_outcomes(db: ProspectDB) -> None:
    banner("PHASE: CAREER OUTCOMES (training labels)")
    from prospects.ingestion.outcomes_loader import pull_outcomes
    pull_outcomes(db, verbose=True)
    print(f"\nOutcomes pull complete. Total outcomes in DB: {db.count_outcomes()}")


def phase_milb(db: ProspectDB, start: int, end: int, levels: list = None) -> None:
    banner(f"PHASE: MILB STATS {start}-{end}")
    from prospects.ingestion.milb_stats import (
        MILB_CSV_PATH, _CsvAppender, pull_milb_season,
    )
    if levels is None:
        levels = ["AAA", "AA", "A+", "A"]
    stat_types = ["batting", "pitching"]
    total = 0
    appender = _CsvAppender(MILB_CSV_PATH)
    print(f"[milb] teeing rows to {MILB_CSV_PATH}")
    try:
        for year in range(start, end + 1):
            for level in levels:
                for stat_type in stat_types:
                    try:
                        n = pull_milb_season(
                            db, season=year, level=level, stats_type=stat_type,
                            verbose=True, csv_appender=appender,
                        )
                        total += n
                        appender.flush()
                    except Exception as e:
                        print(f"[milb] {year} {level} {stat_type} FAILED: {e}")
                    time.sleep(1.0)
    finally:
        appender.close()
    print(f"\nMiLB pull complete. {total} player-season-level records.")
    print(f"Total season_stats rows in DB: {db.count_season_stats()}")
    print(f"CSV sidecar: {MILB_CSV_PATH}")


def phase_ncaa(db: ProspectDB) -> None:
    banner("PHASE: NCAA COLLEGE STATS")
    from prospects.ingestion.ncaa_loader import pull_college_stats_for_player

    # Find all college-drafted players and look up their college stats
    prospects = db.all_prospects()
    college_picks = [
        p for p in prospects
        if not p.get("is_international") and (p.get("origin") or "").strip()
    ]
    print(f"Looking up college stats for {len(college_picks)} college draftees...")

    total = 0
    for i, p in enumerate(college_picks):
        try:
            n = pull_college_stats_for_player(
                db,
                player_id=p["player_id"],
                player_name=p["name"],
                school_substr=p.get("origin"),
                verbose=False,
            )
            total += n
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(college_picks)} processed, {total} season records")
        except Exception as e:
            pass

    print(f"\nNCAA pull complete. {total} college season records added.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["diagnostics", "draft", "outcomes", "milb", "mlb_seasons", "ncaa", "all"],
        required=True,
    )
    parser.add_argument("--start", type=int, default=2005, help="Start year")
    parser.add_argument("--end", type=int, default=2024, help="End year")
    parser.add_argument("--db", default="prospects.db", help="SQLite path")
    parser.add_argument(
        "--levels", nargs="+", default=None,
        help="MiLB levels for the milb phase (default: AAA AA A+ A)",
    )
    parser.add_argument(
        "--permissive-ifa", action="store_true",
        help="During milb phase, also capture unknown players as IFA stubs.",
    )
    args = parser.parse_args()

    print(f"Run started: {datetime.utcnow().isoformat()}")
    print(f"DB path: {args.db}")
    print(f"Phase: {args.phase}")

    if args.phase == "diagnostics":
        phase_diagnostics()
        return

    db = ProspectDB(args.db)

    if args.phase == "draft":
        phase_draft(db, args.start, args.end)
    elif args.phase == "outcomes":
        phase_outcomes(db)
    elif args.phase == "milb":
        if args.permissive_ifa:
            from prospects.ingestion import milb_stats as _ms
            _ms.PERMISSIVE_IFA_MODE = True
            print("[milb] permissive IFA mode ON — unknown mlbam_ids will be "
                  "added as is_international=True stubs")
        phase_milb(db, args.start, args.end, levels=args.levels)
    elif args.phase == "mlb_seasons":
        banner("PHASE: MLB SEASONS (from Lahman)")
        from prospects.ingestion.backfills.mlb_lahman_seasons import pull_mlb_seasons_from_lahman
        pull_mlb_seasons_from_lahman(db, verbose=True)
        print(f"\nMLB seasons complete. Total season_stats rows: {db.count_season_stats()}")
    elif args.phase == "ncaa":
        phase_ncaa(db)
    elif args.phase == "all":
        phase_diagnostics()
        phase_draft(db, args.start, args.end)
        phase_outcomes(db)
        phase_milb(db, args.start, args.end)
        from prospects.ingestion.backfills.mlb_lahman_seasons import pull_mlb_seasons_from_lahman
        pull_mlb_seasons_from_lahman(db, verbose=True)
        phase_ncaa(db)

    banner("FINAL STATS")
    print(f"  prospects:     {db.count_prospects():>8}")
    print(f"  season_stats:  {db.count_season_stats():>8}")
    print(f"  outcomes:      {db.count_outcomes():>8}")


if __name__ == "__main__":
    main()
