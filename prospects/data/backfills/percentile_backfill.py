"""
prospects/data/backfills/percentile_backfill.py
===============================================

Writes the `pct_*` columns on season_stats: each row's rank within its own
(level, season_year) cohort. See features/percentiles.py for what is ranked
and why it is needed — briefly, nearly every rate stat drifts year to year, so
a raw split threshold selects a different population depending on the season,
and a tree has no way to correct for that internally.

MUST run after any pull that adds or changes season_stats rows. A percentile
is only meaningful relative to the cohort it was computed over, so a row added
to a (level, year) after the fact leaves every other rank in that cohort
slightly wrong. The in-progress season gains rows on every pull, which is why
refresh.py runs this directly after `pull`.

Recomputes every cohort from scratch rather than patching — the whole table is
one pass, and a partial update is exactly the failure mode being avoided.

Usage:
    python -m prospects.data.backfills.percentile_backfill
    python -m prospects.data.backfills.percentile_backfill --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from prospects.config import REPO_ROOT
from prospects.features.percentiles import (
    HIT_PCT_NAMES, PIT_PCT_NAMES, attach_percentiles,
)

PCT_COLUMNS = HIT_PCT_NAMES + PIT_PCT_NAMES


def backfill(db_path: str, dry_run: bool = False, verbose: bool = True) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    have = {r[1] for r in conn.execute("PRAGMA table_info(season_stats)")}
    missing = [c for c in PCT_COLUMNS if c not in have]
    if missing:
        raise SystemExit(
            f"[pct] season_stats is missing {len(missing)} pct_ columns "
            f"({', '.join(missing[:4])}...). Open the db through ProspectDB "
            f"once to run the forward migration, then retry.")

    rows = [dict(r) for r in conn.execute("SELECT * FROM season_stats")]
    if verbose:
        print(f"[pct] {len(rows):,} rows loaded")

    # Clear first: a row that no longer qualifies must lose its old rank
    # rather than keep a value computed against a different cohort.
    for r in rows:
        for c in PCT_COLUMNS:
            r.pop(c, None)

    stats = attach_percentiles(rows, verbose=verbose)

    payload = [
        tuple(r.get(c) for c in PCT_COLUMNS)
        + (r["player_id"], r["season_year"], r["level"])
        for r in rows
    ]
    written = sum(1 for r in rows if any(r.get(c) is not None for c in PCT_COLUMNS))
    if not dry_run:
        sets = ", ".join(f"{c} = ?" for c in PCT_COLUMNS)
        with conn:
            conn.executemany(
                f"UPDATE season_stats SET {sets} WHERE player_id = ? "
                f"AND season_year = ? AND level = ?", payload)
    conn.close()

    stats["rows"] = len(rows)
    stats["rows_with_any_pct"] = written
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects.db"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"[pct] no such db: {args.db}")
    print(f"[pct] {args.db}{'  (DRY RUN)' if args.dry_run else ''}")
    s = backfill(args.db, dry_run=args.dry_run)
    print(f"\n[pct] rows                {s['rows']:,}")
    print(f"[pct] cohorts ranked      {s['cohorts']:,}")
    print(f"[pct] cohorts too small   {s['skipped_small']:,}")
    print(f"[pct] rows with a rank    {s['rows_with_any_pct']:,}"
          f"{'  (dry run — nothing written)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
