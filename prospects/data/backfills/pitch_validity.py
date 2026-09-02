"""
prospects/data/backfills/pitch_validity.py
==========================================

Marks which `season_stats` rows have numerically credible pitch-level counts.

THE PROBLEM
-----------
The MLB Stats API returns the same 31 advanced hitting keys for 2005 as for
2026, non-null, at every level. The schema is byte-identical across the span.
But for roughly half the panel the *values* are undercounts, and the failure
is silent — it is visible only by range-checking.

A full census of this database (5,313 level-season-team cells with 3+
qualifying hitters) shows the distribution is sharply bimodal, with almost
nothing in between:

    pitches per PA     cells
    1.5                 1364    <- corrupt
    2.0                 1049    <- corrupt
    2.5                   40
    3.0                   95
    3.5                 1070    <- credible
    4.0                 1680    <- credible

47.1% of cells fall below 3.0 against a true value of ~3.7-4.0. Only 191
cells (3.6%) sit in the ambiguous 2.6-3.4 band, so a 3.0 cut separates the
modes cleanly.

WHY PER-TEAM AND NOT PER-LEAGUE-SEASON
--------------------------------------
The rollout was team-by-team at the boundaries. Measured here:

    AA  2011   30 teams -> 10 corrupt, 20 clean
    A+  2015   30 teams -> 25 corrupt,  5 clean
    A-  2016   21 teams ->  3 corrupt, 18 clean
    RK  2024   75 teams -> 45 corrupt, 30 clean
    AAA 2006   45 teams -> 16 corrupt, 29 clean

A level-season verdict would therefore mislabel entire cohorts in exactly the
transition years. The flag is computed per (level, season_year, org).

WHY THIS MATTERS MORE THAN THE UNDERCOUNT ITSELF
------------------------------------------------
Validity is ordered by level and by era: it clears at AAA first (2007), then
AA (2012), then A/A+ (2016), then the complex leagues (2024), then the DSL
(2025). GCL and AZL are corrupt for their entire existence.

So "has credible pitch counts" encodes "was at a high level in a recent
season" — which is a short walk from the outcome. Feeding the raw values in
lets the model learn the availability pattern instead of the skill. Marking
the cells is what makes the block safe to use at all.

Usage:
    python -m prospects.data.backfills.pitch_validity
    python -m prospects.data.backfills.pitch_validity --db prospects.db --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import Optional


# ============================================================================
# DETECTOR THRESHOLDS
#
# Deliberately loose. The modes sit at ~1.7 and ~3.8, so anything near the cut
# is genuinely ambiguous and is better marked invalid than trusted.
# ============================================================================

MIN_PITCHES_PER_PA = 3.0      # true value ~3.7-4.0
MIN_PITCHES_PER_IP = 10.0     # true value ~15-17
MAX_STRIKE_PCT = 0.75         # true value ~0.60-0.65

# A cell needs this much playing time before its ratio means anything.
MIN_CELL_PA = 300
MIN_CELL_IP = 100.0

# Rows below this are too small to judge alone and inherit the cell verdict.
QUALIFY_PA = 50
QUALIFY_IP = 20.0


def _verdict(pa: float, pitches: float, ip: float,
             p_pitches: float, p_strikes: float) -> Optional[int]:
    """Judge one aggregated cell. None when there is not enough to judge."""
    checks = []

    if pa >= MIN_CELL_PA and pitches > 0:
        checks.append((pitches / pa) >= MIN_PITCHES_PER_PA)

    if ip >= MIN_CELL_IP and p_pitches > 0:
        checks.append((p_pitches / ip) >= MIN_PITCHES_PER_IP)
        if p_strikes > 0:
            checks.append((p_strikes / p_pitches) <= MAX_STRIKE_PCT)

    if not checks:
        return None
    # A cell is valid only if every applicable check passes. The hitting and
    # pitching logs come from the same stringer feed and fail together.
    return 1 if all(checks) else 0


_CELL_SQL = """
SELECT level, season_year, {group_col} AS grp,
       COALESCE(SUM(CASE WHEN pa > 0 THEN pa END), 0)             AS pa,
       COALESCE(SUM(CASE WHEN pa > 0 THEN pitches_seen END), 0)   AS pitches,
       COALESCE(SUM(CASE WHEN ip > 0 THEN ip END), 0)             AS ip,
       COALESCE(SUM(CASE WHEN ip > 0 THEN p_pitches END), 0)      AS p_pitches,
       COALESCE(SUM(CASE WHEN ip > 0 THEN p_strikes END), 0)      AS p_strikes
FROM season_stats
WHERE level != 'NCAA-D1'
GROUP BY level, season_year, {group_col}
"""


def compute(db_path: str = "prospects.db", dry_run: bool = False,
            verbose: bool = True) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(season_stats)")}
    if "pitch_data_valid" not in cols:
        if dry_run:
            print("[validity] column pitch_data_valid missing (dry run)")
        else:
            conn.execute(
                "ALTER TABLE season_stats ADD COLUMN pitch_data_valid INTEGER")

    # Team-level verdicts, then a level-season fallback for cells too small
    # to judge on their own.
    team_verdict: dict = {}
    for r in conn.execute(_CELL_SQL.format(group_col="org")):
        v = _verdict(r["pa"], r["pitches"], r["ip"], r["p_pitches"],
                     r["p_strikes"])
        if v is not None:
            team_verdict[(r["level"], r["season_year"], r["grp"])] = v

    level_verdict: dict = {}
    for r in conn.execute(_CELL_SQL.format(group_col="level")):
        v = _verdict(r["pa"], r["pitches"], r["ip"], r["p_pitches"],
                     r["p_strikes"])
        if v is not None:
            level_verdict[(r["level"], r["season_year"])] = v

    updates = []
    counts = {"valid": 0, "corrupt": 0, "unknown": 0}
    for r in conn.execute(
            "SELECT rowid, level, season_year, org, pa, ip FROM season_stats "
            "WHERE level != 'NCAA-D1'"):
        key_t = (r["level"], r["season_year"], r["org"])
        key_l = (r["level"], r["season_year"])
        v = team_verdict.get(key_t)
        if v is None:
            v = level_verdict.get(key_l)
        if v is None:
            counts["unknown"] += 1
        else:
            counts["valid" if v else "corrupt"] += 1
        updates.append((v, r["rowid"]))

    if not dry_run:
        conn.executemany(
            "UPDATE season_stats SET pitch_data_valid = ? WHERE rowid = ?",
            updates)
        conn.commit()

    if verbose:
        tot = sum(counts.values())
        print(f"[validity] team cells judged : {len(team_verdict):,}")
        print(f"[validity] level cells judged: {len(level_verdict):,}")
        print(f"[validity] rows valid   : {counts['valid']:>8,} "
              f"({100*counts['valid']/tot:.1f}%)")
        print(f"[validity] rows corrupt : {counts['corrupt']:>8,} "
              f"({100*counts['corrupt']/tot:.1f}%)")
        print(f"[validity] rows unknown : {counts['unknown']:>8,} "
              f"({100*counts['unknown']/tot:.1f}%)")
        if dry_run:
            print("[validity] DRY RUN - nothing written")

    conn.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    compute(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
