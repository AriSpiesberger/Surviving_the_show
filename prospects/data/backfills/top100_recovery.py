"""
prospects/data/backfills/top100_recovery.py
===========================================

Recovers top-100 prospect rankings that were dropped during integration,
which biased the TOP_100_PROSPECT label by era.

THE DEFECT
----------
The raw scraped lists (`reference/baseballcube/bbc_top100/bbc_top100_YYYY.csv`)
carry a full 100 names for every year 2004-2026. But the integrated output
carries far fewer in the early years, and the loss is era-graded:

    2004-2011   78.0 of 100 slots present (mean)
    2012-2015   91.0 of 100 slots present (mean)

The loss happens upstream of the name matcher — within the integrated file
only 0-5 rows per year are unmatched, so ~20 names per early year never
reach it at all. Older players have thinner id-crosswalk coverage and were
dropped rather than carried.

WHY IT MATTERS MORE THAN AN ORDINARY GAP
----------------------------------------
This sits inside the *target*, not a feature. A player who was genuinely a
top-100 prospect in 2006 had roughly a 22% chance of not being labelled as
one, against 9% in 2013. At a 1.50% base rate that is a systematic era
effect in the label itself, so it contaminates every metric computed against
it rather than degrading one input.

NOTE ON A COMPETING DIAGNOSIS
-----------------------------
It has been suggested that the era effect comes from MLB Pipeline publishing
a Top 50 rather than a Top 100 before 2012. That is true of Pipeline, but it
is not the cause here: this database holds no Pipeline rows before 2016, and
its actual source for 2004-2015 is Baseball America, which published a full
Top 100 in every one of those years. Verified — the raw files carry 100 names
per year across the whole span, and 36-49 names per year rank 51-100. Capping
the label at top-50 would therefore discard real signal without addressing
the real defect.

MATCHING SAFETY
---------------
Name matching is what caused the v1.10 incident, where a name fallback wrote
~8% of MiLB stat rows to the wrong player. Two guards apply here:

  1. The normalized name must be UNIQUE in the prospect universe. Any name
     resolving to more than one player is skipped, never guessed.
  2. The player must have season stats within +/-2 years of the ranking.

Guard 2 is what kills the same-name-different-generation collision. It is
also evidence the matching is sound: across 2004-2015, exactly ONE unique
name match failed era corroboration, so the unique names are overwhelmingly
the right players rather than coincidences.

A label is also far more forgiving than a stat row: it is one bit attached to
one player-year, and a wrong one is visible as an implausible career.

Usage:
    python -m prospects.data.backfills.top100_recovery --dry-run
    python -m prospects.data.backfills.top100_recovery
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sqlite3
import unicodedata
from typing import Optional

from prospects.config import REPO_ROOT

RAW_GLOB = str(REPO_ROOT / "reference" / "baseballcube" / "bbc_top100" /
               "bbc_top100_*.csv")

# Ranking year must fall within this many years of a recorded playing season.
ERA_TOLERANCE = 2

RECOVERED_SOURCE = "Baseball America (recovered)"


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode(
        "ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", " ", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def recover(db_path: str = "prospects.db", dry_run: bool = False,
            verbose: bool = True) -> dict:
    conn = sqlite3.connect(db_path)

    name_index: dict = collections.defaultdict(list)
    for pid, name in conn.execute("SELECT player_id, name FROM prospects"):
        name_index[_norm(name)].append(pid)

    seasons: dict = collections.defaultdict(set)
    for pid, yr in conn.execute(
            "SELECT player_id, season_year FROM season_stats "
            "GROUP BY player_id, season_year"):
        seasons[pid].add(yr)

    existing: dict = collections.defaultdict(set)
    for pid, yr in conn.execute(
            "SELECT player_id, CAST(SUBSTR(as_of,1,4) AS INT) "
            "FROM rankings_history WHERE overall_rank <= 100"):
        existing[yr].add(pid)

    inserts = []
    stats = collections.Counter()
    per_year: dict = collections.defaultdict(int)

    for path in sorted(glob.glob(RAW_GLOB)):
        year = int(os.path.basename(path)
                   .replace("bbc_top100_", "").replace(".csv", ""))
        with open(path, encoding="utf-8", errors="replace") as fh:
            rows = list(csv.DictReader(fh))

        for row in rows:
            player = (row.get("player") or "").strip()
            rank = _to_int(row.get("rank"))
            if not player or rank is None or rank > 100:
                continue

            matches = name_index.get(_norm(player))
            if not matches:
                stats["absent_from_universe"] += 1
                continue
            if len(matches) > 1:
                stats["ambiguous_name_skipped"] += 1
                continue

            pid = matches[0]
            if pid in existing[year]:
                stats["already_present"] += 1
                continue
            if not any(abs(s - year) <= ERA_TOLERANCE
                       for s in seasons.get(pid, ())):
                stats["era_mismatch_skipped"] += 1
                continue

            inserts.append((pid, f"{year}-01-01", RECOVERED_SOURCE, rank, 100))
            per_year[year] += 1
            stats["recovered"] += 1

    if inserts and not dry_run:
        conn.executemany(
            "INSERT OR IGNORE INTO rankings_history "
            "(player_id, as_of, source, overall_rank, list_size) "
            "VALUES (?, ?, ?, ?, ?)", inserts)
        conn.commit()

    if verbose:
        print(f"[top100] recovered            : {stats['recovered']:,}")
        print(f"[top100] already present      : {stats['already_present']:,}")
        print(f"[top100] ambiguous (skipped)  : "
              f"{stats['ambiguous_name_skipped']:,}")
        print(f"[top100] era mismatch (skipped): "
              f"{stats['era_mismatch_skipped']:,}")
        print(f"[top100] absent from universe : "
              f"{stats['absent_from_universe']:,}")
        if per_year:
            print("[top100] by year: " + "  ".join(
                f"{y}:{n}" for y, n in sorted(per_year.items())))
        if dry_run:
            print("[top100] DRY RUN - nothing written")

    conn.close()
    return dict(stats)


def rebuild_labels(db_path: str = "prospects.db", dry_run: bool = False,
                   verbose: bool = True) -> dict:
    """Recompute best_overall_rank and year_top_100 / year_top_25 from
    rankings_history, so recovered rows actually reach the label."""
    conn = sqlite3.connect(db_path)

    before = conn.execute(
        "SELECT COUNT(*) FROM career_outcomes WHERE year_top_100 IS NOT NULL"
    ).fetchone()[0]

    best = {}
    first100 = {}
    first25 = {}
    for pid, yr, rank in conn.execute(
            "SELECT player_id, CAST(SUBSTR(as_of,1,4) AS INT), overall_rank "
            "FROM rankings_history WHERE overall_rank IS NOT NULL"):
        if pid not in best or rank < best[pid]:
            best[pid] = rank
        if rank <= 100 and (pid not in first100 or yr < first100[pid]):
            first100[pid] = yr
        if rank <= 25 and (pid not in first25 or yr < first25[pid]):
            first25[pid] = yr

    if not dry_run:
        conn.executemany(
            "UPDATE career_outcomes SET best_overall_rank = ? "
            "WHERE player_id = ?", [(v, k) for k, v in best.items()])
        conn.executemany(
            "UPDATE career_outcomes SET year_top_100 = ? WHERE player_id = ?",
            [(v, k) for k, v in first100.items()])
        conn.executemany(
            "UPDATE career_outcomes SET year_top_25 = ? WHERE player_id = ?",
            [(v, k) for k, v in first25.items()])
        conn.commit()

    after = conn.execute(
        "SELECT COUNT(*) FROM career_outcomes WHERE year_top_100 IS NOT NULL"
    ).fetchone()[0]
    if verbose:
        print(f"[top100] players with year_top_100: {before:,} -> "
              f"{after if not dry_run else '(dry run)'}")
    conn.close()
    return {"before": before, "after": after}


def _to_int(v) -> Optional[int]:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    recover(args.db, dry_run=args.dry_run)
    rebuild_labels(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
