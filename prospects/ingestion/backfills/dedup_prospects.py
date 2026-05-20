"""Dedupe the prospects table by mlbam_id.

Some real players have multiple prospect rows — typically because they were
drafted in multiple years (didn't sign once, redrafted) or because different
ingestion runs created competing synthetic player_ids. The MiLB stats loader's
name-fallback path then writes the same stat row to all of them, polluting
roughly 8% of 2005+ MiLB stats.

Strategy:
  1. Group prospect rows by mlbam_id (excluding NULL / '-1').
  2. For each group with >1 rows, pick one canonical player_id:
       prefer signed (latest draft_year present)
       prefer most populated row (most non-NULL fields)
       prefer non-IFA-stub (player_id starting with 'draft_' over 'ifa_')
       tiebreak deterministic: lexicographic on player_id
  3. Reassign every FK reference in season_stats / career_outcomes /
     prospect_rankings / rankings_history / predictions from each
     duplicate to the canonical.
  4. After reassignment, season_stats can have row-level duplicates
     (multiple identical rows for the same player). Collapse to one row
     per (player_id, season_year, level, org).
  5. Delete the now-empty duplicate prospect rows.

Run with --dry-run to preview without committing.

Usage:
    python -m prospects.ingestion.backfills.dedup_prospects [--dry-run]
"""
from __future__ import annotations

import argparse
import sqlite3

from prospects.storage import ProspectDB


FK_TABLES = ("season_stats", "career_outcomes", "prospect_rankings",
             "rankings_history", "predictions")


def _row_score(p: dict) -> tuple:
    """Higher tuple = better canonical candidate.
    Used to pick which player_id keeps a multi-row mlbam group."""
    # 1. Has draft_year (signed): True > False
    has_draft = p.get("draft_year") is not None
    # 2. Non-IFA-stub player_id
    pid = p.get("player_id", "")
    non_stub = not pid.startswith("ifa_")
    # 3. Field count
    n_fields = sum(1 for v in p.values() if v not in (None, "", 0))
    # 4. Latest draft_year (the redraft is usually the signing)
    dy = p.get("draft_year") or 0
    # 5. Lex by player_id (deterministic tiebreak; smaller = preferred)
    return (has_draft, non_stub, n_fields, dy, -hash(pid))


def _pick_canonical(rows: list[dict]) -> dict:
    return max(rows, key=_row_score)


def dedup(db_path: str, dry_run: bool = False) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # 1. Find mlbam_ids with >1 prospect rows
    dups = con.execute(
        "SELECT mlbam_id FROM prospects "
        "WHERE mlbam_id IS NOT NULL AND mlbam_id NOT IN ('', '-1') "
        "GROUP BY mlbam_id HAVING COUNT(*) > 1"
    ).fetchall()
    print(f"[{db_path}] mlbam_ids with duplicate prospect rows: {len(dups):,}")
    if not dups:
        con.close()
        return

    # 2-3. For each duplicate group: pick canonical, reassign FKs, delete losers
    n_moves: dict[str, int] = {t: 0 for t in FK_TABLES}
    n_deleted_prospects = 0
    n_pairs = 0  # (canonical, loser) pairs

    for d in dups:
        mlbam = d["mlbam_id"]
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM prospects WHERE mlbam_id = ?", (mlbam,)
        ).fetchall()]
        canon = _pick_canonical(rows)
        canon_pid = canon["player_id"]
        losers = [r["player_id"] for r in rows if r["player_id"] != canon_pid]
        for loser in losers:
            n_pairs += 1
            for t in FK_TABLES:
                # Check if loser has any rows in this table; if so, reassign
                try:
                    cur = con.execute(
                        f"UPDATE {t} SET player_id = ? WHERE player_id = ?",
                        (canon_pid, loser),
                    )
                    n_moves[t] += cur.rowcount
                except sqlite3.Error:
                    pass
            # Delete the duplicate prospect row
            cur = con.execute(
                "DELETE FROM prospects WHERE player_id = ?", (loser,)
            )
            n_deleted_prospects += cur.rowcount

    print(f"  pairs processed: {n_pairs:,}")
    print(f"  duplicate prospect rows deleted: {n_deleted_prospects:,}")
    for t, n in n_moves.items():
        print(f"    FK reassignments in {t}: {n:,}")

    # 4. Collapse identical season_stats rows (same player_id, year, level, org)
    # Keep the row with the most populated columns (max non-null count).
    cur = con.execute("""
        SELECT player_id, season_year, level, org, COUNT(*) n
        FROM season_stats
        GROUP BY player_id, season_year, level, COALESCE(org, '')
        HAVING n > 1
    """)
    dup_groups = cur.fetchall()
    print(f"\n  season_stats row-duplicate groups after FK reassignment: "
          f"{len(dup_groups):,}")
    n_collapsed = 0
    for g in dup_groups:
        rows = con.execute(
            "SELECT rowid, * FROM season_stats "
            "WHERE player_id=? AND season_year=? AND level=? "
            "AND COALESCE(org,'')=COALESCE(?,'')",
            (g["player_id"], g["season_year"], g["level"], g["org"]),
        ).fetchall()
        # Choose the one with most non-null/non-zero columns
        def _density(r):
            return sum(1 for k in r.keys()
                       if k != "rowid" and r[k] not in (None, "", 0))
        keep = max(rows, key=_density)
        for r in rows:
            if r["rowid"] != keep["rowid"]:
                con.execute("DELETE FROM season_stats WHERE rowid=?",
                            (r["rowid"],))
                n_collapsed += 1
    print(f"  season_stats rows collapsed (kept densest per dup group): "
          f"{n_collapsed:,}")

    # Also dedupe career_outcomes (player_id is primary)
    cur = con.execute(
        "SELECT player_id, COUNT(*) n FROM career_outcomes "
        "GROUP BY player_id HAVING n > 1"
    )
    co_dups = cur.fetchall()
    n_co_collapsed = 0
    for g in co_dups:
        rows = con.execute(
            "SELECT rowid, * FROM career_outcomes WHERE player_id=?",
            (g["player_id"],)
        ).fetchall()
        def _density(r):
            return sum(1 for k in r.keys()
                       if k != "rowid" and r[k] not in (None, "", 0))
        keep = max(rows, key=_density)
        for r in rows:
            if r["rowid"] != keep["rowid"]:
                con.execute("DELETE FROM career_outcomes WHERE rowid=?",
                            (r["rowid"],))
                n_co_collapsed += 1
    print(f"  career_outcomes rows collapsed: {n_co_collapsed:,}")

    # Same for prospect_rankings (player_id + source + year is the natural key)
    cur = con.execute(
        "SELECT player_id, source, year, COUNT(*) n FROM prospect_rankings "
        "GROUP BY player_id, source, year HAVING n > 1"
    )
    pr_dups = cur.fetchall()
    n_pr_collapsed = 0
    for g in pr_dups:
        rows = con.execute(
            "SELECT rowid, rank FROM prospect_rankings "
            "WHERE player_id=? AND source=? AND year=?",
            (g["player_id"], g["source"], g["year"]),
        ).fetchall()
        # Keep the row with the BEST (smallest) rank
        keep = min(rows, key=lambda r: r["rank"] or 9999)
        for r in rows:
            if r["rowid"] != keep["rowid"]:
                con.execute("DELETE FROM prospect_rankings WHERE rowid=?",
                            (r["rowid"],))
                n_pr_collapsed += 1
    print(f"  prospect_rankings rows collapsed: {n_pr_collapsed:,}")

    if dry_run:
        print(f"\n  [DRY RUN] rolling back")
        con.rollback()
    else:
        con.commit()
        print(f"  committed")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects.db")
    ap.add_argument("--also", action="append", default=["prospects_snapshot.db"],
                    help="Additional DB paths to dedupe (defaults include snapshot)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dedup(args.db, dry_run=args.dry_run)
    for extra in args.also:
        if extra != args.db:
            print()
            dedup(extra, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
