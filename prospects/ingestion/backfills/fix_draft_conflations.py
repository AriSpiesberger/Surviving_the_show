"""Repair name-collision draft conflations.

The MiLB/outcome linker assigns mlbam_id by NAME with no draft-year check, so an
active MLB player's id (and therefore his stats + career outcomes) can get welded
onto a DIFFERENT same-name player's draft record. Symptom: the record's career
starts implausibly relative to its draft_year — e.g. "Sean Murphy, 2010 rd 33"
carrying the real 2016 R3 catcher's 2016-2026 stats and 2019 debut.

Left uncorrected this poisons the features (a multi-year phantom gap between
draft_year and first pro season => wrong age-vs-level, years-in-pro, and
lost-season signals) and puts the player in the wrong draft cohort.

Fix: set draft_year to the player's FIRST PRO SEASON. That is the correct entry
year in BOTH cases — a true conflation (the welded career really did start then)
and a legitimate drafted-but-signed-late player (their real pro entry IS that
season) — so the correction cannot misfire.

The original value is preserved in draft_year_orig and the row is flagged with
draft_conflation_fixed = 1.

    python -m prospects.ingestion.backfills.fix_draft_conflations --dry-run
    python -m prospects.ingestion.backfills.fix_draft_conflations --db prospects_snapshot.db
"""
from __future__ import annotations

import argparse
import sqlite3

# A drafted player normally starts pro ball the year of, or the year after, the
# draft. Stats BEFORE the draft are impossible; a gap this large is implausible.
MAX_PLAUSIBLE_GAP = 4


def _ensure_columns(con: sqlite3.Connection) -> None:
    cols = {r[1] for r in con.execute("PRAGMA table_info(prospects)")}
    if "draft_year_orig" not in cols:
        con.execute("ALTER TABLE prospects ADD COLUMN draft_year_orig INTEGER")
    if "draft_conflation_fixed" not in cols:
        con.execute("ALTER TABLE prospects ADD COLUMN draft_conflation_fixed INTEGER")


def run(db_path: str, dry_run: bool = False, max_gap: int = MAX_PLAUSIBLE_GAP):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    _ensure_columns(con)

    # first pro season per player: prefer first MiLB year (true entry); a
    # player whose only rows are MLB falls back to their first season overall.
    rows = con.execute("""
        SELECT p.player_id, p.name, p.draft_year, p.draft_round,
               MIN(CASE WHEN UPPER(s.level) <> 'MLB' THEN s.season_year END) AS first_milb,
               MIN(s.season_year) AS first_any
        FROM prospects p
        JOIN season_stats s ON s.player_id = p.player_id
        WHERE p.draft_year IS NOT NULL
        GROUP BY p.player_id
    """).fetchall()

    fixes = []
    for r in rows:
        first = r["first_milb"] if r["first_milb"] is not None else r["first_any"]
        if first is None:
            continue
        dy = int(r["draft_year"])
        gap = int(first) - dy
        if first < dy or gap >= max_gap:
            fixes.append((int(first), dy, r["player_id"], r["name"], gap))

    print(f"[conflation] scanned {len(rows):,} drafted players with stats")
    print(f"[conflation] flagged {len(fixes):,} with career inconsistent with "
          f"draft_year (starts before draft, or gap >= {max_gap})")
    for first, dy, pid, name, gap in sorted(fixes, key=lambda x: -abs(x[4]))[:10]:
        print(f"    {name:<24} draft {dy} -> entry {first}  (gap {gap:+d})")

    if dry_run:
        print("[conflation] DRY RUN — no changes written")
        con.close()
        return len(fixes)

    con.executemany(
        "UPDATE prospects SET draft_year_orig = COALESCE(draft_year_orig, draft_year), "
        "draft_year = ?, draft_conflation_fixed = 1 "
        "WHERE player_id = ? AND draft_year = ?",
        [(first, pid, dy) for first, dy, pid, _n, _g in fixes],
    )
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM prospects "
                    "WHERE draft_conflation_fixed = 1").fetchone()[0]
    print(f"[conflation] corrected {n:,} records "
          f"(original preserved in draft_year_orig)")
    con.close()
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gap", type=int, default=MAX_PLAUSIBLE_GAP)
    args = ap.parse_args()
    run(args.db, args.dry_run, args.max_gap)
