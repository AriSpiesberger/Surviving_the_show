"""Remove prospect-list rankings welded onto the wrong person, then rebuild the top-100 /
top-25 labels. (2026-09-23)

The rankings linker matches by NAME, so a veteran's rankings can be attached to a young
namesake: "Tony Blanco Jr." (born 2005) carried his father's 2001 top-100 year, "George
Lombard Jr." his father's 1997, a 2011 draftee "Carlos Gonzalez" the veteran's 2005-08 Baseball
America ranks. Two harms:

  * the sheet: build.py drops every "ever top-100" player, so these prospects vanish (e.g.
    Blanco Jr., 2025 draftees Jimmy Anderson and Juan Cruz);
  * training: the TOP_100 label is realized before the snap (so the row is ineligible) and the
    org-rank features (best / recent org rank, years since first ranked) are the veteran's.

Rules (outcome-independent: they use only the record's own birth date and draft year):
  1. a ranking dated before age 15 (list year < birth_year + 15) is not his;
  2. for a drafted player, a ranking dated on or before his draft year is not his: every list in
     rankings_history is a preseason list (as_of = Jan 1), published before the June draft, and
     lists rank only signed professionals.
Rows are deleted from rankings_history and prospect_rankings; for every affected player
best_overall_rank / year_top_100 / year_top_25 and events_json keys "1" (top-100) and "2"
(top-25) are recomputed from the rows that remain.

Idempotent. Runs in `refresh` after `ages`.

    python -m prospects.data.backfills.repair_impossible_rankings --db prospects.db --dry-run
    python -m prospects.data.backfills.repair_impossible_rankings --db prospects.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3

MIN_RANK_AGE = 15

_BAD_RH = """
    SELECT r.rowid, r.player_id FROM rankings_history r JOIN prospects p USING(player_id)
    WHERE (p.birth_date IS NOT NULL
           AND CAST(substr(r.as_of, 1, 4) AS INTEGER) < CAST(substr(p.birth_date, 1, 4) AS INTEGER) + :age)
       OR (p.player_id LIKE 'draft_%' AND p.draft_year IS NOT NULL
           AND CAST(substr(r.as_of, 1, 4) AS INTEGER) <= p.draft_year)
"""
_BAD_PR = """
    SELECT r.rowid, r.player_id FROM prospect_rankings r JOIN prospects p USING(player_id)
    WHERE (p.birth_date IS NOT NULL
           AND r.year < CAST(substr(p.birth_date, 1, 4) AS INTEGER) + :age)
       OR (p.player_id LIKE 'draft_%' AND p.draft_year IS NOT NULL AND r.year <= p.draft_year)
"""


def _has_table(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()


def run(db_path: str, dry_run: bool = False, verbose: bool = True) -> dict:
    con = sqlite3.connect(db_path)
    rh = con.execute(_BAD_RH, {"age": MIN_RANK_AGE}).fetchall()
    pr = con.execute(_BAD_PR, {"age": MIN_RANK_AGE}).fetchall() if _has_table(con, "prospect_rankings") else []
    affected = sorted({pid for _, pid in rh} | {pid for _, pid in pr})
    stats = {"rankings_history_rows": len(rh), "prospect_rankings_rows": len(pr), "players": len(affected)}
    if verbose:
        print(f"[rankings] impossible rows: rankings_history {len(rh):,}, prospect_rankings {len(pr):,} "
              f"({len(affected):,} players)")
    if dry_run:
        before = {pid: con.execute("SELECT year_top_100 FROM career_outcomes WHERE player_id=?",
                                   (pid,)).fetchone() for pid in affected}
        n100 = sum(1 for v in before.values() if v and v[0] is not None)
        print(f"[rankings] {n100:,} of them currently carry a year_top_100. DRY RUN - nothing written")
        con.close()
        return stats
    with con:
        con.executemany("DELETE FROM rankings_history WHERE rowid=?", [(r,) for r, _ in rh])
        if pr:
            con.executemany("DELETE FROM prospect_rankings WHERE rowid=?", [(r,) for r, _ in pr])
        changed = 0
        for pid in affected:
            rows = con.execute("SELECT CAST(substr(as_of, 1, 4) AS INTEGER), overall_rank FROM rankings_history "
                               "WHERE player_id=? AND overall_rank IS NOT NULL", (pid,)).fetchall()
            best = min((rk for _, rk in rows), default=None)
            y100 = min((y for y, rk in rows if rk <= 100), default=None)
            y25 = min((y for y, rk in rows if rk <= 25), default=None)
            old = con.execute("SELECT year_top_100, events_json FROM career_outcomes WHERE player_id=?",
                              (pid,)).fetchone()
            if old is None:
                continue
            changed += old[0] != y100
            con.execute("UPDATE career_outcomes SET best_overall_rank=?, year_top_100=?, year_top_25=? "
                        "WHERE player_id=?", (best, y100, y25, pid))
            if old[1]:
                d = json.loads(old[1])
                d["1"], d["2"] = y100 is not None, y25 is not None
                con.execute("UPDATE career_outcomes SET events_json=? WHERE player_id=?", (json.dumps(d), pid))
    stats["top100_labels_changed"] = changed
    if verbose:
        print(f"[rankings] deleted {len(rh) + len(pr):,} rows; year_top_100 changed for {changed:,} players")
    con.close()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="prospects.db")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.db, a.dry_run)


if __name__ == "__main__":
    main()
