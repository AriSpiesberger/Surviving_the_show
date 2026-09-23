"""Remove careers welded onto the wrong person: nothing professional before age 15,
no MLB event before age 17. (2026-09-22)

The outcome / season linkers match by NAME, so a veteran's career can be attached to a
young namesake whose player_id (and therefore MLBAM birth date) is his own: "Tony Blanco
Jr." (born 2005) carried his father's 2005 debut, a 2006-born "Jose Pirela" the veteran's
2014 one. Once birth dates were backfilled these became visible as impossible ages — a
"12-year-old" playing AAA, a debut at 8. Two harms:

  * training: every such record is a FAKE POSITIVE debut at an absurd age-for-level
    (97 at-risk landmark rows in the 2026-09-17 panel);
  * the sheet: an active young prospect wearing a namesake's debut is dropped as a
    "pre-snap debutee" (17 players active in the 2026 minors, e.g. Blanco Jr.,
    William Bergolla Jr., Pirela, Charlie Zink).

Rule (outcome-independent: it depends only on the record's own birth date):
  1. delete season_stats rows with age_during_season < 15;
  2. for any player whose MLB debut year is < birth_year + 17, the MLB career is not his:
     delete his MLB-level season_stats rows and clear every MLB-derived outcome
     (debut, final year, established, all-star, awards, career PA/IP/WAR, events).

Idempotent. Runs in `refresh` after `birthdates` (it needs birth dates and ages).

    python -m prospects.data.backfills.repair_impossible_ages --db prospects.db --dry-run
    python -m prospects.data.backfills.repair_impossible_ages --db prospects.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3

MIN_PRO_AGE = 15
MIN_MLB_AGE = 17
MLB_OUTCOME_COLS = ["mlb_debut_year", "final_mlb_year", "year_established_mlb",
                    "year_all_star_once", "year_all_star_three", "year_major_award",
                    "year_hof_trajectory", "career_pa", "career_ip", "career_war",
                    "all_star_selections", "mvp_count", "cy_young_count", "roy_count"]
# events_json keys that are MLB events (CareerEvent ints); 1/2 are prospect-list events
MLB_EVENT_KEYS = {"3", "4", "5", "6", "7", "8"}


def run(db_path: str, dry_run: bool = False, verbose: bool = True) -> dict:
    con = sqlite3.connect(db_path)
    young_rows = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT player_id) FROM season_stats "
        "WHERE age_during_season IS NOT NULL AND age_during_season < ?", (MIN_PRO_AGE,)).fetchone()
    welded = [r[0] for r in con.execute(
        "SELECT o.player_id FROM career_outcomes o JOIN prospects p USING(player_id) "
        "WHERE o.mlb_debut_year IS NOT NULL AND p.birth_date IS NOT NULL "
        "AND o.mlb_debut_year < CAST(substr(p.birth_date, 1, 4) AS INTEGER) + ?", (MIN_MLB_AGE,))]
    mlb_rows = 0
    if welded:
        q = ",".join("?" * len(welded))
        mlb_rows = con.execute(f"SELECT COUNT(*) FROM season_stats WHERE level = 'MLB' "
                               f"AND player_id IN ({q})", welded).fetchone()[0]
    stats = {"young_season_rows": young_rows[0], "young_rows_players": young_rows[1],
             "welded_mlb_careers": len(welded), "welded_mlb_rows": mlb_rows}
    if verbose:
        print(f"[ages] season rows before age {MIN_PRO_AGE}: {young_rows[0]:,} "
              f"({young_rows[1]:,} players)")
        print(f"[ages] MLB careers starting before age {MIN_MLB_AGE} (not the record's own): "
              f"{len(welded):,} players, {mlb_rows:,} MLB season rows")
    if dry_run:
        print("[ages] DRY RUN — nothing written")
        con.close()
        return stats
    with con:
        con.execute("DELETE FROM season_stats WHERE age_during_season IS NOT NULL "
                    "AND age_during_season < ?", (MIN_PRO_AGE,))
        for pid in welded:
            con.execute("DELETE FROM season_stats WHERE player_id = ? AND level = 'MLB'", (pid,))
            sets = ", ".join(f"{c} = NULL" for c in MLB_OUTCOME_COLS)
            con.execute(f"UPDATE career_outcomes SET {sets} WHERE player_id = ?", (pid,))
            ej = con.execute("SELECT events_json FROM career_outcomes WHERE player_id = ?",
                             (pid,)).fetchone()[0]
            if ej:
                d = json.loads(ej)
                for k in MLB_EVENT_KEYS & set(d):
                    d[k] = False
                con.execute("UPDATE career_outcomes SET events_json = ? WHERE player_id = ?",
                            (json.dumps(d), pid))
    if verbose:
        print(f"[ages] repaired {len(welded):,} welded careers; "
              f"deleted {young_rows[0]:,} + {mlb_rows:,} season rows")
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
