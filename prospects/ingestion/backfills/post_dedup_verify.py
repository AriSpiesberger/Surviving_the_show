"""Post-rebuild verification: confirm name-collision pollution is gone.

Runs after the MiLB re-pull and career_outcomes rebuild complete.
Reports:
  - season_stats strict-dup count (should be ~0, down from 20K)
  - career_outcomes with implausible debut years (should be 0, down from 315)
  - HOF=1 on modern prospects (should be 0, down from 15)
  - Total row counts for sanity

Usage:
    python -m prospects.ingestion.backfills.post_dedup_verify [--db prospects.db]
"""
from __future__ import annotations

import argparse
import sqlite3


def verify(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    print(f"\n=== {db_path} ===")
    cur.execute("SELECT COUNT(*) FROM prospects")
    print(f"  prospects:       {cur.fetchone()[0]:>8,}")
    cur.execute("SELECT COUNT(*) FROM season_stats")
    print(f"  season_stats:    {cur.fetchone()[0]:>8,}")
    cur.execute("SELECT COUNT(*) FROM career_outcomes")
    print(f"  career_outcomes: {cur.fetchone()[0]:>8,}")
    cur.execute("SELECT COUNT(*) FROM season_stats "
                "WHERE season_year >= 2005 AND level != 'MLB'")
    print(f"  MiLB 2005+:      {cur.fetchone()[0]:>8,}")

    # Hitter strict-dup
    cur.execute("""
    SELECT COUNT(*), COALESCE(SUM(n_pids), 0) FROM (
      SELECT season_year, level, org, pa, avg, obp, slg, k_pct,
             COUNT(DISTINCT player_id) n_pids
      FROM season_stats
      WHERE pa > 20 AND season_year >= 2005 AND level != 'MLB'
      GROUP BY season_year, level, org, pa, avg, obp, slg, k_pct
      HAVING n_pids > 1
    )""")
    h_keys, h_rows = cur.fetchone()
    print(f"  hitter strict-dup keys / polluted rows: {h_keys:,} / {h_rows:,}")

    cur.execute("""
    SELECT COUNT(*), COALESCE(SUM(n_pids), 0) FROM (
      SELECT season_year, level, org, ip, era, k9, bb9,
             COUNT(DISTINCT player_id) n_pids
      FROM season_stats
      WHERE ip > 5 AND season_year >= 2005 AND level != 'MLB'
      GROUP BY season_year, level, org, ip, era, k9, bb9
      HAVING n_pids > 1
    )""")
    p_keys, p_rows = cur.fetchone()
    print(f"  pitcher strict-dup keys / polluted rows: {p_keys:,} / {p_rows:,}")

    cur.execute("""
    SELECT COUNT(*) FROM career_outcomes o JOIN prospects p
      ON p.player_id = o.player_id
    WHERE o.mlb_debut_year IS NOT NULL AND p.draft_year IS NOT NULL
      AND o.mlb_debut_year < p.draft_year - 1
    """)
    print(f"  outcomes with debut < draft-1: {cur.fetchone()[0]}")

    cur.execute("""
    SELECT COUNT(*) FROM career_outcomes o JOIN prospects p
      ON p.player_id = o.player_id
    WHERE o.is_hof_inducted = 1
      AND (p.draft_year IS NULL OR p.draft_year >= 2000)
    """)
    print(f"  modern prospects flagged HOF=1: {cur.fetchone()[0]}")

    cur.execute(
        "SELECT mlbam_id, COUNT(*) c FROM prospects "
        "WHERE mlbam_id IS NOT NULL AND mlbam_id NOT IN ('', '-1') "
        "GROUP BY mlbam_id HAVING c > 1 LIMIT 1"
    )
    res = cur.fetchall()
    print(f"  remaining duplicate mlbam_ids: {len(res)}")

    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="+",
                    default=["prospects.db", "prospects_snapshot.db"])
    args = ap.parse_args()
    for d in args.db:
        verify(d)


if __name__ == "__main__":
    main()
