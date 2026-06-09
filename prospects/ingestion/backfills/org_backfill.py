"""Backfill correct, trade-aware orgs from the MLB Stats API.

1. Pull every MiLB team per (season, sportId) -> (season, affiliate-abbrev) ->
   parentOrgName. Cached to scratch/affiliate_org_map.csv.
2. Update prospects.current_org = parentOrg of each player's LATEST season's
   affiliate (trade-aware: the most recent season reflects their current org,
   so a traded player gets the new org, not the draft org).

Usage: python -m prospects.ingestion.backfills.org_backfill --db prospects_snapshot.db
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

SPORTS = [1, 11, 12, 13, 14, 15, 16, 5442, 17]   # MLB + AAA..rookie + DSL etc.
REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "scratch" / "affiliate_org_map.csv"


def build_affiliate_map(seasons) -> dict:
    rows = []
    for season in seasons:
        for sid in SPORTS:
            try:
                r = requests.get("https://statsapi.mlb.com/api/v1/teams",
                                 params={"sportId": sid, "season": season},
                                 timeout=20)
                teams = r.json().get("teams", [])
            except Exception:
                continue
            for t in teams:
                po = t.get("parentOrgName")
                if not po:
                    continue
                rows.append({"season": season, "abbrev": t.get("abbreviation"),
                             "name": t.get("name"), "parent_org": po})
            time.sleep(0.03)
    df = pd.DataFrame(rows).drop_duplicates()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"affiliate map: {len(df):,} (season,team) rows -> {CACHE.name}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    args = ap.parse_args()

    c = sqlite3.connect(args.db)
    seasons = [r[0] for r in c.execute(
        "SELECT DISTINCT season_year FROM season_stats WHERE season_year IS NOT NULL")]
    seasons = sorted(s for s in seasons if s)
    amap_df = build_affiliate_map(seasons)
    # (season, abbrev) -> parent, plus suffix fallback for 'A-RED' style codes
    by_sa = {(r.season, r.abbrev): r.parent_org
             for r in amap_df.itertuples() if pd.notna(r.abbrev)}
    by_suffix = {(r.season, r.abbrev.split("-")[-1]): r.parent_org
                 for r in amap_df.itertuples() if pd.notna(r.abbrev)}

    def lookup(season, org):
        if (season, org) in by_sa:
            return by_sa[(season, org)]
        return by_suffix.get((season, org.split("-")[-1]))  # strip level prefix

    # latest season affiliate per player
    latest: dict[str, tuple] = {}
    for pid, yr, org in c.execute(
            "SELECT player_id, season_year, org FROM season_stats "
            "WHERE org IS NOT NULL AND season_year IS NOT NULL"):
        if pid not in latest or yr > latest[pid][0]:
            latest[pid] = (yr, org)

    updates, unmapped = [], 0
    for pid, (yr, org) in latest.items():
        po = lookup(yr, org)
        if po:
            updates.append((po, pid))
        else:
            unmapped += 1
    c.executemany("UPDATE prospects SET current_org=? WHERE player_id=?", updates)
    c.commit()

    # coverage report
    tot = c.execute("SELECT COUNT(*) FROM prospects").fetchone()[0]
    have = c.execute("SELECT COUNT(*) FROM prospects WHERE current_org IS NOT NULL").fetchone()[0]
    c.close()
    print(f"current_org updated: {len(updates):,} players "
          f"({unmapped:,} latest-affiliate unmapped)")
    print(f"prospects with current_org now: {have:,}/{tot:,}")


if __name__ == "__main__":
    main()
