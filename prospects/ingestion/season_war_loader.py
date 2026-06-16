"""
prospects/ingestion/season_war_loader.py
==========================================

Populate a per-season WAR table (`season_war`) by pulling Baseball-Reference
WAR from pybaseball (bwar_bat + bwar_pitch) and joining to our prospects by
(normalized name, debut year) -- the SAME matching strategy as bwar_loader,
but keeping the season-by-season detail that bwar_loader throws away.

Why this exists: the forward-WAR regressor needs to know, for any landmark
year S, how much WAR a player accrued in the seasons *after* S. That requires
per-season WAR, not just career totals.

Table schema:
    season_war(player_id TEXT, season_year INT, war REAL, bbref_id TEXT,
               PRIMARY KEY(player_id, season_year))

Two-way players (a year with both batting and pitching WAR) get the sum.

Usage:
    python -m prospects.ingestion.season_war_loader [--db prospects_snapshot.db]
"""

from __future__ import annotations

import argparse
import sqlite3
import time

import pandas as pd

# Reuse the battle-tested normalization + debut-window matching.
from prospects.ingestion.bwar_loader import (
    _norm,
    build_name_lookup,
    match_prospect,
)


def build_bwar_season_index() -> tuple[dict, dict]:
    """Pull bwar_bat + bwar_pitch.

    Returns
    -------
    season_by_bbref : {bbref_id: {year_int: war_float}}
        Per-season WAR (bat + pitch summed within a year).
    career_idx : {bbref_id: {'name', 'debut', 'war'}}
        Career aggregate used by match_prospect for disambiguation.
    """
    import pybaseball as pb

    print("[season-war] downloading bwar_bat...")
    t0 = time.time()
    bat = pb.bwar_bat(return_all=True)
    print(f"  {len(bat):,} batting rows in {time.time()-t0:.1f}s")

    print("[season-war] downloading bwar_pitch...")
    t0 = time.time()
    pit = pb.bwar_pitch(return_all=True)
    print(f"  {len(pit):,} pitching rows in {time.time()-t0:.1f}s")

    keep_bat = bat[["player_ID", "name_common", "year_ID", "WAR"]].copy()
    keep_pit = pit[["player_ID", "name_common", "year_ID", "WAR"]].copy()
    combined = pd.concat([keep_bat, keep_pit], ignore_index=True)
    combined["WAR"] = pd.to_numeric(combined["WAR"], errors="coerce").fillna(0.0)
    combined["year_ID"] = pd.to_numeric(combined["year_ID"], errors="coerce")
    combined = combined.dropna(subset=["player_ID", "year_ID"])

    # Per (player, year): sum WAR across bat+pitch stints (two-way + traded).
    per_season = (combined.groupby(["player_ID", "year_ID"])["WAR"]
                  .sum().reset_index())

    season_by_bbref: dict[str, dict[int, float]] = {}
    for r in per_season.itertuples(index=False):
        season_by_bbref.setdefault(r.player_ID, {})[int(r.year_ID)] = float(r.WAR)

    # Career aggregate for matching disambiguation.
    name_by_bbref = combined.groupby("player_ID")["name_common"].first()
    career_idx: dict[str, dict] = {}
    for bbref_id, seasons in season_by_bbref.items():
        career_idx[bbref_id] = {
            "name": str(name_by_bbref.get(bbref_id, "")),
            "debut": int(min(seasons.keys())),
            "war": float(sum(seasons.values())),
        }
    print(f"[season-war] {len(season_by_bbref):,} unique bbref players, "
          f"{len(per_season):,} player-seasons")
    return season_by_bbref, career_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects_snapshot.db")
    args = parser.parse_args()

    print(f"[season-war] DB: {args.db}")
    season_by_bbref, career_idx = build_bwar_season_index()
    lookup = build_name_lookup(career_idx)
    print(f"[season-war] {len(lookup):,} unique normalized names in lookup")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS season_war (
            player_id   TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            war         REAL NOT NULL,
            bbref_id    TEXT,
            PRIMARY KEY (player_id, season_year)
        )
    """)
    conn.execute("DELETE FROM season_war")  # full rebuild each run

    rows = conn.execute("""
        SELECT p.player_id, p.name, o.mlb_debut_year
        FROM prospects p
        JOIN career_outcomes o ON p.player_id = o.player_id
        WHERE o.mlb_debut_year IS NOT NULL
    """).fetchall()
    print(f"[season-war] {len(rows):,} prospects with mlb_debut_year")

    matched = 0
    season_inserts: list[tuple] = []
    career_updates: list[tuple] = []
    for r in rows:
        war_total, bbref = match_prospect(r["name"], r["mlb_debut_year"], lookup)
        if bbref is None:
            continue
        matched += 1
        for yr, war in sorted(season_by_bbref.get(bbref, {}).items()):
            season_inserts.append((r["player_id"], int(yr), round(float(war), 3),
                                   bbref))
        career_updates.append((round(float(war_total), 2), r["player_id"]))

    conn.executemany(
        "INSERT OR REPLACE INTO season_war "
        "(player_id, season_year, war, bbref_id) VALUES (?, ?, ?, ?)",
        season_inserts,
    )
    # Also refresh career_outcomes.career_war for consistency (it was 0'd out).
    conn.executemany(
        "UPDATE career_outcomes SET career_war = ? WHERE player_id = ?",
        career_updates,
    )
    conn.commit()

    n_players = conn.execute(
        "SELECT COUNT(DISTINCT player_id) FROM season_war").fetchone()[0]
    yr_lo, yr_hi = conn.execute(
        "SELECT MIN(season_year), MAX(season_year) FROM season_war").fetchone()
    conn.close()

    print(f"[season-war] matched {matched:,}/{len(rows):,} "
          f"({matched/max(len(rows),1):.1%})")
    print(f"[season-war] inserted {len(season_inserts):,} player-seasons "
          f"for {n_players:,} players  (years {yr_lo}..{yr_hi})")


if __name__ == "__main__":
    main()
