"""
prospects/ingestion/backfills/ifa_bio_backfill.py
=========================================

After a permissive MiLB pull creates IFA stub prospects, this script fetches
biographical info (birth_date, birth country, position, height/weight, etc.)
for them in bulk via MLB Stats API's batch /v1/people?personIds=... endpoint.

Usage:
    python -m prospects.ingestion.backfills.ifa_bio_backfill [--db prospects.db]
"""

from __future__ import annotations

import argparse
import re
import time
import unicodedata
from datetime import date
from typing import Optional

import requests

from prospects.storage import ProspectDB


UA = {"User-Agent": "Mozilla/5.0 (research)"}
BATCH = 100  # MLB Stats API supports up to ~100 ids in one call


def _fetch_people(ids: list[str]) -> dict[str, dict]:
    """Return {mlbam_id: person_dict} for a batch of MLBAM ids."""
    if not ids:
        return {}
    url = "https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(ids)
    r = requests.get(url, headers=UA, timeout=60)
    if r.status_code != 200:
        return {}
    out = {}
    for p in r.json().get("people", []):
        out[str(p.get("id"))] = p
    return out


def run(db_path: str = "prospects.db", verbose: bool = True) -> None:
    db = ProspectDB(db_path)
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT player_id, mlbam_id FROM prospects
            WHERE is_international = 1
              AND birth_date IS NULL
              AND mlbam_id IS NOT NULL
        """).fetchall()
    targets = [(r["player_id"], r["mlbam_id"]) for r in rows]
    if verbose:
        print(f"[ifa-bio] {len(targets):,} IFAs missing birth_date")
    if not targets:
        return

    pid_by_mlbam = {m: pid for pid, m in targets}
    ids = list(pid_by_mlbam)

    bio_updates: list[tuple] = []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        if verbose:
            print(f"  batch {i//BATCH + 1}/{(len(ids)+BATCH-1)//BATCH}: "
                  f"fetching {len(chunk)} ids")
        people = _fetch_people(chunk)
        for mlbam, person in people.items():
            pid = pid_by_mlbam.get(mlbam)
            if not pid:
                continue
            bio_updates.append((
                person.get("birthDate"),
                person.get("primaryPosition", {}).get("abbreviation"),
                person.get("birthCountry"),
                person.get("nameFirstLast") or person.get("fullName") or "",
                pid,
            ))
        time.sleep(0.15)  # be nice

    if verbose:
        print(f"[ifa-bio] applying {len(bio_updates):,} bio updates")
    with db._connect() as conn:
        for birth, pos, country, name, pid in bio_updates:
            conn.execute("""
                UPDATE prospects
                SET birth_date = COALESCE(?, birth_date),
                    primary_position = CASE WHEN primary_position = 'UNK' AND ? IS NOT NULL
                                            THEN ? ELSE primary_position END,
                    origin = CASE WHEN (origin IS NULL OR origin = '')
                                       AND ? IS NOT NULL
                                       THEN ? ELSE origin END,
                    name = CASE WHEN (name LIKE 'mlbam_%' OR name IS NULL OR name = '')
                                     AND ? <> '' THEN ? ELSE name END
                WHERE player_id = ?
            """, (birth, pos, pos, country, country, name, name, pid))

    # Now also update age_during_season for season_stats rows belonging to
    # these IFAs (only those with birth_date now set).
    if verbose:
        print("[ifa-bio] recomputing age_during_season for IFA rows...")
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT s.rowid AS rid, s.season_year, p.birth_date
            FROM season_stats s
            JOIN prospects p ON p.player_id = s.player_id
            WHERE p.is_international = 1
              AND p.birth_date IS NOT NULL
              AND s.age_during_season IS NULL
        """).fetchall()
        age_updates = []
        for r in rows:
            try:
                y, m, d = (int(x) for x in r["birth_date"].split("-"))
                bd = date(y, m, d)
                cutoff = date(r["season_year"], 7, 1)
                age = round((cutoff - bd).days / 365.25, 2)
                age_updates.append((age, r["rid"]))
            except Exception:
                continue
        conn.executemany(
            "UPDATE season_stats SET age_during_season = ? WHERE rowid = ?",
            age_updates,
        )
    if verbose:
        print(f"[ifa-bio] updated age on {len(age_updates):,} IFA season rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    args = parser.parse_args()
    run(args.db)
