"""
prospects/ingestion/backfills/birthdate_backfill.py
===========================================

Backfill `prospects.birth_date` from the MLB Stats API draft endpoint, then
derive `season_stats.age_during_season` for every existing row.

Strategy:
  1. For each draft year present in DB, call /v1/draft/{year} once. Match each
     API pick to our prospects table by (year, round, pick). If that fails,
     fall back to normalized name match within that year.
  2. Write `birth_date` (ISO 8601 YYYY-MM-DD) and `mlbam_id` to the prospect row.
  3. Compute age_during_season = season_year - birth_year, adjusted by -1 if
     birthday is after July 1 (the standard baseball-age convention).

Usage:
    python -m prospects.ingestion.backfills.birthdate_backfill [--db prospects.db]
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


def _norm(s) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s).lower()).strip()


def _split(full: str) -> tuple[str, str]:
    parts = _norm(full).split()
    sfx = {"jr", "sr", "ii", "iii", "iv"}
    while parts and parts[-1] in sfx:
        parts.pop()
    if len(parts) <= 1:
        return ("", parts[0] if parts else "")
    return parts[0], parts[-1]


def _fetch_draft(year: int) -> list[dict]:
    r = requests.get(f"https://statsapi.mlb.com/api/v1/draft/{year}",
                     headers=UA, timeout=60)
    if r.status_code != 200:
        return []
    picks = []
    for rnd in r.json().get("drafts", {}).get("rounds", []):
        for p in rnd.get("picks", []):
            picks.append(p)
    return picks


def _baseball_age(birth_iso: str, season_year: int) -> Optional[float]:
    """Age as of June 30 of season_year (standard baseball-age convention)."""
    try:
        y, m, d = (int(x) for x in birth_iso.split("-"))
    except Exception:
        return None
    cutoff = date(season_year, 7, 1)
    bd = date(y, m, d)
    return round((cutoff - bd).days / 365.25, 2)


def run(db_path: str = "prospects.db", verbose: bool = True) -> None:
    db = ProspectDB(db_path)

    with db._connect() as conn:
        years = [r["y"] for r in conn.execute(
            "SELECT DISTINCT draft_year AS y FROM prospects "
            "WHERE draft_year IS NOT NULL ORDER BY y"
        ).fetchall()]
        # Index prospects by (year, round, pick) and (year, normalized_name)
        rows = conn.execute(
            "SELECT player_id, name, draft_year, draft_round, draft_pick FROM prospects"
        ).fetchall()
    by_yrp: dict[tuple, str] = {}
    by_yn: dict[tuple, list] = {}
    for r in rows:
        key = (r["draft_year"], r["draft_round"], r["draft_pick"])
        if all(k is not None for k in key):
            by_yrp[key] = r["player_id"]
        first, last = _split(r["name"])
        by_yn.setdefault((r["draft_year"], last, first), []).append(r["player_id"])

    if verbose:
        print(f"[birth] backfilling {len(rows):,} prospects across "
              f"{len(years)} draft years")

    n_birthdate = 0
    n_mlbam = 0
    n_missed = 0
    updates: list[tuple] = []
    for year in years:
        picks = _fetch_draft(year)
        if not picks:
            if verbose:
                print(f"  {year}: no picks from API")
            continue
        matched = 0
        for p in picks:
            person = p.get("person") or {}
            birth = person.get("birthDate")
            mlbam = person.get("id")
            try:
                pickRound = int(str(p.get("pickRound") or 0)) or None
            except (ValueError, TypeError):
                pickRound = None
            pickNum = p.get("pickNumber")
            pid = by_yrp.get((year, pickRound, pickNum))
            if pid is None:
                # name fallback
                full = person.get("fullName") or p.get("name") or ""
                first, last = _split(full)
                cands = by_yn.get((year, last, first), [])
                if len(cands) == 1:
                    pid = cands[0]
            if pid is None:
                continue
            matched += 1
            updates.append((birth, str(mlbam) if mlbam else None, pid))
            if birth:
                n_birthdate += 1
            if mlbam:
                n_mlbam += 1
        if verbose:
            print(f"  {year}: matched {matched}/{len(picks)} API picks")
        n_missed += len(picks) - matched
        time.sleep(0.1)

    # Apply updates
    with db._connect() as conn:
        conn.executemany(
            "UPDATE prospects "
            "SET birth_date = COALESCE(?, birth_date), "
            "    mlbam_id   = COALESCE(?, mlbam_id) "
            "WHERE player_id = ?",
            updates,
        )

    if verbose:
        print(f"\n[birth] wrote {len(updates):,} prospect updates")
        print(f"        birth_date set: {n_birthdate:,}")
        print(f"        mlbam_id set:   {n_mlbam:,}")
        print(f"        unmatched API picks (across all years): {n_missed:,}")

    # Now derive age_during_season for every season_stats row that joins to a
    # prospect with a birth_date.
    if verbose:
        print("\n[birth] computing age_during_season for all season_stats...")
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT s.rowid AS rid, s.season_year, p.birth_date
            FROM season_stats s
            JOIN prospects p ON p.player_id = s.player_id
            WHERE p.birth_date IS NOT NULL
        """).fetchall()
        age_updates = []
        for r in rows:
            age = _baseball_age(r["birth_date"], r["season_year"])
            if age is not None:
                age_updates.append((age, r["rid"]))
        conn.executemany(
            "UPDATE season_stats SET age_during_season = ? WHERE rowid = ?",
            age_updates,
        )
    if verbose:
        print(f"[birth] updated age on {len(age_updates):,} season_stats rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    args = parser.parse_args()
    run(args.db)
