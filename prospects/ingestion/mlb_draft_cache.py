"""
prospects/ingestion/mlb_draft_cache.py
========================================

Draft loader that reads the JSON cache shipped with ncaa_bbStats
(`data/mlb_draft_cache/{year}.json`). Covers 1965-2025 with clean schema:
    Round, Pick, Phase, Player Name, Drafted By, POS, Drafted From, Year

We prefer this over pybaseball.amateur_draft because baseball-reference now
returns 403 to pybaseball's User-Agent, making 2018+ years unreachable.

The cache lacks signing bonus and birth date — those stay None and the
classifier's missingness flags handle it.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from prospects.schema import Pedigree, Prospect
from prospects.storage import ProspectDB


PITCHER_POS = {"P", "RHP", "LHP", "SP", "RP"}


def _cache_dir() -> str:
    """Locate the ncaa_bbStats mlb_draft_cache directory."""
    try:
        import ncaa_bbStats
        pkg_root = os.path.dirname(ncaa_bbStats.__file__)
        candidate = os.path.abspath(os.path.join(pkg_root, "..", "data", "mlb_draft_cache"))
        if os.path.isdir(candidate):
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError("Could not locate mlb_draft_cache from ncaa_bbStats install")


def _origin_looks_collegiate(s: str) -> bool:
    s = (s or "").lower()
    return any(tok in s for tok in (
        "university", "college", "state", "u of",
        "tech", "institute", "polytechnic",
    ))


def _origin_looks_high_school(s: str) -> bool:
    s = (s or "").lower()
    return "hs" in s or "high school" in s or "high sch" in s or "academy" in s


def pull_draft_from_cache(
    db: ProspectDB,
    start_year: int = 2005,
    end_year: int = 2024,
    verbose: bool = True,
) -> int:
    """
    Load draft picks from the JSON cache into the prospects table.

    Returns:
        Number of upserts performed.
    """
    cache_dir = _cache_dir()
    total = 0

    for year in range(start_year, end_year + 1):
        path = os.path.join(cache_dir, f"{year}.json")
        if not os.path.isfile(path):
            if verbose:
                print(f"[draft] {year}: no cache file at {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            picks = json.load(f)

        if verbose:
            print(f"[draft] {year}: {len(picks)} picks")

        for row in picks:
            name = (row.get("Player Name") or "").strip()
            if not name:
                continue

            pos = (row.get("POS") or "").strip().upper()
            round_str = row.get("Round")
            pick_str = row.get("Pick")
            origin = (row.get("Drafted From") or "").strip()
            team = (row.get("Drafted By") or "").strip()

            try:
                round_n: Optional[int] = int(round_str)
            except (TypeError, ValueError):
                round_n = None
            try:
                pick_n: Optional[int] = int(pick_str)
            except (TypeError, ValueError):
                pick_n = None

            player_id = f"draft_{year}_{name.lower().replace(' ', '_').replace('.', '').replace(chr(39), '')}"
            if round_n is not None:
                player_id += f"_r{round_n}"
            if pick_n is not None:
                player_id += f"p{pick_n}"

            p = Prospect(
                player_id=player_id,
                name=name,
                is_pitcher=pos in PITCHER_POS,
                primary_position=pos or "UNK",
                current_org=team or None,
                pedigree=Pedigree(
                    draft_year=year,
                    draft_round=round_n,
                    draft_pick=pick_n,
                    origin=origin,
                ),
            )
            db.upsert_prospect(p)
            total += 1

    if verbose:
        print(f"[draft] cache loader: {total} picks upserted, "
              f"{db.count_prospects()} total prospects in DB")
    return total
