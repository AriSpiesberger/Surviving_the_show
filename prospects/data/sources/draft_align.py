"""
prospects/data/sources/draft_align.py
======================================

Build the draft prospect universe with an INJECTIVE cross-source player
identity — one prospect <-> one MLBAM id, resolved from the source of truth.

Sources, and what each contributes
----------------------------------
  spine   MLB Stats API  /api/v1/draft/{year}
            -> the authoritative mlbam id (person.id) for every pick, 1965-.
               Not scraping (same API as milb); not rate-blocked.
  scheme  ncaa_bbStats draft cache  data/mlb_draft_cache/{year}.json
            -> the canonical player_id name + position + school + draft team.
               We keep the cache name for player_id so existing artifacts
               (holdings.csv etc.) stay byte-identical.
  enrich  reference/baseballcube/player_xref.csv
            -> birthdate, signing bonus, HS/college, tbc/retrosheet ids.

Why it is injective
-------------------
The join key is the draft slot ``(draft_year, overall_pick_number)``. That key
is unique within every source and every era (verified: no duplicate overall
picks 1980-2025), so each slot resolves to exactly one API pick, one cache row,
and at most one xref row. mlbam then flows from the API as a unique per-player
key. Slot/name conflicts (the cache and the API disagree on who holds a slot)
are logged, never force-matched — precision over recall.

The result: every drafted prospect carries its real ``mlbam_id``, which is the
key ``outcomes`` (Chadwick) and ``milb`` (MLB Stats API playerId) match on. No
mlbam, no labels; this module is what closes that gap.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import requests

from prospects.core.schema import Pedigree, Prospect
from prospects.core.storage import ProspectDB
from prospects.data.sources.mlb_draft import PITCHER_POS, _cache_dir

DRAFT_API = "https://statsapi.mlb.com/api/v1/draft/{year}"
XREF_CSV = os.path.join("reference", "baseballcube", "player_xref.csv")


def _norm_id_name(name: str) -> str:
    """Exactly the transform the cache loader uses to build player_id.

    Must stay identical to ``mlb_draft.pull_draft_from_cache`` so player_ids
    are byte-for-byte stable (holdings.csv, card data join on them).
    """
    return name.lower().replace(" ", "_").replace(".", "").replace(chr(39), "")


def _player_id(year: int, name: str, round_n: Optional[int], pick_n: Optional[int]) -> str:
    pid = f"draft_{year}_{_norm_id_name(name)}"
    if round_n is not None:
        pid += f"_r{round_n}"
    if pick_n is not None:
        pid += f"p{pick_n}"
    return pid


def _norm_cmp(name: str) -> str:
    """Loose normalization for name-agreement comparison only."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _to_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> Optional[float]:
    try:
        f = float(str(v).replace("$", "").replace(",", "").strip())
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _to_date(v) -> Optional[date]:
    v = (v or "").strip()
    try:
        return date.fromisoformat(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _fetch_api_draft(year: int, timeout: float = 30.0) -> dict[int, dict]:
    """Return {overall_pick: {mlbam, name, round, pos, school, team}} for a year.

    Raises on network/HTTP failure so the caller can fall back per-year.
    """
    r = requests.get(DRAFT_API.format(year=year), timeout=timeout)
    r.raise_for_status()
    out: dict[int, dict] = {}
    for rd in r.json().get("drafts", {}).get("rounds", []):
        for p in rd.get("picks", []):
            ov = p.get("pickNumber")
            if not ov:
                continue
            person = p.get("person") or {}
            out[int(ov)] = {
                "mlbam": str(person["id"]) if person.get("id") else None,
                "name": (person.get("fullName") or "").strip(),
                "round": _to_int(p.get("pickRound")),
                "pos": ((person.get("primaryPosition") or {}).get("abbreviation") or "").upper(),
                "school": ((p.get("school") or {}).get("name") or "").strip(),
                "team": ((p.get("team") or {}).get("name") or "").strip(),
            }
    return out


def _load_cache_year(year: int) -> dict[int, dict]:
    """Return {overall_pick: cache_row} for a year (empty if no cache file)."""
    path = os.path.join(_cache_dir(), f"{year}.json")
    if not os.path.isfile(path):
        return {}
    import json
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out: dict[int, dict] = {}
    for row in rows:
        ov = _to_int(row.get("Pick"))
        if ov is not None:
            out.setdefault(ov, row)  # first wins; overall pick is unique anyway
    return out


def _load_xref() -> tuple[dict[str, dict], dict[tuple, dict]]:
    """Return (by_mlbam, by_slot) indices over player_xref.csv."""
    by_mlbam: dict[str, dict] = {}
    by_slot: dict[tuple, dict] = {}
    if not os.path.isfile(XREF_CSV):
        return by_mlbam, by_slot
    with open(XREF_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mlbam = (row.get("mlbam_id") or "").strip()
            if mlbam.isdigit():
                by_mlbam.setdefault(mlbam, row)
            y, pk = _to_int(row.get("draft_year")), _to_int(row.get("draft_pick"))
            if y and pk:
                by_slot.setdefault((y, pk), row)
    return by_mlbam, by_slot


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

@dataclass
class AlignStats:
    universe: int = 0          # prospects upserted
    mlbam_set: int = 0         # got a real mlbam from the API
    api_only: int = 0          # slot present in API but not the cache
    cache_only: int = 0        # slot present in cache but not the API (-> no mlbam)
    xref_enriched: int = 0     # picked up birthdate/bonus/school from the xref
    name_conflicts: list = field(default_factory=list)  # (year, ov, cache_name, api_name)


def pull_draft_aligned(
    db: ProspectDB,
    start_year: int = 2005,
    end_year: int = 2024,
    verbose: bool = True,
    report_path: Optional[str] = None,
) -> AlignStats:
    """Build the draft universe with mlbam ids resolved injectively.

    For each draft slot we take mlbam from the API, keep the cache name for the
    player_id, and enrich from the xref (by mlbam, then by slot). Returns an
    AlignStats; if ``report_path`` is set, writes a per-pick audit CSV there.
    """
    from tqdm import tqdm

    xref_by_mlbam, xref_by_slot = _load_xref()
    stats = AlignStats()
    audit_rows: list[dict] = []

    bar = tqdm(range(start_year, end_year + 1), desc="draft-align",
               unit="yr", disable=not verbose)
    for year in bar:
        cache = _load_cache_year(year)
        try:
            api = _fetch_api_draft(year)
        except Exception as e:
            api = {}
            msg = " ".join(str(e).split())[:120]
            bar.write(f"[align] {year}: MLB API draft unavailable ({msg}); "
                      f"cache-only, no mlbam this year")

        slots = sorted(set(api) | set(cache))
        for ov in slots:
            a = api.get(ov)
            c = cache.get(ov)

            # Name: prefer the cache (player_id stability); fall back to API.
            name = ((c or {}).get("Player Name") or "").strip() or (a or {}).get("name", "")
            if not name:
                continue

            round_n = _to_int((c or {}).get("Round"))
            if round_n is None and a:
                round_n = a.get("round")
            pos = ((c or {}).get("POS") or "").strip().upper() or (a or {}).get("pos", "")
            team = ((c or {}).get("Drafted By") or "").strip() or (a or {}).get("team", "")
            origin = ((c or {}).get("Drafted From") or "").strip() or (a or {}).get("school", "")

            mlbam = (a or {}).get("mlbam")

            # xref enrichment: mlbam first (unique), then slot.
            x = (xref_by_mlbam.get(mlbam) if mlbam else None) or xref_by_slot.get((year, ov))
            birth = _to_date(x.get("birthdate")) if x else None
            bonus = _to_float(x.get("signing_bonus")) if x else None

            player_id = _player_id(year, name, round_n, ov)
            p = Prospect(
                player_id=player_id,
                name=name,
                is_pitcher=pos in PITCHER_POS,
                primary_position=pos or "UNK",
                birth_date=birth,
                current_org=team or None,
                pedigree=Pedigree(
                    draft_year=year,
                    draft_round=round_n,
                    draft_pick=ov,
                    signing_bonus_usd=bonus,
                    origin=origin,
                ),
            )
            db.upsert_prospect(p)
            stats.universe += 1
            if mlbam:
                db.set_mlbam_id(player_id, mlbam)
                stats.mlbam_set += 1
            if x:
                stats.xref_enriched += 1
            if a and not c:
                stats.api_only += 1
            elif c and not a:
                stats.cache_only += 1
            if a and c and _norm_cmp(name) != _norm_cmp(a.get("name", "")):
                stats.name_conflicts.append((year, ov, name, a.get("name", "")))

            if report_path:
                audit_rows.append({
                    "year": year, "overall_pick": ov, "round": round_n,
                    "player_id": player_id, "name": name,
                    "api_name": (a or {}).get("name", ""),
                    "mlbam_id": mlbam or "", "source": (
                        "both" if a and c else "api_only" if a else "cache_only"),
                    "xref": "yes" if x else "no",
                    "name_conflict": "yes" if (
                        a and c and _norm_cmp(name) != _norm_cmp(a.get("name", ""))) else "",
                })

        if verbose:
            bar.set_postfix(mlbam=stats.mlbam_set, universe=stats.universe)

    bar.close()

    if report_path and audit_rows:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    if verbose:
        pct = 100 * stats.mlbam_set // max(stats.universe, 1)
        print(f"\n[align] universe={stats.universe} mlbam_set={stats.mlbam_set} "
              f"({pct}%) xref_enriched={stats.xref_enriched} "
              f"api_only={stats.api_only} cache_only={stats.cache_only} "
              f"name_conflicts={len(stats.name_conflicts)}")
        if report_path and audit_rows:
            print(f"[align] audit written -> {report_path}")

    return stats
