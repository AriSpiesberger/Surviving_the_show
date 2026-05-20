"""
prospects/ingestion/pybaseball_loader.py
==========================================

Pulls data via pybaseball:
- MLB career stats from FanGraphs (for training labels - outcomes)
- Draft data 1965-present
- Lahman historical (All-Star selections, awards, HOF)
- Player ID lookup table (Chadwick register)

Install:
    pip install pybaseball

Usage:
    from prospects.ingestion.pybaseball_loader import (
        pull_mlb_outcomes,
        pull_draft_data,
        pull_allstar_awards,
    )
    pull_mlb_outcomes(db, start_year=2000, end_year=2024)
    pull_draft_data(db, start_year=2005, end_year=2024)
    pull_allstar_awards(db)

NOTE: pybaseball functions and column names change across versions. This
module is written for pybaseball >= 2.2.7 (as of May 2026). If columns are
different in your version, the column-name handling will need adjustment.
The verbose=True flag prints columns found, useful for debugging.
"""

from __future__ import annotations

import time
from typing import Optional

from prospects.schema import CareerOutcome
from prospects.outcome_labels import label_career
from prospects.storage import ProspectDB


# ============================================================================
# LAHMAN ZIP SOURCING
# ============================================================================
# As of 2026 the canonical chadwickbureau/baseballdatabank repo was removed
# from GitHub; pybaseball's hardcoded URL 404s. The cbwinslow fork carries the
# same data with the same `baseballdatabank-master/` layout, so pybaseball's
# internal `base_string` still works once we point `url` at the fork.

LAHMAN_MIRROR_URL = "https://github.com/cbwinslow/baseballdatabank/archive/refs/heads/master.zip"
_LAHMAN_READY = False


def _ensure_lahman() -> bool:
    """Ensure Lahman data is downloaded and extracted. Idempotent."""
    global _LAHMAN_READY
    if _LAHMAN_READY:
        return True
    try:
        import pybaseball.lahman as L
        from os import path
        L.url = LAHMAN_MIRROR_URL
        extracted = path.join(L.cache.config.cache_directory, L.base_string)
        if not path.exists(extracted):
            L._handle = None  # force re-download
            L.download_lahman()
        _LAHMAN_READY = True
        return True
    except Exception as e:
        print(f"[lahman] download failed: {e}")
        return False


# ============================================================================
# UTILITY: SAFE COLUMN ACCESS
# ============================================================================

def get_col(row, *candidates, default=None):
    """
    Return the first matching column value from row. Tries candidates in order.

    pybaseball / FanGraphs / Baseball Reference column names vary by source
    and version. Examples:
        get_col(row, 'WAR', 'fWAR', 'bWAR', default=0.0)
        get_col(row, 'IDfg', 'fangraphs_id', 'idfg')
    """
    for cand in candidates:
        if cand in row:
            val = row[cand]
            if val is not None and (not isinstance(val, float) or val == val):  # NaN check
                return val
    return default


def to_str_id(val) -> str:
    """Convert player ID (often float or int from pandas) to canonical string."""
    if val is None:
        return ""
    try:
        return str(int(val))
    except (ValueError, TypeError):
        return str(val).strip()


# ============================================================================
# DRAFT DATA
# ============================================================================

def pull_draft_data(
    db: ProspectDB,
    start_year: int = 2005,
    end_year: int = 2024,
    verbose: bool = True,
) -> int:
    """
    Pull MLB amateur draft data, year by year, and save player_id -> pedigree
    info as Prospect records.

    pybaseball.amateur_draft(year, round_n) -> DataFrame
    Returns:
        Number of player records added/updated.
    """
    import pybaseball as pyb

    total = 0
    for year in range(start_year, end_year + 1):
        if verbose:
            print(f"[draft] Pulling {year} draft...")
        try:
            # Pull all rounds for the year. pyb.amateur_draft signature varies;
            # try both common signatures.
            df = None
            try:
                df = pyb.amateur_draft(year, 1, keep_stats=False)
                # Round 1 only at first, then expand
                # Better: iterate rounds
                rounds = [df]
                for rd in range(2, 21):
                    try:
                        rd_df = pyb.amateur_draft(year, rd, keep_stats=False)
                        if rd_df is not None and len(rd_df) > 0:
                            rounds.append(rd_df)
                    except Exception:
                        break
                import pandas as pd
                df = pd.concat(rounds, ignore_index=True)
            except TypeError:
                # Older pybaseball signature
                df = pyb.amateur_draft(year, 1)
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                continue

            if df is None or len(df) == 0:
                if verbose:
                    print(f"  No data for {year}")
                continue

            if verbose:
                print(f"  Found {len(df)} picks. Columns: {list(df.columns)[:10]}...")

            for _, row in df.iterrows():
                name = get_col(row, "Name", "Player", "name", default="")
                if not name:
                    continue

                # Determine player_id - prefer MLBAM, fallback to constructed
                player_id = to_str_id(
                    get_col(row, "key_mlbam", "mlbam_id", "IDmlb", "MLBAM_ID", default="")
                )
                if not player_id:
                    # Construct deterministic ID from name + year
                    player_id = f"draft_{year}_{name.replace(' ', '_').lower()}"

                pos = get_col(row, "Pos", "Position", "primary_position", default="")
                round_n = get_col(row, "Rnd", "Round", "draft_round", default=None)
                pick = get_col(row, "OvPck", "Overall", "Pick", "draft_pick", default=None)
                bonus = get_col(row, "Signing Bonus", "signing_bonus", "Bonus", default=None)

                # Parse signing bonus if it's a string like "$1,200,000"
                if isinstance(bonus, str):
                    bonus_clean = bonus.replace("$", "").replace(",", "").strip()
                    try:
                        bonus = float(bonus_clean) if bonus_clean else None
                    except ValueError:
                        bonus = None

                # Build minimal Prospect record (will be enriched by other loaders)
                from prospects.schema import Pedigree, Prospect
                p = Prospect(
                    player_id=player_id,
                    name=name,
                    is_pitcher=pos in {"P", "RHP", "LHP", "SP", "RP"},
                    primary_position=pos or "UNK",
                    pedigree=Pedigree(
                        draft_year=year,
                        draft_round=int(round_n) if round_n is not None else None,
                        draft_pick=int(pick) if pick is not None else None,
                        signing_bonus_usd=bonus,
                        origin=get_col(row, "From", "School", "origin", default=""),
                    ),
                )
                db.upsert_prospect(p)
                total += 1

            time.sleep(0.5)  # be nice to the source

        except Exception as e:
            if verbose:
                print(f"  ERROR on {year}: {type(e).__name__}: {e}")
            continue

    if verbose:
        print(f"[draft] Loaded {total} draft picks total.")
    return total


# ============================================================================
# CAREER OUTCOMES (MLB STATS + AWARDS)
# ============================================================================

def pull_mlb_outcomes(
    db: ProspectDB,
    only_player_ids: Optional[list[str]] = None,
    verbose: bool = True,
) -> int:
    """
    For each player in the prospects table, look up their MLB career stats
    + awards + HOF status. Build CareerOutcome records (training labels).

    Uses pybaseball.batting_stats / pitching_stats with stat_types='career'
    if available, otherwise aggregates across seasons.

    Args:
        only_player_ids: if given, only process these players (useful for testing)

    Returns:
        Number of outcome records created.
    """
    import pybaseball as pyb

    # Get list of player_ids to process
    prospects = db.all_prospects()
    if only_player_ids:
        prospects = [p for p in prospects if p["player_id"] in only_player_ids]

    if verbose:
        print(f"[outcomes] Processing {len(prospects)} prospects...")

    # Pull All-Star data once (it's in Lahman)
    allstar_counts = _load_allstar_counts(verbose=verbose)
    award_counts = _load_award_counts(verbose=verbose)
    hof_set = _load_hof_set(verbose=verbose)

    total = 0
    for prospect_row in prospects:
        try:
            outcome = _build_outcome_for_player(
                prospect_row, allstar_counts, award_counts, hof_set
            )
            if outcome:
                label_career(outcome)  # populate events dict
                db.upsert_outcome(outcome)
                total += 1
                if verbose and total % 100 == 0:
                    print(f"  {total} outcomes processed...")
        except Exception as e:
            if verbose:
                print(f"  ERROR on {prospect_row.get('name', 'unknown')}: {e}")
            continue

    if verbose:
        print(f"[outcomes] Created {total} outcome records.")
    return total


def _build_outcome_for_player(
    prospect_row: dict,
    allstar_counts: dict,
    award_counts: dict,
    hof_set: set,
) -> Optional[CareerOutcome]:
    """Build a CareerOutcome from various sources."""
    import pybaseball as pyb

    player_id = prospect_row["player_id"]
    name = prospect_row.get("name", "")

    # Try to look up career MLB stats
    career_pa = 0
    career_ip = 0.0
    career_war = 0.0
    mlb_debut_year = None
    final_mlb_year = None

    try:
        # Use pybaseball's player_id_lookup if available
        # Falls back to name-based lookup
        last, first = (name.split(" ", 1)[1], name.split(" ", 1)[0]) if " " in name else (name, "")
        if last and first:
            lookup_df = pyb.playerid_lookup(last, first)
            if lookup_df is not None and len(lookup_df) > 0:
                # First match
                row = lookup_df.iloc[0]
                mlb_debut_year = int(row["mlb_played_first"]) if row.get("mlb_played_first") else None
                final_mlb_year = int(row["mlb_played_last"]) if row.get("mlb_played_last") else None
    except Exception:
        pass

    # Career stats — try Lahman first since it's reliable
    try:
        career_pa, career_ip, career_war = _lahman_career_stats(player_id, name)
    except Exception:
        pass

    outcome = CareerOutcome(
        player_id=player_id,
        career_complete=(final_mlb_year is None or final_mlb_year < 2022),  # rough heuristic
        career_pa=career_pa,
        career_ip=career_ip,
        career_war=career_war,
        all_star_selections=allstar_counts.get(player_id, 0) or allstar_counts.get(name, 0),
        mvp_count=award_counts.get((player_id, "MVP"), 0) or award_counts.get((name, "MVP"), 0),
        cy_young_count=award_counts.get((player_id, "Cy Young"), 0) or award_counts.get((name, "Cy Young"), 0),
        roy_count=award_counts.get((player_id, "Rookie of the Year"), 0) or award_counts.get((name, "Rookie of the Year"), 0),
        is_hof_inducted=(player_id in hof_set or name in hof_set),
        is_hof_likely=(career_war >= 50.0),
        best_overall_rank=db_get_best_rank(player_id),
        mlb_debut_year=mlb_debut_year,
        final_mlb_year=final_mlb_year,
    )
    return outcome


def db_get_best_rank(player_id: str) -> Optional[int]:
    """Placeholder - real implementation uses ProspectDB.best_rank()."""
    return None


def _lahman_career_stats(player_id: str, name: str) -> tuple[int, float, float]:
    """Pull career PA, IP, WAR from Lahman."""
    import pybaseball as pyb
    if not _ensure_lahman():
        return 0, 0.0, 0.0
    try:
        batting = pyb.lahman.batting()
        if batting is not None and len(batting) > 0:
            # Lahman uses bbrefID typically
            matches = batting[batting["playerID"] == player_id] if "playerID" in batting.columns else batting.iloc[0:0]
            if len(matches) > 0:
                career_pa = int(matches["PA"].sum()) if "PA" in matches.columns else 0
                # Compute PA from components if not available
                if career_pa == 0 and "AB" in matches.columns:
                    career_pa = int(matches["AB"].sum() + matches.get("BB", 0).sum() + matches.get("HBP", 0).sum())
                return career_pa, 0.0, 0.0
    except Exception:
        pass
    return 0, 0.0, 0.0


def _load_allstar_counts(verbose: bool = False) -> dict:
    """Build a dict of player_id -> number of All-Star selections."""
    import pybaseball as pyb
    if not _ensure_lahman():
        return {}
    try:
        df = pyb.lahman.all_star_full()
        if df is None or len(df) == 0:
            return {}
        # Lahman uses playerID
        counts = df.groupby("playerID").size().to_dict() if "playerID" in df.columns else {}
        if verbose:
            print(f"[allstar] Loaded {len(counts)} players with All-Star selections")
        return counts
    except Exception as e:
        if verbose:
            print(f"[allstar] ERROR: {e}")
        return {}


def _load_award_counts(verbose: bool = False) -> dict:
    """Build a dict of (player_id, award_name) -> count."""
    import pybaseball as pyb
    if not _ensure_lahman():
        return {}
    try:
        df = pyb.lahman.awards_players()
        if df is None or len(df) == 0:
            return {}
        counts = {}
        for _, row in df.iterrows():
            pid = row.get("playerID")
            award = row.get("awardID", "")
            if pid and award:
                key = (pid, award)
                counts[key] = counts.get(key, 0) + 1
        if verbose:
            print(f"[awards] Loaded {len(counts)} (player, award) combinations")
        return counts
    except Exception as e:
        if verbose:
            print(f"[awards] ERROR: {e}")
        return {}


def _load_hof_set(verbose: bool = False) -> set:
    """Build set of player_ids inducted into the HOF."""
    import pybaseball as pyb
    if not _ensure_lahman():
        return set()
    try:
        df = pyb.lahman.hall_of_fame()
        if df is None or len(df) == 0:
            return set()
        inducted = df[df["inducted"] == "Y"] if "inducted" in df.columns else df
        ids = set(inducted["playerID"].tolist()) if "playerID" in inducted.columns else set()
        if verbose:
            print(f"[hof] Loaded {len(ids)} inductees")
        return ids
    except Exception as e:
        if verbose:
            print(f"[hof] ERROR: {e}")
        return set()


# ============================================================================
# DIAGNOSTIC HELPER
# ============================================================================

def quick_diagnostic(verbose: bool = True) -> dict:
    """
    Inspect what pybaseball gives us. Run this FIRST before bulk pulls.
    Returns a dict describing what worked and what didn't.
    """
    import pybaseball as pyb
    results = {}
    _ensure_lahman()

    # Player lookup
    try:
        df = pyb.playerid_lookup("trout", "mike")
        results["player_lookup"] = {
            "ok": True,
            "columns": list(df.columns) if df is not None else [],
            "rows": len(df) if df is not None else 0,
        }
        if verbose:
            print(f"[diag] playerid_lookup OK: {len(df)} rows, columns: {list(df.columns)[:8]}")
    except Exception as e:
        results["player_lookup"] = {"ok": False, "error": str(e)}
        if verbose:
            print(f"[diag] playerid_lookup FAILED: {e}")

    # Lahman batting
    try:
        df = pyb.lahman.batting()
        results["lahman_batting"] = {
            "ok": True,
            "columns": list(df.columns) if df is not None else [],
            "rows": len(df) if df is not None else 0,
        }
        if verbose:
            print(f"[diag] lahman.batting OK: {len(df)} rows")
    except Exception as e:
        results["lahman_batting"] = {"ok": False, "error": str(e)}
        if verbose:
            print(f"[diag] lahman.batting FAILED: {e}")

    # Lahman All-Star
    try:
        df = pyb.lahman.all_star_full()
        results["lahman_allstar"] = {
            "ok": True,
            "columns": list(df.columns) if df is not None else [],
            "rows": len(df) if df is not None else 0,
        }
        if verbose:
            print(f"[diag] lahman.allstar_full OK: {len(df)} rows")
    except Exception as e:
        results["lahman_allstar"] = {"ok": False, "error": str(e)}
        if verbose:
            print(f"[diag] lahman.allstar_full FAILED: {e}")

    # Draft data
    try:
        df = pyb.amateur_draft(2021, 1)
        results["amateur_draft"] = {
            "ok": True,
            "columns": list(df.columns) if df is not None else [],
            "rows": len(df) if df is not None else 0,
        }
        if verbose:
            print(f"[diag] amateur_draft(2021,1) OK: {len(df)} rows, columns: {list(df.columns)[:8]}")
    except Exception as e:
        results["amateur_draft"] = {"ok": False, "error": str(e)}
        if verbose:
            print(f"[diag] amateur_draft FAILED: {e}")

    return results
