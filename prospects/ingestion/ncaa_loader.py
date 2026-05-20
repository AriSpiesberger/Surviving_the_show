"""
prospects/ingestion/ncaa_loader.py
====================================

College baseball stats from the ncaa_bbStats package.

Coverage:
- Team stats: 2002-2025 (D1, D2, D3)
- Player stats: 2021-2025 (limited historical depth)
- MLB Draft data: 1965-2025

Install:
    pip install ncaa_bbStats

Usage:
    from prospects.ingestion.ncaa_loader import pull_college_stats_for_player
    pull_college_stats_for_player(db, "Gage Wood", "Arkansas")

The college data is best used as a layer on top of pybaseball draft data:
for any college draftee, look up their college performance and add it as
features.
"""

from __future__ import annotations

from typing import Optional

from prospects.schema import SeasonStats
from prospects.storage import ProspectDB


def pull_college_stats_for_player(
    db: ProspectDB,
    player_id: str,
    player_name: str,
    school_substr: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """
    For a given player, pull all available college batting and pitching stats
    and save as SeasonStats rows with level='NCAA-D1'.

    The player_name must match how they're listed in the NCAA data.
    school_substr is an optional filter to disambiguate common names.

    Returns:
        Number of season-stat rows added.
    """
    try:
        import ncaa_bbStats as ncaa
    except ImportError:
        if verbose:
            print("[ncaa] ncaa_bbStats not installed. pip install ncaa_bbStats")
        return 0

    total = 0

    # Pull batting seasons (qualified pool for accuracy)
    for stat_type, level_tag in [("batting", "NCAA-D1"), ("pitching", "NCAA-D1")]:
        try:
            rows = ncaa.get_player_rows(
                stat_type=stat_type,
                qualifier="noMin",  # most permissive
                player_name=player_name,
                team_substr=school_substr,
            )
            if not rows:
                continue

            for row in rows:
                try:
                    year = int(row.get("year") or row.get("Year") or 0)
                    if not year:
                        continue

                    if stat_type == "batting":
                        s = SeasonStats(
                            player_id=player_id,
                            season_year=year,
                            level=level_tag,
                            org=row.get("team") or row.get("Team", ""),
                            primary_position=row.get("pos", "") or row.get("position", ""),
                            pa=_to_int(row.get("pa") or row.get("PA")),
                            home_runs=_to_int(row.get("hr") or row.get("HR")),
                            stolen_bases=_to_int(row.get("sb") or row.get("SB")),
                            avg=_to_float(row.get("avg") or row.get("AVG")),
                            obp=_to_float(row.get("obp") or row.get("OBP")),
                            slg=_to_float(row.get("slg") or row.get("SLG")),
                        )
                    else:  # pitching
                        s = SeasonStats(
                            player_id=player_id,
                            season_year=year,
                            level=level_tag,
                            org=row.get("team") or row.get("Team", ""),
                            primary_position="P",
                            ip=_to_float(row.get("ip") or row.get("IP")) or 0.0,
                            era=_to_float(row.get("era") or row.get("ERA")),
                            k9=_to_float(row.get("k_9") or row.get("K/9")),
                            bb9=_to_float(row.get("bb_9") or row.get("BB/9")),
                        )
                    db.upsert_season_stats(s)
                    total += 1
                except Exception as e:
                    if verbose:
                        print(f"[ncaa] row parse ERROR for {player_name}: {e}")
        except Exception as e:
            if verbose:
                print(f"[ncaa] {stat_type} ERROR for {player_name}: {e}")

    if verbose and total > 0:
        print(f"[ncaa] {player_name}: {total} college seasons loaded")
    return total


def pull_mlb_draft_year(year: int, verbose: bool = True) -> list[dict]:
    """
    Pull MLB draft picks for a year from ncaa_bbStats.

    Returns a list of dicts with draft info — useful for finding which players
    were drafted from each college program.
    """
    try:
        import ncaa_bbStats as ncaa
        result = ncaa.parse_mlb_draft(year)
        if verbose:
            print(f"[ncaa-draft] {year}: {len(result) if result else 0} picks parsed")
        return result or []
    except ImportError:
        if verbose:
            print("[ncaa] ncaa_bbStats not installed")
        return []
    except Exception as e:
        if verbose:
            print(f"[ncaa-draft] {year} ERROR: {e}")
        return []


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ============================================================================
# DIAGNOSTIC HELPER
# ============================================================================

def quick_diagnostic(verbose: bool = True) -> dict:
    """Inspect what ncaa_bbStats actually returns."""
    results = {}

    try:
        import ncaa_bbStats as ncaa
        results["package"] = "installed"
        if verbose:
            print(f"[diag] ncaa_bbStats available")
            print(f"  exported: {[a for a in dir(ncaa) if not a.startswith('_')][:15]}")
    except ImportError as e:
        results["package"] = f"not installed: {e}"
        if verbose:
            print(f"[diag] ncaa_bbStats NOT INSTALLED: {e}")
        return results

    # list_available_years
    try:
        years = ncaa.list_available_years("batting", "qualified")
        results["years"] = years
        if verbose:
            print(f"[diag] available batting years (qualified): {years}")
    except Exception as e:
        results["years_error"] = str(e)
        if verbose:
            print(f"[diag] list_available_years FAILED: {e}")

    # list_batters
    try:
        latest = max(results.get("years", [2024]))
        batters = ncaa.list_batters("qualified", year=latest)
        results["batters_sample"] = batters[:10] if batters else []
        if verbose:
            print(f"[diag] sample batters ({latest}): {results['batters_sample']}")
    except Exception as e:
        results["batters_error"] = str(e)
        if verbose:
            print(f"[diag] list_batters FAILED: {e}")

    # get_player_rows
    if results.get("batters_sample"):
        try:
            sample_player = results["batters_sample"][0]
            rows = ncaa.get_player_rows("batting", "noMin", sample_player)
            results["sample_row_keys"] = list(rows[0].keys()) if rows else []
            if verbose:
                print(f"[diag] get_player_rows({sample_player!r}) keys: "
                      f"{results['sample_row_keys'][:15]}")
        except Exception as e:
            results["rows_error"] = str(e)
            if verbose:
                print(f"[diag] get_player_rows FAILED: {e}")

    # MLB draft
    try:
        draft = ncaa.parse_mlb_draft(2023)
        results["draft_sample_count"] = len(draft) if draft else 0
        if draft:
            results["draft_sample"] = draft[0] if isinstance(draft, list) else None
            if verbose:
                print(f"[diag] parse_mlb_draft(2023): {len(draft)} picks")
                print(f"  first pick: {draft[0] if isinstance(draft, list) else 'unknown format'}")
    except Exception as e:
        results["draft_error"] = str(e)
        if verbose:
            print(f"[diag] parse_mlb_draft FAILED: {e}")

    return results
