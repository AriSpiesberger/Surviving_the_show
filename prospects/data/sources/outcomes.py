"""
prospects/data/sources/outcomes.py
========================================

Build career outcome (training-label) rows for every prospect in DB.

Strategy:
  1. Load Chadwick register (key_mlbam + key_bbref + name + mlb_played_first/last)
     once. ~26k rows of every player who has appeared in MLB.
  2. For each prospect, look up by (last, first). If not found, the player
     never reached MLB → MLB_DEBUT=False, all derived events=False.
  3. If found, pull career batting/pitching from Lahman keyed by bbrefID, and
     award counts / All-Star / HOF status.
  4. Save CareerOutcome with events labeled via outcome_labels.label_career.

The pybaseball_loader's outcome path tried to key Lahman by our synthetic
draft IDs, which never matched. This module fixes that by introducing the
name -> bbref translation step explicitly.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Optional

import numpy as np

from prospects.core.labels import ESTABLISHED_SUSTAINED_SEASONS, label_career
from prospects.core.schema import CareerEvent, CareerOutcome
from prospects.core.storage import ProspectDB


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _normalize_name(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", " ", s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _split_name(full: str) -> tuple[str, str]:
    """Return (first, last) with suffixes stripped."""
    parts = _normalize_name(full).split()
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[-1]


def _build_name_index(register) -> dict[tuple[str, str], list]:
    """DEPRECATED in v1.10. Name matching produced label pollution where
    modern prospects got attributed historical players' careers. Kept
    for backwards compatibility; not used by the strict-mlbam path."""
    idx: dict[tuple[str, str], list] = {}
    for row in register.itertuples(index=False):
        last = _normalize_name(row.name_last or "")
        first = _normalize_name(row.name_first or "")
        idx.setdefault((last, first), []).append(row)
    return idx


def _build_mlbam_index(register) -> dict[str, "object"]:
    """Map mlbam_id (Chadwick.key_mlbam) -> register row.
    Strict matching: only a prospect with this exact mlbam_id is linked."""
    idx: dict[str, "object"] = {}
    for row in register.itertuples(index=False):
        km = getattr(row, "key_mlbam", None)
        if km is None:
            continue
        try:
            if isinstance(km, float) and np.isnan(km):
                continue
            idx[str(int(km))] = row
        except (TypeError, ValueError):
            continue
    return idx


def _pick_register_match(matches: list, draft_year: Optional[int]):
    """If multiple players share a name, prefer one whose MLB years are
    consistent with the draft year (debut 0-15 years after draft)."""
    if not matches:
        return None
    if len(matches) == 1 or draft_year is None:
        return matches[0]
    best = None
    best_score = -1
    for m in matches:
        debut = getattr(m, "mlb_played_first", None)
        if debut is None or (isinstance(debut, float) and np.isnan(debut)):
            score = 0
        else:
            gap = int(debut) - draft_year
            score = 1 if 0 <= gap <= 15 else 0
        if score > best_score:
            best_score = score
            best = m
    return best or matches[0]


def extend_yearly(lahman: list[tuple[int, float]], later: dict[int, float],
                  lahman_last: int) -> list[tuple[int, float]]:
    """Per-year values: Lahman for every year it covers, `later` (MLB Stats
    API rows in season_stats) only for years after `lahman_last`. Old labels
    stay exactly as Lahman built them; the API fills the years Lahman lacks."""
    out = [(y, v) for y, v in lahman if y <= lahman_last]
    out += [(y, v) for y, v in later.items() if y > lahman_last]
    return sorted(out)


def extend_years(lahman: list[int], later: list[int], lahman_last: int) -> list[int]:
    """Event years (All-Star selections, award wins), same rule as extend_yearly."""
    return sorted([y for y in lahman if y <= lahman_last]
                  + [y for y in later if y > lahman_last])


def sustained_seasons(pa_by_yr: dict[int, float], ip_by_yr: dict[int, float]) -> list[int]:
    """MLB seasons with meaningful usage (>=100 PA or >=20 IP), sorted.
    Role-fair: a multi-year reliever/utility career qualifies even below
    the 200 IP / 500 PA cumulative bars."""
    yrs = sorted(set(pa_by_yr) | set(ip_by_yr))
    return [y for y in yrs if pa_by_yr.get(y, 0) >= 100 or ip_by_yr.get(y, 0) >= 20]


def established_year(pa_years: list[tuple[int, float]],
                     ip_years: list[tuple[int, float]]) -> Optional[int]:
    """First year the player crosses ANY established path: cumulative
    PA >= 500, cumulative IP >= 200, or the 3rd meaningful MLB season."""
    cands: list[int] = []
    cum_pa = 0.0
    for yr, pa in sorted(pa_years):
        cum_pa += pa
        if cum_pa >= 500:
            cands.append(yr)
            break
    cum_ip = 0.0
    for yr, ip in sorted(ip_years):
        cum_ip += ip
        if cum_ip >= 200:
            cands.append(yr)
            break
    sustained = sustained_seasons(dict(pa_years), dict(ip_years))
    if len(sustained) >= ESTABLISHED_SUSTAINED_SEASONS:
        cands.append(sustained[ESTABLISHED_SUSTAINED_SEASONS - 1])
    return min(cands) if cands else None


def pull_outcomes(
    db: ProspectDB,
    verbose: bool = True,
) -> dict:
    """
    Build outcome rows for all prospects in DB. Returns summary stats.
    """
    import pybaseball as pyb
    from prospects.data.sources.pybaseball import _ensure_lahman

    if not _ensure_lahman():
        if verbose:
            print("[outcomes] Lahman unavailable; aborting")
        return {}

    t0 = time.time()
    if verbose:
        print("[outcomes] loading Chadwick register...")
    register = pyb.chadwick_register()
    name_idx = _build_name_index(register)  # legacy; unused under strict mlbam
    mlbam_idx = _build_mlbam_index(register)

    # Populate year_top_100 / year_top_25 from rankings_history (the global
    # top-100 lists; `overall_rank` is the MLB-wide rank). The event triggers
    # once: the FIRST year the player appeared at overall_rank <= 100 (resp.
    # <= 25). best_rank tracks the lowest overall rank ever, which feeds the
    # event flags in label_career.
    #
    # NOTE: this used to read a `prospect_rankings` table that no longer exists
    # (renamed to rankings_history in the restructure) behind a silent
    # `except: pass`, so top_100/top_25 were always empty. Read the real table
    # and surface failures instead of swallowing them.
    first_top100: dict[str, int] = {}
    first_top25: dict[str, int] = {}
    best_rank: dict[str, int] = {}
    try:
        with db._connect() as conn:
            for pid, as_of, rk in conn.execute(
                "SELECT player_id, as_of, overall_rank FROM rankings_history "
                "WHERE overall_rank IS NOT NULL"
            ).fetchall():
                if as_of is None or rk is None:
                    continue
                yr = int(str(as_of)[:4])
                if pid not in best_rank or rk < best_rank[pid]:
                    best_rank[pid] = int(rk)
                if rk <= 100 and (pid not in first_top100 or yr < first_top100[pid]):
                    first_top100[pid] = yr
                if rk <= 25 and (pid not in first_top25 or yr < first_top25[pid]):
                    first_top25[pid] = yr
    except Exception as e:
        if verbose:
            print(f"  [outcomes] WARNING: could not read rankings_history "
                  f"({type(e).__name__}: {e}); top_100/top_25 will be empty")
    if verbose:
        print(f"  register: {len(register)} rows; {len(name_idx)} (last,first) keys "
              f"({time.time()-t0:.1f}s)")

    if verbose:
        print("[outcomes] loading Lahman tables...")
    t0 = time.time()
    batting = pyb.lahman.batting()
    pitching = pyb.lahman.pitching()
    allstar = pyb.lahman.all_star_full()
    awards = pyb.lahman.awards_players()
    hof_df = pyb.lahman.hall_of_fame()

    # Aggregate Lahman by playerID for fast lookup
    bat_agg = batting.groupby("playerID").agg(
        AB=("AB", "sum"),
        BB=("BB", "sum"),
        HBP=("HBP", "sum") if "HBP" in batting.columns else ("BB", "sum"),
        SF=("SF", "sum") if "SF" in batting.columns else ("BB", "sum"),
        debut_year=("yearID", "min"),
        last_year=("yearID", "max"),
    )
    bat_agg["PA"] = (bat_agg["AB"].fillna(0)
                     + bat_agg["BB"].fillna(0)
                     + bat_agg["HBP"].fillna(0)
                     + bat_agg["SF"].fillna(0)).astype(int)
    bat_pa = bat_agg["PA"].to_dict()
    bat_debut = bat_agg["debut_year"].to_dict()
    bat_last = bat_agg["last_year"].to_dict()

    pitch_agg = pitching.groupby("playerID").agg(
        IPouts=("IPouts", "sum"),
        debut_year=("yearID", "min"),
        last_year=("yearID", "max"),
    )
    pitch_ip = (pitch_agg["IPouts"].fillna(0) / 3.0).to_dict()
    pitch_debut = pitch_agg["debut_year"].to_dict()
    pitch_last = pitch_agg["last_year"].to_dict()

    allstar_counts = allstar.groupby("playerID").size().to_dict()

    awards_lower = awards.copy()
    awards_lower["awardID_lc"] = awards_lower["awardID"].str.lower()
    mvp_counts = awards_lower[awards_lower["awardID_lc"].str.contains(
        "most valuable player", na=False)].groupby("playerID").size().to_dict()
    cy_counts = awards_lower[awards_lower["awardID_lc"].str.contains(
        "cy young", na=False)].groupby("playerID").size().to_dict()
    roy_counts = awards_lower[awards_lower["awardID_lc"].str.contains(
        "rookie of the year", na=False)].groupby("playerID").size().to_dict()

    hof_inductees = set(
        hof_df[hof_df["inducted"].astype(str).str.upper() == "Y"]["playerID"].tolist()
    )
    hof_year_map = (hof_df[hof_df["inducted"].astype(str).str.upper() == "Y"]
                    .groupby("playerID")["yearID"].min().to_dict())

    # Per-year cumulative PA / IP — to determine year ESTABLISHED_MLB triggers.
    bat_year_grp = batting.groupby(["playerID", "yearID"]).agg(
        AB=("AB", "sum"), BB=("BB", "sum"),
        HBP=("HBP", "sum") if "HBP" in batting.columns else ("BB", "sum"),
        SF=("SF", "sum") if "SF" in batting.columns else ("BB", "sum"),
    )
    bat_year_grp["PA"] = (bat_year_grp["AB"].fillna(0)
                          + bat_year_grp["BB"].fillna(0)
                          + bat_year_grp["HBP"].fillna(0)
                          + bat_year_grp["SF"].fillna(0))
    bat_year_pa: dict[str, list] = {}
    for (pid, yr), row in bat_year_grp["PA"].items():
        bat_year_pa.setdefault(pid, []).append((int(yr), float(row)))

    pit_year_grp = pitching.groupby(["playerID", "yearID"]).agg(IPouts=("IPouts", "sum"))
    pit_year_grp["IP"] = pit_year_grp["IPouts"].fillna(0) / 3.0
    pit_year_ip: dict[str, list] = {}
    for (pid, yr), row in pit_year_grp["IP"].items():
        pit_year_ip.setdefault(pid, []).append((int(yr), float(row)))

    # Years of All-Star selections and award wins.
    allstar_years: dict[str, list] = {}
    for pid, yr in allstar[["playerID", "yearID"]].itertuples(index=False, name=None):
        allstar_years.setdefault(pid, []).append(int(yr))
    for pid in allstar_years:
        allstar_years[pid].sort()

    major_award_mask = awards_lower["awardID_lc"].str.contains(
        "most valuable player|cy young|rookie of the year", na=False, regex=True
    )
    lahman_award_years: dict[str, list[int]] = (
        awards_lower[major_award_mask].groupby("playerID")["yearID"]
        .apply(lambda s: sorted(int(y) for y in s)).to_dict())

    # Lahman stops at 2021 while the model treats outcomes as observed through
    # MAX_OBS_YEAR. Seasons after Lahman's last year come from the MLB Stats
    # API instead: playing time from the season_stats MLB rows that
    # backfills/mlb_seasons_stitch writes, All-Star and award recipients from
    # sources/mlb_awards. Lahman stays authoritative for every year it covers.
    lahman_last_play = int(max(batting["yearID"].max(), pitching["yearID"].max()))
    lahman_last_as = int(allstar["yearID"].max())
    lahman_last_award = int(awards["yearID"].max())

    api_pa: dict[str, dict[int, float]] = {}
    api_ip: dict[str, dict[int, float]] = {}
    with db._connect() as conn:
        for pid, yr, pa, ip in conn.execute(
                "SELECT player_id, season_year, pa, ip FROM season_stats WHERE level = 'MLB'"):
            if pa:
                api_pa.setdefault(pid, {})[int(yr)] = float(pa)
            if ip:
                api_ip.setdefault(pid, {})[int(yr)] = float(ip)

    late = [y for d in (api_pa, api_ip) for ys in d.values() for y in ys if y > lahman_last_play]
    if not late:
        print(f"  [outcomes] WARNING: season_stats has no MLB rows after {lahman_last_play}; "
              f"run backfills.mlb_seasons_stitch first or every later season is missing "
              f"from established_mlb / career PA / IP")

    from prospects.data.sources import mlb_awards
    api_awards = mlb_awards.load()
    api_as = api_awards.get("all_star", {})
    api_kind_years = {k: api_awards.get(k, {}) for k in ("mvp", "cy_young", "roy")}

    if verbose:
        print(f"  Lahman last season: play {lahman_last_play}, All-Star {lahman_last_as}, "
              f"awards {lahman_last_award}; later seasons from the Stats API "
              f"({len(api_pa) + len(api_ip):,} MLB stat lines by player, "
              f"{sum(len(v) for v in api_awards.values()):,} award winners)")

    def _yearly(bbref: Optional[str], pid: str) -> tuple[list, list]:
        pa = extend_yearly(bat_year_pa.get(bbref, []) if bbref else [],
                           api_pa.get(pid, {}), lahman_last_play)
        ip = extend_yearly(pit_year_ip.get(bbref, []) if bbref else [],
                           api_ip.get(pid, {}), lahman_last_play)
        return pa, ip

    def _award_years(bbref: Optional[str], mlbam_key: Optional[str]) -> dict[str, list[int]]:
        lahman_as = allstar_years.get(bbref, []) if bbref else []
        out = {"all_star": extend_years(lahman_as, api_as.get(mlbam_key, []), lahman_last_as)}
        lahman_major = lahman_award_years.get(bbref, []) if bbref else []
        api_major = sorted(y for kind in api_kind_years.values()
                           for y in kind.get(mlbam_key, []))
        out["major"] = extend_years(lahman_major, api_major, lahman_last_award)
        for kind, by_id in api_kind_years.items():
            out[kind] = [y for y in by_id.get(mlbam_key, []) if y > lahman_last_award]
        return out

    if verbose:
        print(f"  Lahman ready ({time.time()-t0:.1f}s)")

    prospects = db.all_prospects()
    if verbose:
        print(f"[outcomes] scanning {len(prospects)} prospects...")

    n_matched = 0
    n_unmatched = 0
    n_written = 0

    for i, p in enumerate(prospects):
        # STRICT MLBAM matching: a prospect is linked to the Chadwick row
        # iff its mlbam_id matches exactly. No name fallback (the prior
        # name-based fallback caused career outcomes for ~315 modern
        # prospects to be 19th-century player careers, including 15
        # spurious HOF inductions).
        mlbam = p.get("mlbam_id")
        match = None
        if mlbam and str(mlbam) not in ("", "-1"):
            match = mlbam_idx.get(str(mlbam))

        if match is None:
            outcome = CareerOutcome(
                player_id=p["player_id"],
                career_complete=True,
                best_overall_rank=best_rank.get(p["player_id"]),
                mlb_debut_year=None,
                final_mlb_year=None,
            )
            n_unmatched += 1
        else:
            bbref = getattr(match, "key_bbref", None)
            mlbam = getattr(match, "key_mlbam", None)
            if mlbam is not None and not (isinstance(mlbam, float) and np.isnan(mlbam)):
                db.set_mlbam_id(p["player_id"], str(int(mlbam)))
            debut = getattr(match, "mlb_played_first", None)
            last_year = getattr(match, "mlb_played_last", None)
            debut_int = (int(debut) if debut is not None
                         and not (isinstance(debut, float) and np.isnan(debut))
                         else None)
            last_int = (int(last_year) if last_year is not None
                        and not (isinstance(last_year, float) and np.isnan(last_year))
                        else None)

            mlbam_key = (str(int(mlbam)) if mlbam is not None
                         and not (isinstance(mlbam, float) and np.isnan(mlbam)) else None)
            pa_years, ip_years = _yearly(bbref, p["player_id"])
            award_years = _award_years(bbref, mlbam_key)
            n_late_as = sum(1 for y in award_years["all_star"] if y > lahman_last_as)

            # Lahman totals plus the post-Lahman seasons. The Lahman counts are
            # kept as-is (all_star_full has two rows in the 1959-62 two-game
            # years), so only the years Lahman lacks are added.
            career_pa = int(bat_pa.get(bbref, 0) or 0) + int(sum(
                v for y, v in pa_years if y > lahman_last_play))
            career_ip = float(pitch_ip.get(bbref, 0.0) or 0.0) + float(sum(
                v for y, v in ip_years if y > lahman_last_play))
            n_sustained = len(sustained_seasons(dict(pa_years), dict(ip_years)))
            allstar_n = int(allstar_counts.get(bbref, 0)) + n_late_as
            mvp_n = int(mvp_counts.get(bbref, 0)) + len(award_years["mvp"])
            cy_n = int(cy_counts.get(bbref, 0)) + len(award_years["cy_young"])
            roy_n = int(roy_counts.get(bbref, 0)) + len(award_years["roy"])
            is_hof = bbref in hof_inductees

            outcome = CareerOutcome(
                player_id=p["player_id"],
                career_complete=(last_int is None or last_int < 2025),
                career_pa=career_pa,
                career_ip=career_ip,
                n_sustained_mlb_seasons=n_sustained,
                career_war=0.0,  # Lahman doesn't carry WAR
                all_star_selections=allstar_n,
                mvp_count=mvp_n,
                cy_young_count=cy_n,
                roy_count=roy_n,
                is_hof_inducted=is_hof,
                is_hof_likely=False,
                best_overall_rank=best_rank.get(p["player_id"]),
                mlb_debut_year=debut_int,
                final_mlb_year=last_int,
            )
            n_matched += 1

        label_career(outcome)
        db.upsert_outcome(outcome)

        # Per-event trigger years (for future-only loss masking).
        # v1.11: year_top_100 / year_top_25 are populated from
        # prospect_rankings (BBC top-100). These are now full first-class
        # hazard events the model trains on.
        trigger: dict = {
            "year_top_100": first_top100.get(p["player_id"]),
            "year_top_25": first_top25.get(p["player_id"]),
            "year_established_mlb": None,
            "year_all_star_once": None,
            "year_all_star_three": None,
            "year_major_award": None,
            "year_hof_trajectory": None,
        }
        if match is not None:
            bbref_id = getattr(match, "key_bbref", None)
            # A player Lahman never saw (debut after its last season) has no
            # bbref row there but does have Stats API seasons and awards.
            trigger["year_established_mlb"] = established_year(pa_years, ip_years)
            as_yrs = award_years["all_star"]
            if as_yrs:
                trigger["year_all_star_once"] = as_yrs[0]
            if len(as_yrs) >= 3:
                trigger["year_all_star_three"] = as_yrs[2]
            if award_years["major"]:
                trigger["year_major_award"] = award_years["major"][0]
            if bbref_id and bbref_id in hof_year_map:
                trigger["year_hof_trajectory"] = int(hof_year_map[bbref_id])
        db.set_event_trigger_years(p["player_id"], trigger)

        n_written += 1

        if verbose and (i + 1) % 2500 == 0:
            print(f"  {i+1}/{len(prospects)} processed "
                  f"(matched={n_matched}, no-MLB={n_unmatched})")

    summary = {
        "prospects_scanned": len(prospects),
        "matched_in_register": n_matched,
        "no_mlb_appearance": n_unmatched,
        "outcomes_written": n_written,
    }
    if verbose:
        print(f"[outcomes] done: {summary}")
    return summary
