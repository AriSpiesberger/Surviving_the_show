"""
prospects/ingestion/outcomes_loader.py
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

from prospects.outcome_labels import label_career
from prospects.schema import CareerEvent, CareerOutcome
from prospects.storage import ProspectDB


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


def pull_outcomes(
    db: ProspectDB,
    verbose: bool = True,
) -> dict:
    """
    Build outcome rows for all prospects in DB. Returns summary stats.
    """
    import pybaseball as pyb
    from prospects.ingestion.pybaseball_loader import _ensure_lahman

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

    # v1.11: Load prospect_rankings to populate year_top_100 / year_top_25.
    # The event triggers once: the FIRST year the player appeared on the
    # BBC top-100 (resp. top-25).
    first_top100: dict[str, int] = {}
    first_top25: dict[str, int] = {}
    try:
        with db._connect() as conn:
            for r in conn.execute(
                "SELECT player_id, year, rank FROM prospect_rankings"
            ).fetchall():
                pid, yr, rk = r[0], r[1], r[2]
                if yr is None or rk is None:
                    continue
                if pid not in first_top100 or yr < first_top100[pid]:
                    first_top100[pid] = int(yr)
                if rk <= 25:
                    if pid not in first_top25 or yr < first_top25[pid]:
                        first_top25[pid] = int(yr)
    except Exception:
        # prospect_rankings table may not exist on older DBs; skip silently
        pass
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
    award_year_map = (awards_lower[major_award_mask]
                      .groupby("playerID")["yearID"].min().to_dict())

    def _established_year(bbref: str) -> Optional[int]:
        """First year cumulative PA >= 500 or cumulative IP >= 200."""
        cum_pa = 0.0
        for yr, pa in sorted(bat_year_pa.get(bbref, [])):
            cum_pa += pa
            if cum_pa >= 500:
                return yr
        cum_ip = 0.0
        for yr, ip in sorted(pit_year_ip.get(bbref, [])):
            cum_ip += ip
            if cum_ip >= 200:
                return yr
        return None

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

            career_pa = int(bat_pa.get(bbref, 0) or 0)
            career_ip = float(pitch_ip.get(bbref, 0.0) or 0.0)
            allstar_n = int(allstar_counts.get(bbref, 0))
            mvp_n = int(mvp_counts.get(bbref, 0))
            cy_n = int(cy_counts.get(bbref, 0))
            roy_n = int(roy_counts.get(bbref, 0))
            is_hof = bbref in hof_inductees

            outcome = CareerOutcome(
                player_id=p["player_id"],
                career_complete=(last_int is None or last_int < 2025),
                career_pa=career_pa,
                career_ip=career_ip,
                career_war=0.0,  # Lahman doesn't carry WAR
                all_star_selections=allstar_n,
                mvp_count=mvp_n,
                cy_young_count=cy_n,
                roy_count=roy_n,
                is_hof_inducted=is_hof,
                is_hof_likely=False,
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
            if bbref_id:
                trigger["year_established_mlb"] = _established_year(bbref_id)
                as_yrs = allstar_years.get(bbref_id, [])
                if as_yrs:
                    trigger["year_all_star_once"] = as_yrs[0]
                if len(as_yrs) >= 3:
                    trigger["year_all_star_three"] = as_yrs[2]
                if bbref_id in award_year_map:
                    trigger["year_major_award"] = int(award_year_map[bbref_id])
                if bbref_id in hof_year_map:
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
