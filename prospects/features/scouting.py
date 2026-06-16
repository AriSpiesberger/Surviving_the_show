"""
prospects/features/scouting.py
================================

Rich, descriptive feature builder for the hazard model.

Design intent: the downstream hazard model is not great at extracting signal
on its own, so the features need to encode everything a scout / front office
would actually look at. That means:

  - Raw rate stats at each level
  - **Level-adjusted** versions of every rate stat (woba_vs_level etc.)
    because a .350 wOBA at A-ball and at AAA mean very different things
  - **Age-vs-level** (the single most-cited prospect signal)
  - **Career-to-date cumulative** stats (career_milb_pa, best_so_far, etc.)
  - **Trajectory**: max level reached, time to A/AA/AAA, years stuck, etc.
  - **Deltas** (year-over-year change in every rate stat)
  - **Acceleration** (delta-of-delta — second derivative of trajectory)
  - **Sample-size flags** so the model can discount tiny-PA samples

Baselines (league medians by level) are computed once from the dataset and
cached to JSON so that this builder is deterministic at inference time.

CLI:
    python -m prospects.features.scouting --compute-baselines \\
        --out baselines/milb_baselines.json

Module API:
    FEATURE_NAMES, N_FEATURES
    build_scouting_features(prospect, stats, as_of_year, baselines, milb_only=True) -> np.ndarray
    compute_baselines(db) -> dict
    load_baselines(path) -> dict
    save_baselines(baselines, path) -> None
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Optional

import numpy as np

from prospects.features.scouting_grades import (
    SCOUTING_GRADE_NAMES, scouting_grade_dict,
)
from prospects.storage import ProspectDB


# ============================================================================
# Constants
# ============================================================================

LEVEL_RANK: dict[str, int] = {
    "DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
    "A-": 2, "A": 3, "A+": 4,
    "AA": 5, "AAA": 6,
    "MLB": 7,
    "NCAA-D1": 3, "NCAA-D2": 2, "NCAA-D3": 2,
}

LEVELS_FOR_BASELINES = ["RK", "A-", "A", "A+", "AA", "AAA"]

PREMIUM_POSITIONS = {"SS", "C", "CF"}
PITCHER_POS = {"P", "RHP", "LHP", "SP", "RP"}

import math as _math

# Missing-value sentinel. NaN (not -1.0) so sklearn HistGradientBoosting
# treats it as native missingness and learns optimal branching, rather
# than treating it as a real numeric value.
MISSING = float("nan")
WINDOW = 3  # yT, y1, y2


def _is_missing(v) -> bool:
    """True iff v is the MISSING sentinel (NaN). Robust against int/float."""
    if v is None:
        return True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return _math.isnan(f)

HIT_RATE_STATS = ["woba", "iso", "k_pct", "bb_pct", "babip", "obp", "slg", "avg"]
PIT_RATE_STATS = ["era", "fip", "k9", "bb9", "hr9", "whip"]


# ============================================================================
# Feature name registry
# ============================================================================

PEDIGREE_FEATS = [
    "is_pitcher",
    "is_international",
    "is_college_draftee",
    "is_drafted",
    "is_premium_position",
    "is_catcher",
    "is_shortstop",
    "is_center_field",
    "draft_round",
    "draft_pick",
    "log_signing_bonus",
    "has_signing_bonus",
    "age_at_signing",
    # v1.9: biometrics + bonus-vs-slot (from MLB Stats API backfill).
    "height_inches",
    "weight_lbs",
    "bmi",
    "bats_L",        # 1 if bats Left (R is baseline)
    "bats_S",        # 1 if switch
    "throws_L",      # 1 if throws Left (R is baseline)
    "log_pick_value",
    "bonus_vs_slot",  # signing_bonus / pick_value (overpay/underpay slot)
    # v1.10: BBC Top-100 prospect rankings (as-of aware - never look ahead).
    "ever_top100",            # 1 if ranked in any year <= as_of_year
    "best_top100_rank",       # min rank in years <= as_of_year (MISSING if never)
    "recent_top100_rank",     # rank in the latest year <= as_of_year
    "times_top100",           # count of years in top-100 <= as_of_year
    "years_since_first_top100",  # as_of_year - earliest top-100 year
    "log_best_top100_rank",   # log(rank+1) — better as a linear feature
    # v2.0c: TheBaseballCube ORG (team-level) prospect rankings, as-of aware.
    # Much higher coverage than top-100 (every org top-30 vs national top-100),
    # so this fires for far more players. Per year we collapse multi-source
    # lists to the best (min) org rank. Trends are anchored to the most recent
    # ranked year <= as_of; POSITIVE trend = climbed toward #1 in the org.
    "ever_org_ranked",            # 1 if on any org list in a year <= as_of_year
    "best_org_rank",              # min org rank over years <= as_of (MISSING if never)
    "recent_org_rank",            # org rank in the latest year <= as_of
    "times_org_ranked",           # count of distinct years on an org list <= as_of
    "years_since_first_org_ranked",  # as_of_year - earliest org-ranked year
    "log_best_org_rank",          # log(best_org_rank + 1)
    "org_rank_trend_1y",          # rank(recent-1) - rank(recent); + = climbed
    "org_rank_trend_2y",          # rank(recent-2) - rank(recent); + = climbed
]

# Per-year features (computed for yT, y1, y2)
HIT_PER_YEAR = [
    "pa",
    "woba", "woba_vs_level",
    "iso", "iso_vs_level",
    "k_pct", "k_pct_vs_level",
    "bb_pct", "bb_pct_vs_level",
    "bb_k_ratio",
    "babip",
    "obp",
    "slg",
    "avg",
    "hr_per_pa",
    "sb_per_pa",
]
PIT_PER_YEAR = [
    "ip",
    "era", "era_vs_level",
    "fip", "fip_vs_level",
    "k9", "k9_vs_level",
    "bb9", "bb9_vs_level",
    "whip",
    "hr9",
    "k_bb_ratio",
    "velo_avg",
]
SHARED_PER_YEAR = [
    "level_rank",
    "age",
    "age_vs_level",
    "n_levels_in_year",  # 1 = stable, 2+ = promoted mid-season
    "highest_level_in_year",
]

CAREER_TO_DATE_FEATS = [
    "career_milb_pa",
    "career_milb_ip",
    "career_milb_hr",
    "career_milb_sb",
    "career_milb_seasons",
    "distinct_levels_played",
    "max_level_ever",
    "years_since_max_level_ever",  # as_of - last year touched max_level_ever
                                    # (any appearance, no IP/PA threshold)
    "max_level_qualified",       # max level w/ >=30 IP or >=100 PA at it
    "years_since_max_level",     # as_of - last year at max_level_qualified
    "bottom_since_max_level",    # min level played after reaching max_qualified
                                  # (with >=25 IP or >=75 PA at that lower lvl)
    # Tier 1: gap/missed-time signals
    "had_lost_season",                 # any gap year with no PA/IP between active yrs
    "seasons_missed_career",           # count of such gap years
    "consecutive_active_seasons",      # current active-year streak
    "current_pa_vs_max_pa",            # this year's PA / max single-year PA
    "current_ip_vs_max_ip",            # this year's IP / max single-year IP
    # Tier 2: performance regression vs personal peak
    "current_woba_vs_best_woba",
    "current_bb9_vs_best_bb9",         # >1 = worse (BB/9 up = bad)
    "current_k9_vs_best_k9",           # <1 = worse (K/9 down = bad)
    "current_era_vs_best_era",         # >1 = worse
    # Tier 3: trajectory (age_at_first_* and n_org_changes dropped — age data
    # all NULL in DB; season_stats.org is the MiLB affiliate not MLB org)
    "age_at_first_AA",                 # from birth_date in prospects table
    "age_at_first_AAA",
    "n_demotions_career",              # count of yr-over-yr level decreases
    # Tier 4: pitcher workload signals
    "career_max_ip_in_year",           # peak single-season IP
    "pct_seasons_above_50_ip",         # fraction of seasons w/ IP >= 50
    "delta_ip_yT_vs_yT_minus_1",       # this yr IP - last yr IP
    "min_level_played",
    "years_to_A_or_higher",
    "years_to_AA",
    "years_to_AAA",
    "reached_AA",
    "reached_AAA",
    "best_woba",
    "best_iso",
    "best_obp",
    "best_slg",
    "best_k_pct",   # for hitter: lowest K%
    "best_bb_pct",  # for hitter: highest BB%
    "best_era",     # lowest
    "best_fip",     # lowest
    "best_k9",      # highest
    "best_bb9",     # lowest
    "best_whip",    # lowest
    "pa_at_AAA_career",
    "ip_at_AAA_career",
    "pa_at_AA_career",
    "ip_at_AA_career",
    "pa_at_max_level_career",
    "ip_at_max_level_career",
    "pct_pa_at_AAA",
    "pct_ip_at_AAA",
]

TRAJECTORY_FEATS = [
    "promotion_velocity",          # (max_level_ever - 1) / years_in_pro
    "promotion_acceleration",      # (level_yT - level_y1) - (level_y1 - level_y2)
    "years_stuck_at_max_level",    # consecutive years at the max level ever reached
    "years_at_current_level",      # consecutive years at as_of_year's level
    "level_change_yT_vs_y1",
    "level_change_y1_vs_y2",
    "repeat_level_woba_slope",     # for hitters with multiple years at one level
    "repeat_level_fip_slope",      # for pitchers ditto
]

DELTA_HIT_FEATS = [
    "delta_woba", "delta_iso", "delta_k_pct", "delta_bb_pct",
    "delta_babip", "delta_obp", "delta_slg", "delta_avg",
    "delta_hr_per_pa", "delta_sb_per_pa",
    "delta_pa",
    "delta_woba_vs_level", "delta_iso_vs_level",
    "delta_k_pct_vs_level", "delta_bb_pct_vs_level",
]
DELTA_PIT_FEATS = [
    "delta_era", "delta_fip", "delta_k9", "delta_bb9",
    "delta_whip", "delta_hr9",
    "delta_ip",
    "delta_era_vs_level", "delta_fip_vs_level",
    "delta_k9_vs_level", "delta_bb9_vs_level",
]
DELTA_SHARED_FEATS = [
    "delta_age",
    "delta_age_vs_level",
]

ACCEL_HIT_FEATS = [
    "accel_woba", "accel_iso", "accel_k_pct", "accel_bb_pct",
    "accel_obp", "accel_slg",
]
ACCEL_PIT_FEATS = [
    "accel_era", "accel_fip", "accel_k9", "accel_bb9",
]
ACCEL_SHARED_FEATS = [
    "accel_level",  # second derivative of level rank
]

WINDOW_SUMMARY = [
    "n_years_observed_in_window",
    "max_level_seen_in_window",
    "pa_in_window",
    "ip_in_window",
    "years_in_pro",
    "age_at_as_of",
    "age_at_as_of_vs_level",
    "years_in_current_system",
    "has_any_milb_data",
    "has_pitching_data",
    "has_hitting_data",
]


def _build_feature_names() -> list[str]:
    names: list[str] = []
    names += PEDIGREE_FEATS

    for lag in range(WINDOW):
        sfx = "_yT" if lag == 0 else f"_y{lag}"
        names += [f + sfx for f in HIT_PER_YEAR]
        names += [f + sfx for f in PIT_PER_YEAR]
        names += [f + sfx for f in SHARED_PER_YEAR]

    names += CAREER_TO_DATE_FEATS
    names += TRAJECTORY_FEATS
    names += DELTA_HIT_FEATS + DELTA_PIT_FEATS + DELTA_SHARED_FEATS
    names += ACCEL_HIT_FEATS + ACCEL_PIT_FEATS + ACCEL_SHARED_FEATS
    names += WINDOW_SUMMARY
    names += SCOUTING_GRADE_NAMES  # FG-board + TWTC point-in-time grades
    return names


FEATURE_NAMES = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)


# ============================================================================
# Baseline computation (league medians by level)
# ============================================================================

def compute_baselines(
    db: ProspectDB,
    min_pa: int = 50,
    min_ip: float = 20.0,
) -> dict:
    """Compute median rate stat and median age per level, from all
    season_stats rows with enough sample size. Median is more robust than
    mean against the skinny right-tail in the dataset.

    Returns:
        {
          "hit": {level: {stat: median_value}},
          "pit": {level: {stat: median_value}},
          "age_hit": {level: median_age},
          "age_pit": {level: median_age},
          "_meta": {"min_pa": ..., "min_ip": ...},
        }
    """
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM season_stats").fetchall()]

    hit_by_level: dict[str, dict[str, list[float]]] = {}
    pit_by_level: dict[str, dict[str, list[float]]] = {}
    age_hit: dict[str, list[float]] = {}
    age_pit: dict[str, list[float]] = {}

    for r in rows:
        lvl = (r.get("level") or "").upper()
        if lvl not in LEVELS_FOR_BASELINES:
            continue
        pa = r.get("pa") or 0
        ip = r.get("ip") or 0.0
        age = r.get("age_during_season")

        if pa >= min_pa:
            bucket = hit_by_level.setdefault(lvl, {})
            for stat in HIT_RATE_STATS:
                v = r.get(stat)
                if v is not None:
                    bucket.setdefault(stat, []).append(float(v))
            # hitter-perspective hr/pa, sb/pa
            hr = r.get("home_runs"); sb = r.get("stolen_bases")
            if hr is not None and pa:
                bucket.setdefault("hr_per_pa", []).append(float(hr) / float(pa))
            if sb is not None and pa:
                bucket.setdefault("sb_per_pa", []).append(float(sb) / float(pa))
            if age is not None:
                age_hit.setdefault(lvl, []).append(float(age))

        if ip >= min_ip:
            bucket = pit_by_level.setdefault(lvl, {})
            for stat in PIT_RATE_STATS:
                v = r.get(stat)
                if v is not None:
                    bucket.setdefault(stat, []).append(float(v))
            if age is not None:
                age_pit.setdefault(lvl, []).append(float(age))

    def _med_dict(d):
        out = {}
        for lvl, stats in d.items():
            out[lvl] = {k: float(statistics.median(v)) for k, v in stats.items() if v}
        return out

    return {
        "hit": _med_dict(hit_by_level),
        "pit": _med_dict(pit_by_level),
        "age_hit": {k: float(statistics.median(v)) for k, v in age_hit.items() if v},
        "age_pit": {k: float(statistics.median(v)) for k, v in age_pit.items() if v},
        "_meta": {"min_pa": min_pa, "min_ip": min_ip,
                  "n_rows": len(rows), "levels": LEVELS_FOR_BASELINES},
    }


def save_baselines(baselines: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(baselines, f, indent=2)


def load_baselines(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Helpers — aggregation
# ============================================================================

def _weighted_avg(rows: list[dict], key: str, weight_key: str,
                  transform=None) -> Optional[float]:
    pairs = []
    for r in rows:
        w = r.get(weight_key) or 0
        if w <= 0:
            continue
        v = transform(r) if transform else r.get(key)
        if v is None:
            continue
        pairs.append((float(v), float(w)))
    denom = sum(w for _, w in pairs)
    if denom <= 0:
        return None
    return sum(v * w for v, w in pairs) / denom


def _woba_value(r: dict) -> Optional[float]:
    """wOBA. Source-of-truth never populated by ingest (MLB Stats API
    doesn't expose it), so we always fall through to the proxy.

    Proxy: 0.55*OBP + 0.45*(AVG + 1.65*ISO)
      OBP captures walks (which AVG+ISO alone misses entirely).
      AVG+1.65*ISO is the standard slugging-weighted contribution.
      The 0.55/0.45 split was fit on a sample of MLB blocks where
      true wOBA is published — RMSE vs true ~0.012 wOBA points.

    If OBP is missing, falls back to walks-blind AVG + 1.65*ISO.
    """
    v = r.get("woba")
    if v is not None:
        return float(v)
    obp = r.get("obp")
    avg = r.get("avg")
    iso = r.get("iso")
    if obp is not None and (avg is not None or iso is not None):
        slug_part = float(avg or 0) + 1.65 * float(iso or 0)
        return 0.55 * float(obp) + 0.45 * slug_part
    # Walks-blind fallback
    if avg is not None or iso is not None:
        return float(avg or 0) + 1.65 * float(iso or 0)
    return None


def _fip_value(r: dict) -> Optional[float]:
    """FIP. Source-of-truth never populated by ingest, so we always
    compute from rate components.

    Real FIP identity (since K = k9*IP/9 etc):
        FIP = (13*HR + 3*BB - 2*K) / IP + cFIP
            = (13*hr9 + 3*bb9 - 2*k9) / 9 + cFIP

    cFIP normalizes league ERA; ~3.10 for MiLB across our era.
    Skips HBP (~negligible vs other terms).

    If k9/bb9/hr9 are absent, fall back to ERA (worse — includes luck).
    """
    v = r.get("fip")
    if v is not None:
        return float(v)
    k9 = r.get("k9"); bb9 = r.get("bb9"); hr9 = r.get("hr9")
    if k9 is not None and bb9 is not None and hr9 is not None:
        return (13.0*float(hr9) + 3.0*float(bb9) - 2.0*float(k9)) / 9.0 + 3.10
    era = r.get("era")
    return float(era) if era is not None else None


def _level_baseline(baselines: dict, side: str, level: str, stat: str) -> Optional[float]:
    bl = baselines.get(side, {})
    if level not in bl:
        return None
    return bl[level].get(stat)


def _age_baseline(baselines: dict, side: str, level: str) -> Optional[float]:
    key = "age_hit" if side == "hit" else "age_pit"
    return baselines.get(key, {}).get(level)


# ============================================================================
# Per-year aggregate
# ============================================================================

def _year_aggregate(
    rows: list[dict],
    year: int,
    baselines: dict,
) -> dict:
    """Aggregate one season into hitter + pitcher feature dicts."""
    matches = [r for r in rows if r.get("season_year") == year]

    # Determine the "primary level" of this season as the level with the
    # most PA (hitter) or IP (pitcher).
    levels_seen = set()
    for r in matches:
        lvl = (r.get("level") or "").upper()
        if lvl:
            levels_seen.add(lvl)

    out: dict[str, float] = {}

    # ---- Hitter side ----
    hit_rows = [r for r in matches if (r.get("pa") or 0) > 0]
    hit = {f: MISSING for f in HIT_PER_YEAR}
    primary_hit_level = None
    if hit_rows:
        total_pa = sum((r.get("pa") or 0) for r in hit_rows)
        hit["pa"] = float(total_pa)
        hit["woba"] = _miss(_weighted_avg(hit_rows, "woba", "pa", _woba_value))
        hit["iso"] = _miss(_weighted_avg(hit_rows, "iso", "pa"))
        hit["k_pct"] = _miss(_weighted_avg(hit_rows, "k_pct", "pa"))
        hit["bb_pct"] = _miss(_weighted_avg(hit_rows, "bb_pct", "pa"))
        hit["babip"] = _miss(_weighted_avg(hit_rows, "babip", "pa"))
        hit["obp"] = _miss(_weighted_avg(hit_rows, "obp", "pa"))
        hit["slg"] = _miss(_weighted_avg(hit_rows, "slg", "pa"))
        hit["avg"] = _miss(_weighted_avg(hit_rows, "avg", "pa"))
        total_hr = sum((r.get("home_runs") or 0) for r in hit_rows)
        total_sb = sum((r.get("stolen_bases") or 0) for r in hit_rows)
        hit["hr_per_pa"] = float(total_hr) / float(total_pa) if total_pa else MISSING
        hit["sb_per_pa"] = float(total_sb) / float(total_pa) if total_pa else MISSING
        # BB/K ratio: if k_pct positive
        if (not _is_missing(hit["bb_pct"])
                and not _is_missing(hit["k_pct"]) and hit["k_pct"] != 0):
            hit["bb_k_ratio"] = hit["bb_pct"] / hit["k_pct"]
        else:
            hit["bb_k_ratio"] = MISSING
        # Primary level by PA
        lvl_pa: dict[str, float] = {}
        for r in hit_rows:
            lvl = (r.get("level") or "").upper()
            lvl_pa[lvl] = lvl_pa.get(lvl, 0) + (r.get("pa") or 0)
        if lvl_pa:
            primary_hit_level = max(lvl_pa, key=lvl_pa.get)

        # vs-level baselines
        if primary_hit_level:
            for stat in ("woba", "iso", "k_pct", "bb_pct"):
                bl = _level_baseline(baselines, "hit", primary_hit_level, stat)
                key = f"{stat}_vs_level"
                if not _is_missing(hit[stat]) and bl is not None:
                    hit[key] = hit[stat] - bl
                else:
                    hit[key] = MISSING

    # ---- Pitcher side ----
    pit_rows = [r for r in matches if (r.get("ip") or 0) > 0]
    pit = {f: MISSING for f in PIT_PER_YEAR}
    primary_pit_level = None
    if pit_rows:
        total_ip = sum((r.get("ip") or 0) for r in pit_rows)
        pit["ip"] = float(total_ip)
        pit["era"] = _miss(_weighted_avg(pit_rows, "era", "ip"))
        pit["fip"] = _miss(_weighted_avg(pit_rows, "fip", "ip", _fip_value))
        pit["k9"] = _miss(_weighted_avg(pit_rows, "k9", "ip"))
        pit["bb9"] = _miss(_weighted_avg(pit_rows, "bb9", "ip"))
        pit["whip"] = _miss(_weighted_avg(pit_rows, "whip", "ip"))
        pit["hr9"] = _miss(_weighted_avg(pit_rows, "hr9", "ip"))
        pit["velo_avg"] = _miss(_weighted_avg(pit_rows, "velo_avg", "ip"))
        # K/BB ratio
        if (not _is_missing(pit["k9"])
                and not _is_missing(pit["bb9"]) and pit["bb9"] != 0):
            pit["k_bb_ratio"] = pit["k9"] / pit["bb9"]
        else:
            pit["k_bb_ratio"] = MISSING
        # Primary level by IP
        lvl_ip: dict[str, float] = {}
        for r in pit_rows:
            lvl = (r.get("level") or "").upper()
            lvl_ip[lvl] = lvl_ip.get(lvl, 0) + (r.get("ip") or 0)
        if lvl_ip:
            primary_pit_level = max(lvl_ip, key=lvl_ip.get)

        if primary_pit_level:
            for stat in ("era", "fip", "k9", "bb9"):
                bl = _level_baseline(baselines, "pit", primary_pit_level, stat)
                key = f"{stat}_vs_level"
                if not _is_missing(pit[stat]) and bl is not None:
                    pit[key] = pit[stat] - bl
                else:
                    pit[key] = MISSING

    # ---- Shared ----
    # Primary level = whichever side has the row; for two-way players pick
    # the one with more sample.
    primary_level = primary_hit_level or primary_pit_level
    if primary_hit_level and primary_pit_level:
        # heuristic: pick by which side had bigger sample (PA vs IP)
        if (hit["pa"] or 0) >= 3 * (pit["ip"] or 0):
            primary_level = primary_hit_level
        else:
            primary_level = primary_pit_level

    level_rank = LEVEL_RANK.get(primary_level, 0) if primary_level else 0
    # Age: prefer hitter side, else pitcher
    ages = [r.get("age_during_season") for r in matches
            if r.get("age_during_season") is not None]
    age = float(sum(ages) / len(ages)) if ages else None
    if age is not None and primary_level:
        side = "hit" if primary_hit_level == primary_level else "pit"
        age_bl = _age_baseline(baselines, side, primary_level)
        age_vs_level = (age - age_bl) if age_bl is not None else MISSING
    else:
        age_vs_level = MISSING

    highest_level_in_year = max(
        [LEVEL_RANK.get(lvl, 0) for lvl in levels_seen] or [0]
    )

    shared = {
        "level_rank": float(level_rank) if level_rank > 0 else MISSING,
        "age": float(age) if age is not None else MISSING,
        "age_vs_level": float(age_vs_level) if not _is_missing(age_vs_level) else MISSING,
        "n_levels_in_year": float(len(levels_seen)) if levels_seen else MISSING,
        "highest_level_in_year": float(highest_level_in_year) if highest_level_in_year > 0 else MISSING,
    }

    return {"hit": hit, "pit": pit, "shared": shared,
            "primary_level": primary_level,
            "primary_hit_level": primary_hit_level,
            "primary_pit_level": primary_pit_level}


def _miss(v) -> float:
    return float(v) if v is not None else MISSING


# ============================================================================
# Career-to-date and trajectory
# ============================================================================

def _career_to_date(
    rows: list[dict],
    as_of_year: int,
    baselines: dict,
    draft_year: Optional[int],
    birth_date: Optional[str] = None,
) -> dict:
    """Cumulative MiLB-career stats observed up to and including as_of_year."""
    past = [r for r in rows
            if r.get("season_year") is not None
            and r["season_year"] <= as_of_year]
    out = {f: MISSING for f in CAREER_TO_DATE_FEATS}

    if not past:
        out["career_milb_pa"] = 0.0
        out["career_milb_ip"] = 0.0
        out["career_milb_hr"] = 0.0
        out["career_milb_sb"] = 0.0
        out["career_milb_seasons"] = 0.0
        out["distinct_levels_played"] = 0.0
        out["reached_AA"] = 0.0
        out["reached_AAA"] = 0.0
        return out

    total_pa = sum((r.get("pa") or 0) for r in past)
    total_ip = sum((r.get("ip") or 0) for r in past)
    total_hr = sum((r.get("home_runs") or 0) for r in past)
    total_sb = sum((r.get("stolen_bases") or 0) for r in past)
    seasons = len({r.get("season_year") for r in past if r.get("season_year")})
    levels = {(r.get("level") or "").upper() for r in past
              if (r.get("level") or "").upper()}
    ranks = [LEVEL_RANK.get(l, 0) for l in levels]
    max_lvl = max(ranks) if ranks else 0
    min_lvl = min((r for r in ranks if r > 0), default=0)

    out["career_milb_pa"] = float(total_pa)
    out["career_milb_ip"] = float(total_ip)
    out["career_milb_hr"] = float(total_hr)
    out["career_milb_sb"] = float(total_sb)
    out["career_milb_seasons"] = float(seasons)
    out["distinct_levels_played"] = float(len(levels))
    out["max_level_ever"] = float(max_lvl) if max_lvl > 0 else MISSING
    # years_since_max_level_ever: unqualified (any appearance at max level).
    # Captures Hackenberg's case where his last AAA touch was 2024 (23 IP)
    # — qualified max stays at AA, but the model can still see "AAA was 2 yrs ago".
    if max_lvl > 0:
        max_lvl_strs = {l for l, r in LEVEL_RANK.items() if r == max_lvl}
        years_at_max = [r["season_year"] for r in past
                        if (r.get("level") or "").upper() in max_lvl_strs
                        and r.get("season_year") is not None]
        if years_at_max:
            out["years_since_max_level_ever"] = float(as_of_year - max(years_at_max))
        else:
            out["years_since_max_level_ever"] = MISSING
    else:
        out["years_since_max_level_ever"] = MISSING
    out["min_level_played"] = float(min_lvl) if min_lvl > 0 else MISSING
    out["reached_AA"] = 1.0 if "AA" in levels or "AAA" in levels else 0.0
    out["reached_AAA"] = 1.0 if "AAA" in levels else 0.0

    # ---- Qualified max-level features (filter out brief stints) ----
    # Aggregate IP/PA per (year, level) so a single short cameo doesn't lift
    # max_level_ever. A (year, level) qualifies for MAX if IP >= 30 or PA
    # >= 100; for BOTTOM-SINCE-MAX if IP >= 25 or PA >= 75.
    yr_lvl_stats: dict[tuple, dict] = {}
    for r in past:
        y = r.get("season_year")
        lvl = (r.get("level") or "").upper()
        if y is None or not lvl: continue
        key = (int(y), lvl)
        s = yr_lvl_stats.setdefault(key, {"ip": 0.0, "pa": 0})
        s["ip"] += (r.get("ip") or 0)
        s["pa"] += (r.get("pa") or 0)

    QUAL_IP, QUAL_PA = 30.0, 100
    BOT_IP, BOT_PA = 25.0, 75
    qualified_pairs = [(y, lvl, s) for (y, lvl), s in yr_lvl_stats.items()
                       if s["ip"] >= QUAL_IP or s["pa"] >= QUAL_PA]
    if qualified_pairs:
        # max_level_qualified = highest LEVEL_RANK among qualified (y, lvl)
        max_lvl_q = max(LEVEL_RANK.get(lvl, 0) for _, lvl, _ in qualified_pairs)
        out["max_level_qualified"] = float(max_lvl_q) if max_lvl_q > 0 else MISSING
        # First year that level was qualified
        first_year_at_q_max = min(y for y, lvl, _ in qualified_pairs
                                  if LEVEL_RANK.get(lvl, 0) == max_lvl_q)
        last_year_at_q_max = max(y for y, lvl, _ in qualified_pairs
                                 if LEVEL_RANK.get(lvl, 0) == max_lvl_q)
        out["years_since_max_level"] = float(as_of_year - last_year_at_q_max)
        # bottom_since: among (year, level) pairs in years > first_year_at_q_max
        # with IP>=25 or PA>=75, take the min LEVEL_RANK. If no such row exists,
        # default to the max itself (no regression observed).
        bot_pairs = [(y, lvl) for (y, lvl), s in yr_lvl_stats.items()
                     if y > first_year_at_q_max
                     and (s["ip"] >= BOT_IP or s["pa"] >= BOT_PA)
                     and LEVEL_RANK.get(lvl, 0) > 0]
        if bot_pairs:
            bot_rank = min(LEVEL_RANK.get(lvl, 0) for _, lvl in bot_pairs)
            out["bottom_since_max_level"] = float(bot_rank)
        else:
            out["bottom_since_max_level"] = float(max_lvl_q)
    else:
        # No level cleared the qualifying threshold yet.
        out["max_level_qualified"] = MISSING
        out["years_since_max_level"] = MISSING
        out["bottom_since_max_level"] = MISSING

    # ---- Tier 1: gap-year / missed-time signals ----
    # An "active" year has any PA>0 or IP>0. Look at years between first active
    # year and as_of_year - 1 (exclude current year; mid-season has 0 stats
    # legitimately during early-season scoring). Gaps within that span = missed.
    yrs_by_active = sorted({y for y, s in yr_lvl_stats.items() if s["ip"] > 0 or s["pa"] > 0}
                           if yr_lvl_stats else set())
    # yr_lvl_stats keyed by (y, lvl); take unique y where any (y, lvl) has activity
    active_years = sorted({y for (y, _), s in yr_lvl_stats.items()
                           if s["ip"] > 0 or s["pa"] > 0})
    if len(active_years) >= 2:
        span = list(range(active_years[0], (as_of_year - 1) + 1))
        missed = [y for y in span if y not in set(active_years)]
        out["had_lost_season"] = 1.0 if missed else 0.0
        out["seasons_missed_career"] = float(len(missed))
        # Consecutive active-year streak ending at the most recent active year
        streak = 1
        for i in range(len(active_years) - 1, 0, -1):
            if active_years[i] - active_years[i-1] == 1:
                streak += 1
            else:
                break
        out["consecutive_active_seasons"] = float(streak)
    elif len(active_years) == 1:
        out["had_lost_season"] = 0.0
        out["seasons_missed_career"] = 0.0
        out["consecutive_active_seasons"] = 1.0
    else:
        out["had_lost_season"] = MISSING
        out["seasons_missed_career"] = MISSING
        out["consecutive_active_seasons"] = MISSING

    # current_pa_vs_max_pa / current_ip_vs_max_ip
    # "Current" = stats at as_of_year. "Max" = max single-year total across past.
    pa_by_year = {}
    ip_by_year = {}
    for (y, _lvl), s in yr_lvl_stats.items():
        pa_by_year[y] = pa_by_year.get(y, 0) + s["pa"]
        ip_by_year[y] = ip_by_year.get(y, 0.0) + s["ip"]
    cur_pa = pa_by_year.get(as_of_year, 0)
    cur_ip = ip_by_year.get(as_of_year, 0.0)
    max_pa = max(pa_by_year.values()) if pa_by_year else 0
    max_ip = max(ip_by_year.values()) if ip_by_year else 0.0
    out["current_pa_vs_max_pa"] = (cur_pa / max_pa) if max_pa > 0 else MISSING
    out["current_ip_vs_max_ip"] = (cur_ip / max_ip) if max_ip > 0 else MISSING

    # (Tier 2 features computed below, after seas_hit/seas_pit/best_* are built)

    # ---- Tier 3: age_at_first_AA/AAA from birth_date; n_demotions ----
    def _first_year_at_levels(target):
        ys = [r["season_year"] for r in past
              if (r.get("level") or "").upper() in target]
        return min(ys) if ys else None
    fy_aa_set = _first_year_at_levels({"AA", "AAA"})
    fy_aaa_set = _first_year_at_levels({"AAA"})
    birth_year = None
    if birth_date:
        try:
            birth_year = int(str(birth_date)[:4])
        except Exception:
            birth_year = None
    out["age_at_first_AA"] = (
        float(fy_aa_set - birth_year) if (fy_aa_set and birth_year) else MISSING)
    out["age_at_first_AAA"] = (
        float(fy_aaa_set - birth_year) if (fy_aaa_set and birth_year) else MISSING)

    lvl_by_year = {}
    yrs_sorted = sorted({y for y, _ in (yr_lvl_stats.keys() if yr_lvl_stats else [])})
    for y in yrs_sorted:
        rows_y = [r for r in past if r.get("season_year") == y]
        if not rows_y: continue
        lvls_y = [(r.get("level") or "").upper() for r in rows_y
                  if (r.get("level") or "").upper()]
        lvl_by_year[y] = max((LEVEL_RANK.get(l, 0) for l in lvls_y), default=0)
    n_demotions = 0
    yrs_seq = sorted(lvl_by_year.keys())
    for i in range(1, len(yrs_seq)):
        prev_y, cur_y = yrs_seq[i-1], yrs_seq[i]
        prev_lvl = lvl_by_year.get(prev_y, 0)
        cur_lvl = lvl_by_year.get(cur_y, 0)
        if prev_lvl > 0 and cur_lvl > 0 and cur_lvl < prev_lvl:
            n_demotions += 1
    out["n_demotions_career"] = float(n_demotions)

    # ---- Tier 4: pitcher workload ----
    if ip_by_year:
        max_yr_ip = max(ip_by_year.values())
        out["career_max_ip_in_year"] = float(max_yr_ip)
        n_ip_seasons = sum(1 for v in ip_by_year.values() if v > 0)
        if n_ip_seasons > 0:
            n_above_50 = sum(1 for v in ip_by_year.values() if v >= 50)
            out["pct_seasons_above_50_ip"] = float(n_above_50) / n_ip_seasons
        else:
            out["pct_seasons_above_50_ip"] = MISSING
        # delta_ip yT vs yT-1
        if as_of_year in ip_by_year and (as_of_year - 1) in ip_by_year:
            out["delta_ip_yT_vs_yT_minus_1"] = float(
                ip_by_year[as_of_year] - ip_by_year[as_of_year - 1])
        else:
            out["delta_ip_yT_vs_yT_minus_1"] = MISSING
    else:
        out["career_max_ip_in_year"] = MISSING
        out["pct_seasons_above_50_ip"] = MISSING
        out["delta_ip_yT_vs_yT_minus_1"] = MISSING

    # First year at each level
    def first_year_at(target_levels: set[str]) -> Optional[int]:
        ys = [r["season_year"] for r in past
              if (r.get("level") or "").upper() in target_levels]
        return min(ys) if ys else None

    fya = first_year_at({"A", "A+", "AA", "AAA"})
    fy_aa = first_year_at({"AA", "AAA"})
    fy_aaa = first_year_at({"AAA"})

    if draft_year is not None:
        out["years_to_A_or_higher"] = float(fya - draft_year) if fya else MISSING
        out["years_to_AA"] = float(fy_aa - draft_year) if fy_aa else MISSING
        out["years_to_AAA"] = float(fy_aaa - draft_year) if fy_aaa else MISSING

    # Best-so-far rate stats (PA-weighted across the season's primary row);
    # use simple per-season aggregates re-computed cheaply.
    by_year: dict[int, dict] = {}
    for r in past:
        y = r["season_year"]
        if y is None:
            continue
        by_year.setdefault(y, []).append(r)

    seas_hit = []
    seas_pit = []
    for y, ys in by_year.items():
        hit_rs = [r for r in ys if (r.get("pa") or 0) > 0]
        pit_rs = [r for r in ys if (r.get("ip") or 0) > 0]
        if hit_rs:
            pa = sum((r.get("pa") or 0) for r in hit_rs)
            seas_hit.append({
                "year": y, "pa": pa,
                "woba": _weighted_avg(hit_rs, "woba", "pa", _woba_value),
                "iso": _weighted_avg(hit_rs, "iso", "pa"),
                "k_pct": _weighted_avg(hit_rs, "k_pct", "pa"),
                "bb_pct": _weighted_avg(hit_rs, "bb_pct", "pa"),
                "obp": _weighted_avg(hit_rs, "obp", "pa"),
                "slg": _weighted_avg(hit_rs, "slg", "pa"),
            })
        if pit_rs:
            ip = sum((r.get("ip") or 0) for r in pit_rs)
            seas_pit.append({
                "year": y, "ip": ip,
                "era": _weighted_avg(pit_rs, "era", "ip"),
                "fip": _weighted_avg(pit_rs, "fip", "ip", _fip_value),
                "k9": _weighted_avg(pit_rs, "k9", "ip"),
                "bb9": _weighted_avg(pit_rs, "bb9", "ip"),
                "whip": _weighted_avg(pit_rs, "whip", "ip"),
            })

    def _best(seq, key, mode="max"):
        vals = [s[key] for s in seq if s.get(key) is not None]
        if not vals:
            return MISSING
        return float(max(vals) if mode == "max" else min(vals))

    out["best_woba"] = _best(seas_hit, "woba", "max")
    out["best_iso"] = _best(seas_hit, "iso", "max")
    out["best_obp"] = _best(seas_hit, "obp", "max")
    out["best_slg"] = _best(seas_hit, "slg", "max")
    out["best_k_pct"] = _best(seas_hit, "k_pct", "min")    # lower is better for batters
    out["best_bb_pct"] = _best(seas_hit, "bb_pct", "max")
    out["best_era"] = _best(seas_pit, "era", "min")
    out["best_fip"] = _best(seas_pit, "fip", "min")
    out["best_k9"] = _best(seas_pit, "k9", "max")
    out["best_bb9"] = _best(seas_pit, "bb9", "min")
    out["best_whip"] = _best(seas_pit, "whip", "min")

    # ---- Tier 2: current vs best (relies on seas_hit/pit + best_* above) ----
    def _cur_year_agg(seas, year):
        for s in seas:
            if s["year"] == year: return s
        return None
    cur_hit = _cur_year_agg(seas_hit, as_of_year)
    cur_pit = _cur_year_agg(seas_pit, as_of_year)
    def _ratio(cur_val, best_val):
        if cur_val is None: return MISSING
        if _is_missing(best_val): return MISSING
        if best_val == 0: return MISSING
        return float(cur_val) / float(best_val)
    out["current_woba_vs_best_woba"] = (
        _ratio(cur_hit.get("woba"), out["best_woba"]) if cur_hit else MISSING)
    out["current_bb9_vs_best_bb9"] = (
        _ratio(cur_pit.get("bb9"), out["best_bb9"]) if cur_pit else MISSING)
    out["current_k9_vs_best_k9"] = (
        _ratio(cur_pit.get("k9"), out["best_k9"]) if cur_pit else MISSING)
    out["current_era_vs_best_era"] = (
        _ratio(cur_pit.get("era"), out["best_era"]) if cur_pit else MISSING)

    # PA/IP at AAA / AA / max level
    def _sum(rs, key):
        return float(sum((r.get(key) or 0) for r in rs))
    pa_aaa = _sum([r for r in past if (r.get("level") or "").upper() == "AAA"], "pa")
    ip_aaa = _sum([r for r in past if (r.get("level") or "").upper() == "AAA"], "ip")
    pa_aa = _sum([r for r in past if (r.get("level") or "").upper() == "AA"], "pa")
    ip_aa = _sum([r for r in past if (r.get("level") or "").upper() == "AA"], "ip")
    out["pa_at_AAA_career"] = pa_aaa
    out["ip_at_AAA_career"] = ip_aaa
    out["pa_at_AA_career"] = pa_aa
    out["ip_at_AA_career"] = ip_aa
    out["pct_pa_at_AAA"] = (pa_aaa / total_pa) if total_pa else MISSING
    out["pct_ip_at_AAA"] = (ip_aaa / total_ip) if total_ip else MISSING

    if max_lvl > 0:
        # Find the level string for max_lvl
        max_level_strs = {l for l, r in LEVEL_RANK.items() if r == max_lvl}
        at_max = [r for r in past
                  if (r.get("level") or "").upper() in max_level_strs]
        out["pa_at_max_level_career"] = _sum(at_max, "pa")
        out["ip_at_max_level_career"] = _sum(at_max, "ip")
    else:
        out["pa_at_max_level_career"] = MISSING
        out["ip_at_max_level_career"] = MISSING

    return out


def _trajectory_block(
    rows: list[dict],
    as_of_year: int,
    year_aggs: dict[int, dict],
    draft_year: Optional[int],
    career: dict,
) -> dict:
    out = {f: MISSING for f in TRAJECTORY_FEATS}

    years_in_pro = (as_of_year - draft_year) if draft_year is not None else None
    max_lvl = career.get("max_level_ever")
    if (isinstance(max_lvl, float) and not _is_missing(max_lvl)
            and years_in_pro and years_in_pro > 0):
        out["promotion_velocity"] = (max_lvl - 1.0) / float(years_in_pro)

    # level changes year-over-year
    def _level_for(y):
        agg = year_aggs.get(y)
        if not agg:
            return None
        v = agg["shared"]["level_rank"]
        return None if _is_missing(v) else v

    l_yT = _level_for(as_of_year)
    l_y1 = _level_for(as_of_year - 1)
    l_y2 = _level_for(as_of_year - 2)
    if l_yT is not None and l_y1 is not None:
        out["level_change_yT_vs_y1"] = l_yT - l_y1
    if l_y1 is not None and l_y2 is not None:
        out["level_change_y1_vs_y2"] = l_y1 - l_y2
    if l_yT is not None and l_y1 is not None and l_y2 is not None:
        out["promotion_acceleration"] = (l_yT - l_y1) - (l_y1 - l_y2)

    # Years stuck at max level: count consecutive years (ending at as_of_year)
    # where season level == max level ever.
    if isinstance(max_lvl, float) and not _is_missing(max_lvl):
        max_lvl_int = int(max_lvl)
        stuck = 0
        for back in range(0, 25):
            y = as_of_year - back
            agg = year_aggs.get(y)
            if not agg:
                break
            lv = agg["shared"]["level_rank"]
            if _is_missing(lv):
                break
            if int(lv) == max_lvl_int:
                stuck += 1
            else:
                break
        out["years_stuck_at_max_level"] = float(stuck)

    # Years at current level (level at as_of_year)
    if l_yT is not None:
        cur = int(l_yT)
        run = 0
        for back in range(0, 25):
            y = as_of_year - back
            agg = year_aggs.get(y)
            if not agg:
                break
            lv = agg["shared"]["level_rank"]
            if _is_missing(lv):
                break
            if int(lv) == cur:
                run += 1
            else:
                break
        out["years_at_current_level"] = float(run)

    # Repeat-level slope: among all years observed (up to as_of_year), find
    # any (level, year_count >= 2) pair and compute slope of wOBA / FIP.
    by_level_hit: dict[str, list[tuple[int, float]]] = {}
    by_level_pit: dict[str, list[tuple[int, float]]] = {}
    for y, agg in year_aggs.items():
        if y > as_of_year:
            continue
        lvl_h = agg.get("primary_hit_level")
        if lvl_h and not _is_missing(agg["hit"].get("woba", MISSING)):
            by_level_hit.setdefault(lvl_h, []).append((y, agg["hit"]["woba"]))
        lvl_p = agg.get("primary_pit_level")
        if lvl_p and not _is_missing(agg["pit"].get("fip", MISSING)):
            by_level_pit.setdefault(lvl_p, []).append((y, agg["pit"]["fip"]))

    def _slope(pairs):
        if len(pairs) < 2:
            return None
        xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
        mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
        num = sum((x-mx)*(y-my) for x, y in pairs)
        den = sum((x-mx)**2 for x in xs)
        return num/den if den else None

    # Use the highest-level repeat
    if by_level_hit:
        best_level = max(by_level_hit.keys(), key=lambda l: LEVEL_RANK.get(l, 0))
        s = _slope(by_level_hit[best_level])
        if s is not None:
            out["repeat_level_woba_slope"] = float(s)
    if by_level_pit:
        best_level = max(by_level_pit.keys(), key=lambda l: LEVEL_RANK.get(l, 0))
        s = _slope(by_level_pit[best_level])
        if s is not None:
            out["repeat_level_fip_slope"] = float(s)

    return out


# ============================================================================
# Main builder
# ============================================================================

def _delta(a: float, b: float) -> float:
    if _is_missing(a) or _is_missing(b):
        return MISSING
    return a - b


def _accel(yT: float, y1: float, y2: float) -> float:
    d1 = _delta(yT, y1)
    d2 = _delta(y1, y2)
    if _is_missing(d1) or _is_missing(d2):
        return MISSING
    return d1 - d2


def build_scouting_features(
    prospect: dict,
    season_stats: list[dict],
    as_of_year: int,
    baselines: dict,
    *,
    milb_only: bool = True,
) -> np.ndarray:
    """Build the full scouting feature vector (length N_FEATURES)."""

    if milb_only:
        season_stats = [s for s in season_stats
                        if (s.get("level") or "").upper() != "MLB"]
    # No-lookahead guard (v2.1): drop any season AFTER the snapshot. The
    # windowed features filter season_year<=as_of locally, but has_any_milb_data
    # / has_hitting / has_pitching and the years_in_pro / years_in_current_system
    # fallbacks scan this list RAW — without this they leak whether (and when) a
    # player will eventually play, and future position conversions. At inference
    # no future rows exist, so this also removes train/serve skew.
    season_stats = [s for s in season_stats
                    if (s.get("season_year") or 0) <= as_of_year]

    pos = (prospect.get("primary_position") or "").upper()
    is_pitcher = pos in PITCHER_POS or bool(prospect.get("is_pitcher"))
    origin = (prospect.get("origin") or "").lower()
    is_college = any(t in origin for t in (
        "univ", "college", "state", "u of", "tech", "institute"
    ))
    draft_year = prospect.get("draft_year")
    bonus = prospect.get("signing_bonus_usd")

    # v1.9: biometrics + bonus-vs-slot
    h_in = prospect.get("height_inches")
    w_lbs = prospect.get("weight_lbs")
    bmi = MISSING
    if h_in and w_lbs and float(h_in) > 0:
        bmi = float(w_lbs) * 703.0 / (float(h_in) ** 2)
    bats = (prospect.get("bats") or "").upper()
    throws = (prospect.get("throws") or "").upper()
    pick_value = prospect.get("pick_value_usd")
    bonus_vs_slot = MISSING
    if bonus and pick_value and float(pick_value) > 0:
        bonus_vs_slot = float(bonus) / float(pick_value)

    # ---- A. Pedigree ----
    pedigree = {
        "is_pitcher": float(is_pitcher),
        "is_international": float(prospect.get("is_international") or 0),
        "is_college_draftee": float(is_college),
        "is_drafted": float(draft_year is not None),
        "is_premium_position": float(pos in PREMIUM_POSITIONS),
        "is_catcher": float(pos == "C"),
        "is_shortstop": float(pos == "SS"),
        "is_center_field": float(pos == "CF"),
        "draft_round": float(prospect["draft_round"]) if prospect.get("draft_round") is not None else MISSING,
        "draft_pick": float(prospect["draft_pick"]) if prospect.get("draft_pick") is not None else MISSING,
        "log_signing_bonus": float(math.log1p(bonus)) if bonus and bonus > 0 else MISSING,
        "has_signing_bonus": 1.0 if bonus and bonus > 0 else 0.0,
        "age_at_signing": float(prospect["age_at_signing"]) if prospect.get("age_at_signing") is not None else MISSING,
        "height_inches": float(h_in) if h_in is not None else MISSING,
        "weight_lbs": float(w_lbs) if w_lbs is not None else MISSING,
        "bmi": float(bmi) if not _is_missing(bmi) else MISSING,
        "bats_L": 1.0 if bats == "L" else (0.0 if bats in ("R", "S") else MISSING),
        "bats_S": 1.0 if bats == "S" else (0.0 if bats in ("R", "L") else MISSING),
        "throws_L": 1.0 if throws == "L" else (0.0 if throws in ("R", "S") else MISSING),
        "log_pick_value": float(math.log1p(float(pick_value))) if pick_value and float(pick_value) > 0 else MISSING,
        "bonus_vs_slot": bonus_vs_slot,
    }

    # v1.10: BBC Top-100 prospect rankings, as-of-year aware.
    # prospect["_top100_rankings"] is a list of (year, rank, source) tuples
    # attached at panel-build time. We only use entries with year <= as_of_year
    # so the feature builder never leaks future rankings into a snapshot.
    rankings = prospect.get("_top100_rankings") or []
    past_ranks = [(y, r) for (y, r, *_rest) in rankings
                  if y is not None and r is not None and int(y) <= as_of_year]
    if past_ranks:
        ranks_only = [int(r) for (_y, r) in past_ranks]
        years_only = [int(y) for (y, _r) in past_ranks]
        best_rank = min(ranks_only)
        latest_year = max(years_only)
        recent_rank = next(int(r) for (y, r) in past_ranks if y == latest_year)
        first_year = min(years_only)
        pedigree["ever_top100"] = 1.0
        pedigree["best_top100_rank"] = float(best_rank)
        pedigree["recent_top100_rank"] = float(recent_rank)
        pedigree["times_top100"] = float(len(past_ranks))
        pedigree["years_since_first_top100"] = float(as_of_year - first_year)
        pedigree["log_best_top100_rank"] = float(math.log1p(best_rank))
    else:
        pedigree["ever_top100"] = 0.0
        pedigree["best_top100_rank"] = MISSING
        pedigree["recent_top100_rank"] = MISSING
        pedigree["times_top100"] = 0.0
        pedigree["years_since_first_top100"] = MISSING
        pedigree["log_best_top100_rank"] = MISSING

    # v2.0c: TBC ORG rankings, as-of-year aware. prospect["_org_rankings"] is a
    # list of (year, org_rank) attached at panel-build time. Collapse to one
    # best (min) org rank per year, using only years <= as_of_year so no future
    # ranking leaks into a snapshot.
    org_rankings = prospect.get("_org_rankings") or []
    org_by_year: dict[int, int] = {}
    for (y, rk) in org_rankings:
        if y is None or rk is None:
            continue
        y = int(y)
        if y > as_of_year:
            continue
        rk = int(rk)
        if y not in org_by_year or rk < org_by_year[y]:
            org_by_year[y] = rk
    if org_by_year:
        years_sorted = sorted(org_by_year)
        best_org = min(org_by_year.values())
        recent_year = years_sorted[-1]
        recent_org = org_by_year[recent_year]
        first_year = years_sorted[0]
        pedigree["ever_org_ranked"] = 1.0
        pedigree["best_org_rank"] = float(best_org)
        pedigree["recent_org_rank"] = float(recent_org)
        pedigree["times_org_ranked"] = float(len(org_by_year))
        pedigree["years_since_first_org_ranked"] = float(as_of_year - first_year)
        pedigree["log_best_org_rank"] = float(math.log1p(best_org))
        # Trend anchored to the most recent ranked year (+ = climbed toward #1).
        prev1 = org_by_year.get(recent_year - 1)
        prev2 = org_by_year.get(recent_year - 2)
        pedigree["org_rank_trend_1y"] = float(prev1 - recent_org) if prev1 is not None else MISSING
        pedigree["org_rank_trend_2y"] = float(prev2 - recent_org) if prev2 is not None else MISSING
    else:
        pedigree["ever_org_ranked"] = 0.0
        pedigree["best_org_rank"] = MISSING
        pedigree["recent_org_rank"] = MISSING
        pedigree["times_org_ranked"] = 0.0
        pedigree["years_since_first_org_ranked"] = MISSING
        pedigree["log_best_org_rank"] = MISSING
        pedigree["org_rank_trend_1y"] = MISSING
        pedigree["org_rank_trend_2y"] = MISSING

    # ---- B. Per-year aggregates (compute for yT, y1, y2) ----
    year_aggs: dict[int, dict] = {}
    for lag in range(WINDOW):
        y = as_of_year - lag
        year_aggs[y] = _year_aggregate(season_stats, y, baselines)

    # ---- C. Career-to-date ----
    career = _career_to_date(season_stats, as_of_year, baselines, draft_year,
                             birth_date=prospect.get("birth_date"))

    # ---- D. Trajectory ----
    trajectory = _trajectory_block(season_stats, as_of_year, year_aggs,
                                   draft_year, career)

    # ---- E. Deltas (yT vs y1) ----
    yT = year_aggs[as_of_year]
    y1 = year_aggs[as_of_year - 1]
    y2 = year_aggs[as_of_year - 2]

    deltas: dict[str, float] = {}
    # Hitter deltas
    for stat in ("woba", "iso", "k_pct", "bb_pct", "babip",
                 "obp", "slg", "avg", "hr_per_pa", "sb_per_pa"):
        deltas[f"delta_{stat}"] = _delta(yT["hit"][stat], y1["hit"][stat])
    deltas["delta_pa"] = _delta(yT["hit"]["pa"], y1["hit"]["pa"])
    for stat in ("woba", "iso", "k_pct", "bb_pct"):
        deltas[f"delta_{stat}_vs_level"] = _delta(
            yT["hit"][f"{stat}_vs_level"], y1["hit"][f"{stat}_vs_level"]
        )
    # Pitcher deltas
    for stat in ("era", "fip", "k9", "bb9", "whip", "hr9"):
        deltas[f"delta_{stat}"] = _delta(yT["pit"][stat], y1["pit"][stat])
    deltas["delta_ip"] = _delta(yT["pit"]["ip"], y1["pit"]["ip"])
    for stat in ("era", "fip", "k9", "bb9"):
        deltas[f"delta_{stat}_vs_level"] = _delta(
            yT["pit"][f"{stat}_vs_level"], y1["pit"][f"{stat}_vs_level"]
        )
    # Shared deltas
    deltas["delta_age"] = _delta(yT["shared"]["age"], y1["shared"]["age"])
    deltas["delta_age_vs_level"] = _delta(yT["shared"]["age_vs_level"],
                                          y1["shared"]["age_vs_level"])

    # ---- F. Acceleration (yT-y1) - (y1-y2) ----
    accel: dict[str, float] = {}
    for stat in ("woba", "iso", "k_pct", "bb_pct", "obp", "slg"):
        accel[f"accel_{stat}"] = _accel(yT["hit"][stat], y1["hit"][stat], y2["hit"][stat])
    for stat in ("era", "fip", "k9", "bb9"):
        accel[f"accel_{stat}"] = _accel(yT["pit"][stat], y1["pit"][stat], y2["pit"][stat])
    accel["accel_level"] = _accel(yT["shared"]["level_rank"],
                                  y1["shared"]["level_rank"],
                                  y2["shared"]["level_rank"])

    # ---- H. Window summary ----
    window_years = {as_of_year - i for i in range(WINDOW)}
    in_window = [s for s in season_stats if s.get("season_year") in window_years]
    n_years_obs = len({s.get("season_year") for s in in_window})
    max_lvl_window = max(
        (LEVEL_RANK.get((s.get("level") or "").upper(), 0) for s in in_window),
        default=0,
    )
    pa_window = sum((s.get("pa") or 0) for s in in_window)
    ip_window = sum((s.get("ip") or 0) for s in in_window)

    # Age at as_of_year: prefer observed in current year, then derive from
    # birth_date, then back-propagate from any observed age + year offset,
    # then fall back to the draft-year heuristic for drafted players.
    # Critical for IFAs (no draft_year) at as_of_years where they have no
    # current-year stat row yet -- otherwise age_at_as_of stays MISSING and
    # the inference-time feature aging silently no-ops on them.
    age_at_as_of = MISSING
    same_year = [s for s in season_stats if s.get("season_year") == as_of_year]
    ages = [s.get("age_during_season") for s in same_year
            if s.get("age_during_season") is not None]
    if ages:
        age_at_as_of = float(sum(ages) / len(ages))
    else:
        bd = prospect.get("birth_date")
        if bd:
            try:
                # birth_date is "YYYY-MM-DD" -- midyear approximation
                bd_year = int(str(bd)[:4])
                age_at_as_of = float(as_of_year - bd_year)
            except (ValueError, TypeError):
                pass
        if _is_missing(age_at_as_of):
            # Back-propagate from any observed age
            any_age_rows = [
                (s.get("season_year"), s.get("age_during_season"))
                for s in season_stats
                if s.get("season_year") is not None
                and s.get("age_during_season") is not None
            ]
            if any_age_rows:
                yr, ag = max(any_age_rows, key=lambda t: t[0])
                age_at_as_of = float(ag + (as_of_year - yr))
        if _is_missing(age_at_as_of) and draft_year is not None:
            age_at_as_of = (21.0 if is_college else 19.0) + max(0, as_of_year - draft_year)

    age_at_as_of_vs_level = MISSING
    primary_level_T = yT.get("primary_level")
    if not _is_missing(age_at_as_of) and primary_level_T:
        side = "hit" if yT.get("primary_hit_level") == primary_level_T else "pit"
        bl = _age_baseline(baselines, side, primary_level_T)
        if bl is not None:
            age_at_as_of_vs_level = age_at_as_of - bl

    # Years in current system (org). Anchor on the org from same_year rows
    # if present; otherwise fall back to the latest year with org info
    # (critical for as_of_year > latest-stat-year, e.g. mid-season 2026
    # grading of IFAs whose 2026 row has not landed yet).
    years_in_current_system = MISSING
    org_rows = same_year
    if not org_rows:
        # Latest year with at least one org-tagged row
        years_with_org = sorted({s.get("season_year") for s in season_stats
                                 if s.get("season_year") is not None
                                 and s.get("org")},
                                reverse=True)
        if years_with_org:
            latest = years_with_org[0]
            org_rows = [s for s in season_stats
                        if s.get("season_year") == latest]
    if org_rows:
        org_counts: dict[str, float] = {}
        for s in org_rows:
            org = s.get("org") or ""
            if not org:
                continue
            org_counts[org] = org_counts.get(org, 0) + (s.get("pa") or 0) + (s.get("ip") or 0)
        if org_counts:
            anchor = max(org_counts, key=org_counts.get)
            run = 0
            for back in range(0, 25):
                y = as_of_year - back
                rs_y = [s for s in season_stats if s.get("season_year") == y]
                if not rs_y:
                    break
                orgs_y = {s.get("org") for s in rs_y if s.get("org")}
                if anchor in orgs_y:
                    run += 1
                else:
                    break
            years_in_current_system = float(run)

    has_any_milb_data = float(bool(season_stats))
    has_hitting = float(any((s.get("pa") or 0) > 0 for s in season_stats))
    has_pitching = float(any((s.get("ip") or 0) > 0 for s in season_stats))

    window_summary = {
        "n_years_observed_in_window": float(n_years_obs),
        "max_level_seen_in_window": float(max_lvl_window) if max_lvl_window > 0 else MISSING,
        "pa_in_window": float(pa_window),
        "ip_in_window": float(ip_window),
        # years_in_pro: drafted -> as_of - draft_year; IFA -> as_of - signing_year
        # (international_signing_year), then fall back to first observed
        # MiLB season year. Without this, IFAs have years_in_pro=NaN and
        # the inference-time aging never advances time-scalar features.
        "years_in_pro": (
            float(as_of_year - draft_year) if draft_year is not None
            else (
                float(as_of_year - int(prospect["international_signing_year"]))
                if prospect.get("international_signing_year") is not None
                else (
                    float(as_of_year - min(
                        s["season_year"] for s in season_stats
                        if s.get("season_year") is not None
                    ))
                    if any(s.get("season_year") is not None for s in season_stats)
                    else MISSING
                )
            )
        ),
        "age_at_as_of": age_at_as_of,
        "age_at_as_of_vs_level": age_at_as_of_vs_level,
        "years_in_current_system": years_in_current_system,
        "has_any_milb_data": has_any_milb_data,
        "has_pitching_data": has_pitching,
        "has_hitting_data": has_hitting,
    }

    # ---- Assemble in canonical FEATURE_NAMES order ----
    vec_dict: dict[str, float] = {}
    vec_dict.update(pedigree)
    for lag in range(WINDOW):
        sfx = "_yT" if lag == 0 else f"_y{lag}"
        agg = year_aggs[as_of_year - lag]
        for f in HIT_PER_YEAR:
            vec_dict[f + sfx] = agg["hit"][f]
        for f in PIT_PER_YEAR:
            vec_dict[f + sfx] = agg["pit"][f]
        for f in SHARED_PER_YEAR:
            vec_dict[f + sfx] = agg["shared"][f]
    vec_dict.update(career)
    vec_dict.update(trajectory)
    vec_dict.update(deltas)
    vec_dict.update(accel)
    vec_dict.update(window_summary)
    # point-in-time scouting grades (season <= as_of_year), NaN where absent
    vec_dict.update(scouting_grade_dict(prospect.get("player_id"), as_of_year))

    vec = np.array([vec_dict[name] for name in FEATURE_NAMES], dtype=np.float64)
    assert vec.shape == (N_FEATURES,), (vec.shape, N_FEATURES)
    return vec


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--compute-baselines", action="store_true")
    parser.add_argument("--out", default="baselines/milb_baselines.json")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Build features for a few prospects to sanity-check")
    args = parser.parse_args()

    db = ProspectDB(args.db)

    if args.compute_baselines:
        print(f"[scouting] computing baselines from {args.db}...")
        bl = compute_baselines(db)
        save_baselines(bl, args.out)
        print(f"[scouting] saved -> {args.out}")
        print(f"  hit levels: {sorted(bl['hit'].keys())}")
        print(f"  pit levels: {sorted(bl['pit'].keys())}")
        def _f(v, fmt=".3f"):
            return format(v, fmt) if v is not None else "n/a"
        for lvl in sorted(bl["hit"]):
            d = bl["hit"][lvl]
            print(f"  hit[{lvl}]: woba={_f(d.get('woba'))}  "
                  f"k_pct={_f(d.get('k_pct'))}  bb_pct={_f(d.get('bb_pct'))}  "
                  f"age={_f(bl['age_hit'].get(lvl), '.1f')}")
        for lvl in sorted(bl["pit"]):
            d = bl["pit"][lvl]
            print(f"  pit[{lvl}]: era={_f(d.get('era'), '.2f')}  "
                  f"fip={_f(d.get('fip'), '.2f')}  k9={_f(d.get('k9'), '.2f')}  "
                  f"age={_f(bl['age_pit'].get(lvl), '.1f')}")

    if args.smoke_test:
        baselines = load_baselines(args.out)
        with db._connect() as conn:
            ps = [dict(r) for r in conn.execute(
                "SELECT * FROM prospects WHERE draft_year IN (2018, 2019) "
                "AND draft_round = 1 ORDER BY draft_pick LIMIT 3").fetchall()]
            stats = [dict(r) for r in conn.execute("SELECT * FROM season_stats").fetchall()]
        by_pid: dict[str, list] = {}
        for s in stats:
            by_pid.setdefault(s["player_id"], []).append(s)
        print(f"\n[scouting] N_FEATURES = {N_FEATURES}")
        for p in ps:
            v = build_scouting_features(p, by_pid.get(p["player_id"], []),
                                        as_of_year=2022, baselines=baselines)
            n_missing = int(np.isnan(v).sum())
            print(f"  {p['name']:<28} draft={p['draft_year']}  "
                  f"shape={v.shape}  missing={n_missing}/{N_FEATURES}")


if __name__ == "__main__":
    main()
