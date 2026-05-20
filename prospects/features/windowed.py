"""
prospects/features/windowed.py
================================

Feature builder for the random-year-sample training design.

For each prospect we pick one random "as-of" year drawn uniformly from the
years they appear in season_stats (or, if no stats, draft_year). Features:

  Pedigree (static):
      - is_pitcher, draft_round, draft_pick, log_signing_bonus, has_bonus,
        is_international, is_premium_position, is_college_draftee

  Per-year stats for {as_of_year, as_of_year-1, as_of_year-2}:
      - Hitter: pa, woba_proxy, iso, k_pct, bb_pct, level_rank
      - Pitcher: ip, era, k9, bb9, fip, level_rank
      - missing -> -1.0 sentinel

  Trajectory (derived from window):
      - n_years_observed, max_level_seen, age_at_as_of

The sentinel is -1.0 because rate stats are non-negative on the real domain;
gradient-boosted trees pick up the discontinuity at the sentinel cleanly.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from prospects.storage import ProspectDB


LEVEL_RANK = {
    "DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
    "A-": 2, "A": 3, "A+": 4,
    "AA": 5, "AAA": 6,
    "MLB": 7,
    "NCAA-D1": 3, "NCAA-D2": 2, "NCAA-D3": 2,
}

PREMIUM_POSITIONS = {"SS", "C", "CF"}
PITCHER_POS = {"P", "RHP", "LHP", "SP", "RP"}

MISSING = -1.0
WINDOW = 3  # last 3 years inclusive of as_of_year

# Static pedigree feature names (8)
PEDIGREE_FEATS = [
    "is_pitcher",
    "draft_round",
    "draft_pick",
    "log_signing_bonus",
    "has_signing_bonus",
    "is_international",
    "is_premium_position",
    "is_college_draftee",
]

# Per-year hitter feature names (5 stats + level_rank)
HIT_FEATS = [
    "pa", "woba_proxy", "iso", "k_pct", "bb_pct",
    "avg", "obp", "slg",
    "hr_per_pa", "sb_per_pa",
    "level_rank_h",
]
PIT_FEATS = [
    "ip", "era", "k9", "bb9", "fip",
    "whip", "hr9",
    "level_rank_p",
]


DELTA_FEATS = [
    # YoY change from y1 -> yT. Positive = better unless noted.
    "delta_woba",
    "delta_iso",
    "delta_k_pct",      # NEGATIVE direction = better (less K)
    "delta_bb_pct",
    "delta_obp",
    "delta_slg",
    "delta_level_h",
    "delta_era",        # NEGATIVE = better
    "delta_k9",
    "delta_bb9",        # NEGATIVE = better
    "delta_fip",        # NEGATIVE = better
    "delta_level_p",
]


def _build_feature_names() -> list[str]:
    names = list(PEDIGREE_FEATS)
    for lag in range(WINDOW):
        suffix = f"_y{lag}" if lag > 0 else "_yT"  # _yT = as_of year, _y1 = year-1
        for f in HIT_FEATS:
            names.append(f + suffix)
        for f in PIT_FEATS:
            names.append(f + suffix)
    names += list(DELTA_FEATS)
    names += [
        "n_years_observed_in_window",
        "max_level_seen_in_window",
        # `as_of_year` removed in v0.8: created temporal leakage via the
        # future-only mask (positive labels exited the training distribution
        # after their trigger year, leaving recent calendar years correlated
        # with non-establishment). years_in_pro + age_at_as_of carry the
        # actual career-stage information player-relatively.
        "years_in_pro",
        "age_at_as_of",
        "years_in_current_system",
    ]
    return names


FEATURE_NAMES = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)


def _woba_proxy(row: dict) -> Optional[float]:
    """Simple wOBA-ish proxy that exists for both MLB and MiLB rows.
    Falls back to OBP + 0.5*ISO if wOBA absent."""
    woba = row.get("woba")
    if woba is not None:
        return float(woba)
    obp = row.get("obp")
    iso = row.get("iso")
    if obp is None and iso is None:
        return None
    return float((obp or 0)) + 0.5 * float((iso or 0))


def _fip_proxy(row: dict) -> Optional[float]:
    """FIP if available, else ERA as a fallback."""
    fip = row.get("fip")
    if fip is not None:
        return float(fip)
    era = row.get("era")
    return float(era) if era is not None else None


def _stats_for_year(rows: list[dict], year: int, is_pitcher: bool) -> list[float]:
    """Aggregate rows for one year into the 6 per-year features. Multiple
    levels in same year are collapsed by max PA/IP (= best snapshot)."""
    matches = [r for r in rows if r.get("season_year") == year]
    if not matches:
        return [MISSING] * len(HIT_FEATS) + [MISSING] * len(PIT_FEATS)

    # ---- Hitter aggregate (weighted by PA) ----
    hit_rows = [r for r in matches if (r.get("pa") or 0) > 0]
    if hit_rows:
        total_pa = sum((r.get("pa") or 0) for r in hit_rows)
        def wavg(key, transform=None):
            tf = transform if transform is not None else (lambda r, _k=key: r.get(_k))
            vals = [(tf(r), r.get("pa") or 0)
                    for r in hit_rows if tf(r) is not None]
            denom = sum(w for _, w in vals)
            return (sum(v * w for v, w in vals) / denom) if denom > 0 else None
        h_pa = float(total_pa)
        h_woba = wavg("woba", _woba_proxy)
        h_iso = wavg("iso")
        h_k = wavg("k_pct")
        h_bb = wavg("bb_pct")
        h_avg = wavg("avg")
        h_obp = wavg("obp")
        h_slg = wavg("slg")
        total_hr = sum((r.get("home_runs") or 0) for r in hit_rows)
        total_sb = sum((r.get("stolen_bases") or 0) for r in hit_rows)
        h_hr_per_pa = (total_hr / total_pa) if total_pa > 0 else None
        h_sb_per_pa = (total_sb / total_pa) if total_pa > 0 else None
        h_level = max((LEVEL_RANK.get((r.get("level") or "").upper(), 0)
                       for r in hit_rows), default=0)
        hit_vec = [
            h_pa,
            h_woba if h_woba is not None else MISSING,
            h_iso if h_iso is not None else MISSING,
            h_k if h_k is not None else MISSING,
            h_bb if h_bb is not None else MISSING,
            h_avg if h_avg is not None else MISSING,
            h_obp if h_obp is not None else MISSING,
            h_slg if h_slg is not None else MISSING,
            h_hr_per_pa if h_hr_per_pa is not None else MISSING,
            h_sb_per_pa if h_sb_per_pa is not None else MISSING,
            float(h_level) if h_level > 0 else MISSING,
        ]
    else:
        hit_vec = [MISSING] * len(HIT_FEATS)

    # ---- Pitcher aggregate (weighted by IP) ----
    pit_rows = [r for r in matches if (r.get("ip") or 0) > 0]
    if pit_rows:
        total_ip = sum((r.get("ip") or 0) for r in pit_rows)
        def wavgp(key, transform=None):
            tf = transform if transform is not None else (lambda r, _k=key: r.get(_k))
            vals = [(tf(r), r.get("ip") or 0)
                    for r in pit_rows if tf(r) is not None]
            denom = sum(w for _, w in vals)
            return (sum(v * w for v, w in vals) / denom) if denom > 0 else None
        p_ip = float(total_ip)
        p_era = wavgp("era")
        p_k9 = wavgp("k9")
        p_bb9 = wavgp("bb9")
        p_fip = wavgp("fip", _fip_proxy)
        p_whip = wavgp("whip")
        p_hr9 = wavgp("hr9")
        p_level = max((LEVEL_RANK.get((r.get("level") or "").upper(), 0)
                       for r in pit_rows), default=0)
        pit_vec = [
            p_ip,
            p_era if p_era is not None else MISSING,
            p_k9 if p_k9 is not None else MISSING,
            p_bb9 if p_bb9 is not None else MISSING,
            p_fip if p_fip is not None else MISSING,
            p_whip if p_whip is not None else MISSING,
            p_hr9 if p_hr9 is not None else MISSING,
            float(p_level) if p_level > 0 else MISSING,
        ]
    else:
        pit_vec = [MISSING] * len(PIT_FEATS)

    return hit_vec + pit_vec


def build_windowed_features(
    prospect: dict,
    season_stats: list[dict],
    as_of_year: int,
    *,
    milb_only: bool = False,
) -> np.ndarray:
    """Build a single fixed-length feature vector for (prospect, as_of_year).

    If `milb_only=True`, exclude any MLB-level rows from the feature window
    (pure prospect-state classifier — features can only see minor-league play).
    """
    if milb_only:
        season_stats = [s for s in season_stats
                        if (s.get("level") or "").upper() != "MLB"]
    pos = (prospect.get("primary_position") or "").upper()
    is_pitcher = pos in PITCHER_POS or bool(prospect.get("is_pitcher"))

    # Pedigree
    bonus = prospect.get("signing_bonus_usd")
    log_bonus = float(np.log1p(bonus)) if bonus and bonus > 0 else MISSING
    has_bonus = 1.0 if bonus and bonus > 0 else 0.0
    draft_round = prospect.get("draft_round")
    draft_pick = prospect.get("draft_pick")
    origin = (prospect.get("origin") or "").lower()
    is_college = float(any(t in origin for t in (
        "univ", "college", "state", "u of", "tech", "institute"
    )))
    is_intl = float(prospect.get("is_international") or 0)
    is_premium = float(pos in PREMIUM_POSITIONS)

    pedigree_vec = [
        float(is_pitcher),
        float(draft_round) if draft_round is not None else MISSING,
        float(draft_pick) if draft_pick is not None else MISSING,
        log_bonus,
        has_bonus,
        is_intl,
        is_premium,
        is_college,
    ]

    # Per-year stats for as_of_year, as_of_year-1, as_of_year-2
    per_year_block = []
    per_year_segments = []  # keep per-lag lists for delta computation
    for lag in range(WINDOW):
        year = as_of_year - lag
        seg = _stats_for_year(season_stats, year, is_pitcher)
        per_year_segments.append(seg)
        per_year_block.extend(seg)

    # YoY deltas between yT (lag=0) and y1 (lag=1).
    # Mapping from DELTA_FEATS name -> (block, offset within HIT_FEATS or PIT_FEATS)
    hit_idx = {n: i for i, n in enumerate(HIT_FEATS)}
    pit_idx = {n: i for i, n in enumerate(PIT_FEATS)}
    n_hit = len(HIT_FEATS)
    def _delta(side: str, key: str) -> float:
        if side == "h":
            i = hit_idx[key]
            yT = per_year_segments[0][i]
            y1 = per_year_segments[1][i]
        else:
            i = pit_idx[key]
            yT = per_year_segments[0][n_hit + i]
            y1 = per_year_segments[1][n_hit + i]
        if yT == MISSING or y1 == MISSING:
            return MISSING
        return float(yT - y1)
    delta_block = [
        _delta("h", "woba_proxy"),
        _delta("h", "iso"),
        _delta("h", "k_pct"),
        _delta("h", "bb_pct"),
        _delta("h", "obp"),
        _delta("h", "slg"),
        _delta("h", "level_rank_h"),
        _delta("p", "era"),
        _delta("p", "k9"),
        _delta("p", "bb9"),
        _delta("p", "fip"),
        _delta("p", "level_rank_p"),
    ]

    # Window summary
    window_years = {as_of_year - i for i in range(WINDOW)}
    years_observed = len({s.get("season_year") for s in season_stats
                          if s.get("season_year") in window_years})
    max_level = max(
        (LEVEL_RANK.get((s.get("level") or "").upper(), 0)
         for s in season_stats if s.get("season_year") in window_years),
        default=0,
    )
    draft_year = prospect.get("draft_year")
    years_in_pro = (as_of_year - draft_year) if draft_year else None

    # Age at as_of_year: prefer season_stats.age_during_season for that year;
    # else infer from draft_year heuristically (HS=18, college=21 by default).
    age_at_as_of = None
    same_year_rows = [s for s in season_stats if s.get("season_year") == as_of_year]
    ages = [s.get("age_during_season") for s in same_year_rows
            if s.get("age_during_season") is not None]
    if ages:
        age_at_as_of = float(sum(ages) / len(ages))
    elif draft_year is not None:
        base_age = 21.0 if is_college else 19.0
        age_at_as_of = base_age + max(0, as_of_year - draft_year)

    # Years in current system: longest run of consecutive years (ending at
    # as_of_year) where the player was in the same org. Uses the as_of_year's
    # org as the anchor; if multiple orgs appeared in that year (mid-season
    # trade), use the one with the most PA/IP.
    years_in_current_system = None
    if same_year_rows:
        org_counts: dict[str, float] = {}
        for s in same_year_rows:
            org = s.get("org") or ""
            if not org:
                continue
            org_counts[org] = org_counts.get(org, 0) + (s.get("pa") or 0) + (s.get("ip") or 0)
        if org_counts:
            anchor_org = max(org_counts, key=org_counts.get)
            run = 0
            for back in range(0, 25):  # cap at 25 years of lookback
                y = as_of_year - back
                rows_y = [s for s in season_stats if s.get("season_year") == y]
                if not rows_y:
                    break
                orgs_y = {s.get("org") for s in rows_y if s.get("org")}
                if anchor_org in orgs_y:
                    run += 1
                else:
                    break
            years_in_current_system = float(run)

    tail = [
        float(years_observed),
        float(max_level) if max_level > 0 else MISSING,
        float(years_in_pro) if years_in_pro is not None else MISSING,
        float(age_at_as_of) if age_at_as_of is not None else MISSING,
        float(years_in_current_system) if years_in_current_system is not None else MISSING,
    ]

    vec = np.array(pedigree_vec + per_year_block + delta_block + tail,
                   dtype=np.float64)
    assert vec.shape == (N_FEATURES,), (vec.shape, N_FEATURES)
    return vec


def sample_as_of_year(
    rng: np.random.Generator,
    prospect: dict,
    stats: list[dict],
    *,
    milb_only: bool = False,
) -> Optional[int]:
    """Pick a random as-of year from the player's active years, bounded
    to [draft_year, draft_year+20] to drop name-collision rows from old eras.

    If `milb_only=True`, restrict to years that had at least one non-MLB row.
    Returns None if no eligible year exists (caller can drop the player).
    """
    dy = prospect.get("draft_year")
    lo = (dy or 1990) - 1
    hi = (dy or 2024) + 20
    eligible = [s for s in stats
                if s.get("season_year") is not None
                and lo <= s["season_year"] <= hi
                and (not milb_only or (s.get("level") or "").upper() != "MLB")]
    years = sorted({s["season_year"] for s in eligible})
    if years:
        return int(rng.choice(years))
    if milb_only:
        return None  # caller drops this player
    if dy is not None:
        return int(dy + rng.integers(0, 7))
    return 2017


def build_training_dataset(
    db: ProspectDB,
    seed: int = 42,
    require_outcome: bool = True,
    milb_only: bool = False,
) -> tuple[np.ndarray, list[str], list[int], list[dict]]:
    """
    Sample one (prospect, as_of_year) row per player; return X, player_ids,
    as_of_years, and the joined prospect+outcome dicts (with events_json).

    If `milb_only=True`, restrict both the as-of-year sample space and the
    feature window to non-MLB rows, and drop players who have no MiLB rows.
    """
    rng = np.random.default_rng(seed)
    with db._connect() as conn:
        if require_outcome:
            rows = conn.execute(
                """
                SELECT p.*, o.events_json, o.mlb_debut_year, o.career_pa, o.career_ip,
                       o.year_top_100, o.year_top_25, o.year_established_mlb,
                       o.year_all_star_once, o.year_all_star_three,
                       o.year_major_award, o.year_hof_trajectory
                FROM prospects p
                JOIN career_outcomes o ON p.player_id = o.player_id
                """
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM prospects").fetchall()
        prospect_rows = [dict(r) for r in rows]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()

    stats_by_player: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_player.setdefault(d["player_id"], []).append(d)

    X_rows = []
    pids = []
    years = []
    joined = []
    dropped = 0
    for pr in prospect_rows:
        stats = stats_by_player.get(pr["player_id"], [])
        year = sample_as_of_year(rng, pr, stats, milb_only=milb_only)
        if year is None:
            dropped += 1
            continue
        vec = build_windowed_features(pr, stats, year, milb_only=milb_only)
        X_rows.append(vec)
        pids.append(pr["player_id"])
        years.append(year)
        joined.append(pr)
    if milb_only and dropped:
        # informational; caller can also see by comparing len(joined) to all_prospects
        pass

    X = np.vstack(X_rows) if X_rows else np.zeros((0, N_FEATURES))
    return X, pids, years, joined


def build_panel_dataset(
    db: ProspectDB,
    require_outcome: bool = True,
    max_year: int | None = None,
) -> tuple[np.ndarray, list[str], list[int], list[dict]]:
    """
    Panel version of build_training_dataset.

    For each prospect, emit ONE row per pre-MLB year the player had MiLB stats.
    A player who played MiLB 2017-2020 then debuted MLB in 2021 contributes 4
    rows (as_of in {2017, 2018, 2019, 2020}). Same player can therefore appear
    multiple times — caller MUST split train/val/test by player_id, not by row,
    to prevent leakage.

    Features always built MiLB-only. as_of_year capped at `max_year` if given
    (e.g. last completed season).

    Returns:
        X        (n_rows, n_features)
        pids     parallel list of player_ids (duplicates allowed)
        years    parallel list of as_of_years
        joined   parallel list of joined prospect+outcome dicts
    """
    with db._connect() as conn:
        if require_outcome:
            rows = conn.execute(
                """
                SELECT p.*, o.events_json, o.mlb_debut_year, o.career_pa, o.career_ip,
                       o.year_top_100, o.year_top_25, o.year_established_mlb,
                       o.year_all_star_once, o.year_all_star_three,
                       o.year_major_award, o.year_hof_trajectory
                FROM prospects p
                JOIN career_outcomes o ON p.player_id = o.player_id
                """
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM prospects").fetchall()
        prospect_rows = [dict(r) for r in rows]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()

    stats_by_player: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_player.setdefault(d["player_id"], []).append(d)

    X_rows = []
    pids = []
    years = []
    joined = []
    for pr in prospect_rows:
        pid = pr["player_id"]
        stats = stats_by_player.get(pid, [])
        debut = pr.get("mlb_debut_year")
        dy = pr.get("draft_year")
        lo = (dy or 1990) - 1
        hi = (dy or 2024) + 20
        if max_year is not None:
            hi = min(hi, max_year)
        # eligible years: every distinct season the player had a non-MLB row in,
        # bounded to be strictly before MLB debut (if they debuted).
        eligible_years = sorted({
            s["season_year"] for s in stats
            if s.get("season_year") is not None
            and lo <= s["season_year"] <= hi
            and (s.get("level") or "").upper() != "MLB"
            and (debut is None or s["season_year"] < int(debut))
        })
        for yr in eligible_years:
            vec = build_windowed_features(pr, stats, yr, milb_only=True)
            X_rows.append(vec)
            pids.append(pid)
            years.append(yr)
            joined.append(pr)

    X = np.vstack(X_rows) if X_rows else np.zeros((0, N_FEATURES))
    return X, pids, years, joined


def y_for_event(joined: list[dict], event_int: int) -> np.ndarray:
    y = np.zeros(len(joined), dtype=np.int8)
    key = str(event_int)
    for i, r in enumerate(joined):
        ej = r.get("events_json")
        if not ej:
            continue
        d = json.loads(ej) if isinstance(ej, str) else ej
        if d.get(key):
            y[i] = 1
    return y
