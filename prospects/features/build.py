"""
prospects/features/build.py
============================

Turn a Prospect + its SeasonStats history into a numeric feature vector.

Training-time semantics: features are computed using only data that would have
been observable *before* the player's MLB debut (or all data, if they never
debuted). This prevents target leakage — we can't use MLB performance to
predict whether the player made the MLB.

Feature blocks:
  - Pedigree (draft round/pick/bonus/origin)
  - Best minor-league rate stats (hitting + pitching)
  - Highest level reached + age-vs-level
  - Trajectory (years to top level, total minor-league PA/IP)

Output is a fixed-length feature vector + parallel feature names list.
Missing values are imputed to 0.0 (with a sentinel _missing flag per feature
group); the model is robust to this because tree-based classifiers handle it
natively and logistic regression sees the missingness flag.
"""

from __future__ import annotations

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
PITCHER_POSITIONS = {"P", "RHP", "LHP", "SP", "RP"}


FEATURE_NAMES = [
    # Pedigree (8)
    "is_pitcher",
    "draft_round",
    "draft_pick",
    "log_signing_bonus",
    "has_signing_bonus",
    "is_international",
    "is_premium_position",
    "is_college_draftee",
    # Performance — hitters (6)
    "best_iso",
    "best_woba",
    "best_bb_pct",
    "best_k_pct",
    "best_hitter_pa_in_season",
    "total_minor_pa",
    # Performance — pitchers (6)
    "best_k9",
    "best_bb9",
    "best_fip",
    "best_era",
    "best_pitcher_ip_in_season",
    "total_minor_ip",
    # Trajectory (5)
    "highest_minor_level",
    "age_at_highest_level",
    "age_vs_level_avg",
    "years_to_high_level",
    "n_minor_seasons",
    # Missingness flags (3)
    "missing_hitter_stats",
    "missing_pitcher_stats",
    "missing_pedigree",
]

N_FEATURES = len(FEATURE_NAMES)

# Rough "average age" by level for the age-vs-level feature.
LEVEL_AVG_AGE = {
    "DSL": 18.0, "FCL": 19.0, "CPX": 19.5, "RK": 19.5, "ROK": 19.5,
    "A-": 20.5, "A": 21.0, "A+": 22.0,
    "AA": 23.5, "AAA": 25.0, "MLB": 27.5,
    "NCAA-D1": 20.5, "NCAA-D2": 20.0, "NCAA-D3": 20.0,
}


def _safe_log_bonus(bonus: Optional[float]) -> tuple[float, float]:
    """Returns (log_bonus, has_bonus_flag). log of 0 -> 0."""
    if bonus is None or bonus <= 0:
        return 0.0, 0.0
    return float(np.log1p(bonus)), 1.0


def _filter_pre_mlb(stats: list[dict], outcome: Optional[dict]) -> list[dict]:
    """
    Drop MLB seasons + any minor-league seasons after MLB debut. This is the
    training-time projection: we only see what was observable before the
    classifier had to make a call.
    """
    out = []
    debut = (outcome or {}).get("mlb_debut_year")
    for s in stats:
        if s.get("level") == "MLB":
            continue
        if debut is not None and s.get("season_year", 9999) >= debut:
            continue
        out.append(s)
    return out


def _best(values: list[Optional[float]], *, higher_is_better: bool) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return max(clean) if higher_is_better else min(clean)


def build_feature_vector(
    prospect: dict,
    season_stats: list[dict],
    outcome: Optional[dict] = None,
) -> np.ndarray:
    """
    Build the feature vector for a single prospect.

    prospect: row from `prospects` table (as dict).
    season_stats: list of `season_stats` rows for the player.
    outcome: optional row from `career_outcomes` (used only for the pre-MLB cut).
             At inference time outcome is None and all stats are used.
    """
    is_pitcher = bool(prospect.get("is_pitcher"))
    pos = (prospect.get("primary_position") or "").upper()

    # ---- Pedigree ----
    draft_round = prospect.get("draft_round")
    draft_pick = prospect.get("draft_pick")
    log_bonus, has_bonus = _safe_log_bonus(prospect.get("signing_bonus_usd"))
    is_international = float(prospect.get("is_international") or 0)
    is_premium = float(pos in PREMIUM_POSITIONS)
    origin = (prospect.get("origin") or "").lower()
    is_college = float(any(tok in origin for tok in ("univ", "college", "state", "u of", "tech", "institute")))

    missing_pedigree = float(draft_round is None and draft_pick is None and not has_bonus)

    # ---- Minor-league stats (pre-MLB) ----
    minor = _filter_pre_mlb(season_stats, outcome)
    hitter_rows = [s for s in minor if (s.get("pa") or 0) >= 50]
    pitcher_rows = [s for s in minor if (s.get("ip") or 0) >= 20]

    best_iso = _best([s.get("iso") for s in hitter_rows], higher_is_better=True)
    best_woba = _best([s.get("woba") for s in hitter_rows], higher_is_better=True)
    best_bb = _best([s.get("bb_pct") for s in hitter_rows], higher_is_better=True)
    best_k_hit = _best([s.get("k_pct") for s in hitter_rows], higher_is_better=False)
    best_hit_pa = max([s.get("pa") or 0 for s in hitter_rows], default=0)
    total_pa = sum((s.get("pa") or 0) for s in minor)

    best_k9 = _best([s.get("k9") for s in pitcher_rows], higher_is_better=True)
    best_bb9 = _best([s.get("bb9") for s in pitcher_rows], higher_is_better=False)
    best_fip = _best([s.get("fip") for s in pitcher_rows], higher_is_better=False)
    best_era = _best([s.get("era") for s in pitcher_rows], higher_is_better=False)
    best_p_ip = max([s.get("ip") or 0 for s in pitcher_rows], default=0)
    total_ip = sum((s.get("ip") or 0) for s in minor)

    missing_hit = float(not hitter_rows)
    missing_pit = float(not pitcher_rows)

    # ---- Trajectory ----
    levels_reached = [LEVEL_RANK.get((s.get("level") or "").upper(), 0) for s in minor]
    highest_level = max(levels_reached, default=0)

    age_at_high = None
    age_vs_level = None
    years_to_high = None
    if highest_level > 0:
        top_seasons = [s for s in minor
                       if LEVEL_RANK.get((s.get("level") or "").upper(), 0) == highest_level]
        if top_seasons:
            ages = [s.get("age_during_season") for s in top_seasons if s.get("age_during_season") is not None]
            if ages:
                age_at_high = float(min(ages))  # youngest age at this level
                level_name = (top_seasons[0].get("level") or "").upper()
                avg_age = LEVEL_AVG_AGE.get(level_name)
                if avg_age is not None:
                    age_vs_level = age_at_high - avg_age
            first_year = min(s.get("season_year") for s in minor if s.get("season_year"))
            high_year = min(s.get("season_year") for s in top_seasons if s.get("season_year"))
            if first_year is not None and high_year is not None:
                years_to_high = float(high_year - first_year)

    n_seasons = len({(s.get("season_year"), s.get("level")) for s in minor})

    # ---- Pack into fixed-length vector ----
    def f(v, default=0.0):
        return default if v is None else float(v)

    vec = np.array([
        float(is_pitcher),
        f(draft_round),
        f(draft_pick),
        log_bonus,
        has_bonus,
        is_international,
        is_premium,
        is_college,

        f(best_iso),
        f(best_woba),
        f(best_bb),
        f(best_k_hit),
        float(best_hit_pa),
        float(total_pa),

        f(best_k9),
        f(best_bb9),
        f(best_fip),
        f(best_era),
        float(best_p_ip),
        float(total_ip),

        float(highest_level),
        f(age_at_high),
        f(age_vs_level),
        f(years_to_high),
        float(n_seasons),

        missing_hit,
        missing_pit,
        missing_pedigree,
    ], dtype=np.float64)

    assert vec.shape == (N_FEATURES,), (vec.shape, N_FEATURES)
    return vec


def build_training_matrix(
    db: ProspectDB,
    require_outcome: bool = True,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Build feature matrix X for all prospects that have a career_outcome row.

    Returns:
        X: (n_players, n_features) float64 ndarray
        player_ids: list of n_players strings, parallel to rows of X
        outcomes: list of outcome dicts (so caller can build y labels per event)
    """
    with db._connect() as conn:
        if require_outcome:
            rows = conn.execute(
                """
                SELECT p.*, o.events_json, o.mlb_debut_year, o.career_pa, o.career_ip
                FROM prospects p
                JOIN career_outcomes o ON p.player_id = o.player_id
                """
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM prospects").fetchall()
        prospect_rows = [dict(r) for r in rows]

        # Pre-fetch all season stats in one query, group by player_id
        stats_rows = conn.execute(
            "SELECT * FROM season_stats"
        ).fetchall()

    stats_by_player: dict[str, list[dict]] = {}
    for sr in stats_rows:
        d = dict(sr)
        stats_by_player.setdefault(d["player_id"], []).append(d)

    X_rows = []
    player_ids = []
    outcomes = []
    for pr in prospect_rows:
        pid = pr["player_id"]
        stats = stats_by_player.get(pid, [])
        outcome = {
            "mlb_debut_year": pr.get("mlb_debut_year"),
            "events_json": pr.get("events_json"),
        } if require_outcome else None
        vec = build_feature_vector(pr, stats, outcome)
        X_rows.append(vec)
        player_ids.append(pid)
        outcomes.append(pr if require_outcome else {})

    X = np.vstack(X_rows) if X_rows else np.zeros((0, N_FEATURES))
    return X, player_ids, outcomes
