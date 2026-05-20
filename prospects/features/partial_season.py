"""
prospects/features/partial_season.py
======================================

Empirical-Bayes shrinkage for in-progress seasons.

Problem: the hazard model was trained on COMPLETE seasons. At inference time
mid-season, raw stats like "PA=80, OBP=.245, K%=28%" are read by the model as
if those were full-season values - heavily penalizing the player.

Fix: combine the current partial-season row with the player's most-recent
complete-season row via shrinkage.

Rate stats:
    posterior = (n_obs * x_obs + K_eff * x_prior) / (n_obs + K_eff)
  K_eff = min(K_max, prior_sample_n). Tiny-sample priors automatically
  get downweighted, so even an imperfect prior selection cannot dominate.

Counting stats (PA, IP, HR, SB):
    If observed sample is small (PA<50 / IP<15), shrink the naive forward
    projection toward the prior. Otherwise return the naive projection
    clipped to a sane full-season ceiling. The result is NEVER below the
    observed value, so a small or fluky prior cannot pull a real sample
    down.

Returned: a synthesized "virtual full-season" stats dict that downstream
feature builders can consume as if it were a normal complete year.

Usage:
    from prospects.features.partial_season import blend_partial_season

    virtual = blend_partial_season(
        partial=stats_2026_row,
        prior=stats_2025_row,
        season_progress=0.20,    # ~20% of season elapsed
    )
"""
from __future__ import annotations

from typing import Optional


# Default cap on prior weight in rate-stat shrinkage. Effective weight is
# min(K_MAX, prior_sample_n) so a tiny prior never dominates a real sample.
DEFAULT_K_HITTER_PA = 150.0
DEFAULT_K_PITCHER_IP = 50.0

# Below these thresholds, treat the observed counting stat as "small sample"
# and shrink the naive projection toward the prior.
SMALL_SAMPLE_PA = 100.0
SMALL_SAMPLE_IP = 30.0

# Sanity ceiling on projected counting stats.
MAX_PROJECTED_PA = 700.0
MAX_PROJECTED_IP = 200.0

# Cap forward projection at this multiple of the observed value. Prevents
# a hot-start small sample (e.g. 4 HR in 156 PA) from being extrapolated
# into an elite-pace full season (17 HR) on a feature pattern the model
# treats as elite. 2x observed lets genuine breakouts push modestly without
# producing runaway predictions.
MAX_OBSERVED_MULTIPLIER = 2.0

# Sample-size thresholds that qualify a prior row as a "real" anchor.
PRIOR_MIN_PA = 100.0
PRIOR_MIN_IP = 25.0

HITTER_RATE_KEYS = (
    "avg", "obp", "slg", "iso", "k_pct", "bb_pct", "babip", "woba",
    "hr_per_pa", "sb_per_pa",
)
PITCHER_RATE_KEYS = ("era", "k9", "bb9", "fip", "whip", "hr9")

HITTER_COUNT_KEYS = ("pa", "home_runs", "stolen_bases")
PITCHER_COUNT_KEYS = ("ip",)


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _shrink_rate(x_obs, x_prior, n_obs: float, K_max: float,
                 prior_n: Optional[float]) -> Optional[float]:
    """EB shrinkage where prior weight is capped at the prior's own sample size.

    K_eff = min(K_max, prior_n). A 27-PA prior contributes at most 27 PA of
    weight, not the default 150, so an imperfectly-selected small-sample prior
    cannot overwhelm a meaningful observation.
    """
    xo, xp = _num(x_obs), _num(x_prior)
    if xo is None and xp is None:
        return None
    if xo is None:
        return xp
    if xp is None:
        return xo
    pn = _num(prior_n)
    K_eff = K_max if pn is None else min(K_max, max(pn, 0.0))
    denom = n_obs + K_eff
    if denom <= 0:
        return xp
    return (n_obs * xo + K_eff * xp) / denom


def _project_count(n_obs, n_prior, season_progress: float,
                   small_sample: float, ceiling: float) -> Optional[float]:
    """Project a counting stat forward.

    Rules:
      - Result is NEVER below the observed value. A small / fluky prior can
        no longer pull a real sample down.
      - If observed >= small_sample threshold: trust the naive forward
        projection, clipped to a sane ceiling.
      - If observed < small_sample: shrink the naive projection toward the
        prior (when present) to prevent runaway extrapolation on tiny samples.
    """
    no = _num(n_obs)
    np_prior = _num(n_prior)
    if no is None and np_prior is None:
        return None
    if no is None:
        return np_prior
    if season_progress is None or season_progress <= 0 or season_progress > 1:
        return max(no, np_prior or 0.0)

    naive = no / season_progress

    # Always cap forward projection at MAX_OBSERVED_MULTIPLIER * observed.
    # This is the primary defense against a small hot-start sample being
    # extrapolated into an elite-pace full season.
    observed_cap = MAX_OBSERVED_MULTIPLIER * no

    if no >= small_sample or np_prior is None or np_prior <= 0:
        return max(no, min(naive, ceiling, observed_cap))

    # Small sample: linear blend of naive projection and prior total,
    # weighted by how much of the small-sample threshold we have. Then
    # clamp so we never return less than the observed value and never
    # exceed the observed-multiplier cap.
    w = min(1.0, no / small_sample)
    blended = w * naive + (1.0 - w) * np_prior
    return max(no, min(blended, ceiling, observed_cap))


def blend_partial_season(
    partial: dict,
    prior: Optional[dict],
    season_progress: float = 0.25,
    k_hitter_pa: float = DEFAULT_K_HITTER_PA,
    k_pitcher_ip: float = DEFAULT_K_PITCHER_IP,
) -> dict:
    """Return a synthesized "virtual full-season" stats row.

    Args:
        partial: season_stats row for the current (incomplete) year.
        prior:   most-recent usable prior-season row, or None.
        season_progress: estimated fraction of season elapsed in `partial`.
    """
    out = dict(partial)

    n_obs_pa = _num(partial.get("pa")) or 0.0
    n_obs_ip = _num(partial.get("ip")) or 0.0
    prior_pa = _num((prior or {}).get("pa")) if prior else None
    prior_ip = _num((prior or {}).get("ip")) if prior else None

    if n_obs_pa > 0 or (prior_pa and prior_pa > 0):
        for key in HITTER_RATE_KEYS:
            prior_val = (prior or {}).get(key) if prior else None
            out[key] = _shrink_rate(partial.get(key), prior_val,
                                    n_obs_pa, k_hitter_pa, prior_pa)

    if n_obs_ip > 0 or (prior_ip and prior_ip > 0):
        for key in PITCHER_RATE_KEYS:
            prior_val = (prior or {}).get(key) if prior else None
            out[key] = _shrink_rate(partial.get(key), prior_val,
                                    n_obs_ip, k_pitcher_ip, prior_ip)

    out["pa"] = _project_count(partial.get("pa"), prior_pa, season_progress,
                               SMALL_SAMPLE_PA, MAX_PROJECTED_PA)
    out["ip"] = _project_count(partial.get("ip"), prior_ip, season_progress,
                               SMALL_SAMPLE_IP, MAX_PROJECTED_IP)
    if prior:
        out["home_runs"] = _project_count(
            partial.get("home_runs"), prior.get("home_runs"),
            season_progress, SMALL_SAMPLE_PA, MAX_PROJECTED_PA,
        )
        out["stolen_bases"] = _project_count(
            partial.get("stolen_bases"), prior.get("stolen_bases"),
            season_progress, SMALL_SAMPLE_PA, MAX_PROJECTED_PA,
        )

    out["_blended_partial"] = True
    out["_season_progress"] = season_progress
    out["_n_obs_pa"] = n_obs_pa
    out["_n_obs_ip"] = n_obs_ip
    if prior:
        out["_prior_year"] = prior.get("season_year")
        out["_prior_level"] = prior.get("level")
        out["_prior_pa"] = prior_pa
        out["_prior_ip"] = prior_ip
    else:
        out["_prior_year"] = None
        out["_prior_level"] = None
        out["_prior_pa"] = None
        out["_prior_ip"] = None
    return out


def _is_real_sample(row: dict) -> bool:
    pa = _num(row.get("pa")) or 0.0
    ip = _num(row.get("ip")) or 0.0
    return pa >= PRIOR_MIN_PA or ip >= PRIOR_MIN_IP


def select_prior(season_stats: list[dict], partial_year: int) -> Optional[dict]:
    """Pick the best prior-season row for blending.

    Preference order:
      1. Same level as partial year, year-1, AND meaningful sample
         (PA >= PRIOR_MIN_PA or IP >= PRIOR_MIN_IP). Pick largest sample.
      2. Most recent year with meaningful sample (any level except MLB).
      3. Most recent year, any sample.
    """
    partial_rows = [s for s in season_stats if s.get("season_year") == partial_year]
    partial_levels = {(s.get("level") or "").upper() for s in partial_rows}
    candidates = [s for s in season_stats
                  if s.get("season_year") is not None
                  and s["season_year"] < partial_year
                  and (s.get("level") or "").upper() != "MLB"]
    if not candidates:
        return None

    same_level_y1 = [s for s in candidates
                     if s["season_year"] == partial_year - 1
                     and (s.get("level") or "").upper() in partial_levels
                     and _is_real_sample(s)]
    if same_level_y1:
        return max(same_level_y1,
                   key=lambda s: (_num(s.get("pa")) or 0)
                                 + (_num(s.get("ip")) or 0))

    real = [s for s in candidates if _is_real_sample(s)]
    if real:
        return max(real, key=lambda s: (s["season_year"],
                                        (_num(s.get("pa")) or 0)
                                        + (_num(s.get("ip")) or 0)))

    return max(candidates, key=lambda s: s["season_year"])


def apply_blender_to_stats(
    stats: list[dict],
    current_year: int,
    season_progress: float = 0.25,
) -> list[dict]:
    """Replace any partial-current-year rows with blended versions."""
    if not stats:
        return list(stats)
    partial = [s for s in stats if s.get("season_year") == current_year
               and (s.get("level") or "").upper() != "MLB"]
    if not partial:
        return list(stats)
    prior = select_prior(stats, current_year)
    blended_rows = [
        blend_partial_season(p, prior, season_progress=season_progress)
        for p in partial
    ]
    keep = [s for s in stats if s.get("season_year") != current_year]
    return keep + blended_rows
