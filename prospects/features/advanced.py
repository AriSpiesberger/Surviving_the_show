"""
prospects/features/advanced.py
==============================

Advanced (sabermetric) statistics computed from the raw counting stats stored
on `season_stats`.

Design: the database stores *counts*, this module derives *rates*. Nothing
here re-pulls or caches — call it on a season_stats row dict and it hands back
the analyst-facing metrics. Keeping the derivation separate from ingest means
weights can change (era-specific wOBA coefficients, a new xFIP constant)
without a re-pull.

Three tiers of metric:

  1. Self-contained — computable from one row alone (wOBA, K%, SwStr%, GB%,
     FIP). Available for MLB *and* every MiLB level back to 2005.
  2. League-relative — need a (season, level) baseline (wRC+, ERA-, FIP-,
     xFIP). Baselines come from `league_context()`.
  3. Park-adjusted — need a park factor for the player's team, from the
     `park_factors` table.

What is deliberately *not* here: anything requiring pitch tracking (chase
rate, exit velocity, spin). Those exist for MLB only and live in the
statcast_* tables — see `prospects/data/sources/statcast.py`.
"""

from __future__ import annotations

from typing import Optional


# ============================================================================
# wOBA linear weights
#
# Run values shift with the league run environment. These are the canonical
# FanGraphs-style weights for the modern era; `WOBA_WEIGHTS_BY_ERA` lets a
# season use coefficients closer to its own run environment. wOBA is scaled to
# league OBP, so the scale term matters as much as the weights.
# ============================================================================

_WOBA_MODERN = {
    "bb": 0.690, "hbp": 0.720, "1b": 0.880,
    "2b": 1.247, "3b": 1.578, "hr": 2.031,
    "scale": 1.157, "lg_woba": 0.310,
}
_WOBA_HIGH_OFFENSE = {   # 2000-2009, steroid-era run environment
    "bb": 0.700, "hbp": 0.732, "1b": 0.887,
    "2b": 1.254, "3b": 1.590, "hr": 2.043,
    "scale": 1.180, "lg_woba": 0.338,
}
_WOBA_DEADBALL_2010S = {  # 2010-2015, depressed offense
    "bb": 0.690, "hbp": 0.722, "1b": 0.888,
    "2b": 1.271, "3b": 1.616, "hr": 2.101,
    "scale": 1.185, "lg_woba": 0.317,
}


def woba_weights(season_year: Optional[int]) -> dict:
    """Linear weights appropriate to a season's run environment."""
    if season_year is None:
        return _WOBA_MODERN
    if season_year < 2010:
        return _WOBA_HIGH_OFFENSE
    if season_year < 2016:
        return _WOBA_DEADBALL_2010S
    return _WOBA_MODERN


# ============================================================================
# Small helpers
# ============================================================================

def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _div(num: Optional[float], den: Optional[float],
         min_den: float = 0.0) -> Optional[float]:
    """Safe ratio. Returns None on missing input or a denominator at/below
    `min_den` — a rate off 3 batted balls is noise, not a measurement."""
    if num is None or den is None or den <= min_den:
        return None
    return num / den


def _sum(row: dict, *keys: str) -> Optional[float]:
    """Sum of the named columns. None if *every* component is missing;
    missing components among present ones count as 0."""
    vals = [_f(row, k) for k in keys]
    present = [v for v in vals if v is not None]
    if not present:
        return None
    return sum(present)


# ============================================================================
# PITCH-DATA RELIABILITY GATE
#
# Pitch-level logging is not uniform across leagues that share a sportId. The
# Dominican Summer League — and the Venezuelan Summer League, which ran
# through 2015 — record roughly HALF the pitches actually thrown. Measured on
# 2024 (players with 50+ PA / 20+ IP), against the complex leagues that share
# sportId 16:
#
#     league    P/PA    P/IP    Swing%   Whiff%   SwStr%
#     DSL       1.97     8.54    0.600    0.480    0.288
#     ACL       3.80    17.45    0.443    0.314    0.139
#     FCL       3.80    16.87    0.432    0.282    0.122
#
# Note that `totalSwings` and `swingAndMisses` are corrupted independently of
# `numberOfPitches` — Whiff% is swings-and-misses over swings and never
# touches the pitch count, yet it still reads ~1.6x the complex-league rate.
# So the whole pitch-tracking block has to go, not only the ratios with
# pitches in the denominator.
#
# What survives is everything built on plate appearances and balls in play.
# Same 2024 sample:
#
#     league      K%      BB%     GB%      LD%     BABIP
#     DSL       0.219    0.139   0.463    0.179    0.308
#     ACL       0.249    0.130   0.458    0.181    0.348
#     FCL       0.235    0.131   0.464    0.168    0.314
#
# Those are league differences, not artifacts. So a DSL line still yields
# wOBA, K%, BB% and the full batted-ball profile — only the swing rates are
# unusable.
# ============================================================================

PITCH_DATA_UNRELIABLE_LEAGUES = frozenset({
    "DSL",   # Dominican Summer League — ~50% of pitches logged
    "VSL",   # Venezuelan Summer League — same, ran through 2015
})

# Metrics that must be suppressed when the pitch log is unreliable.
_PITCH_DERIVED_HIT = (
    "swing_pct", "whiff_pct", "swstr_pct", "contact_pct",
    "pitches_per_pa", "k_minus_swstr",
)
_PITCH_DERIVED_PIT = (
    "strike_pct", "swing_pct", "whiff_pct", "swstr_pct", "contact_pct",
    "pitches_per_bf", "pitches_per_ip",
)


def pitch_data_reliable(row: dict) -> bool:
    """Whether a row's pitch-level counts can be trusted.

    Primary source is `season_stats.pitch_data_valid`, a per-(level, season,
    team) flag written by data/backfills/pitch_validity.py from a numeric
    range check. Prefer it over any league or era rule: the logging rollout
    was team-by-team at the boundaries, so AAA 2006, AA 2008-2011 and RK 2024
    are all genuinely split cells that no league-level rule can express.

    Falls back to the league name only when the flag has not been computed.
    Anything undetermined is treated as UNRELIABLE — the corrupt half of the
    panel is large enough that silence must not read as consent.
    """
    flag = row.get("pitch_data_valid")
    if flag is not None:
        return bool(flag)

    lg = row.get("league")
    if lg is not None:
        return str(lg).strip().upper() not in PITCH_DATA_UNRELIABLE_LEAGUES

    # No flag and no league. Range-check the row itself as a last resort.
    pa, pitches = _f(row, "pa"), _f(row, "pitches_seen")
    if pa and pitches and pa >= 50:
        return (pitches / pa) >= 3.0
    ip, p_pitches = _f(row, "ip"), _f(row, "p_pitches")
    if ip and p_pitches and ip >= 20:
        return (p_pitches / ip) >= 10.0
    return False


def _suppress_pitch_metrics(out: dict, keys) -> dict:
    for k in keys:
        if k in out:
            out[k] = None
    return out


# ============================================================================
# HITTER METRICS
# ============================================================================

# Minimum denominators below which a rate is noise rather than signal.
MIN_PA = 20
MIN_BIP = 15
MIN_PITCHES = 50
MIN_FB = 5


def hitter_advanced(row: dict) -> dict:
    """All self-contained advanced hitting metrics for one season_stats row.

    Returns a dict of metric -> value (None where inputs are missing).
    """
    out: dict[str, Optional[float]] = {}

    pa = _f(row, "pa")
    ab = _f(row, "ab")
    h = _f(row, "hits")
    d2 = _f(row, "doubles")
    d3 = _f(row, "triples")
    hr = _f(row, "home_runs")
    bb = _f(row, "bb")
    ibb = _f(row, "ibb") or 0.0
    hbp = _f(row, "hbp")
    sf = _f(row, "sf")
    so = _f(row, "so")
    tb = _f(row, "total_bases")

    # ---- Slash line & wOBA ------------------------------------------------
    singles = None
    if None not in (h, d2, d3, hr):
        singles = h - d2 - d3 - hr

    w = woba_weights(row.get("season_year"))
    if singles is not None and bb is not None and pa:
        ubb = bb - ibb
        num = (w["bb"] * ubb + w["hbp"] * (hbp or 0) + w["1b"] * singles
               + w["2b"] * d2 + w["3b"] * d3 + w["hr"] * hr)
        den = ((ab or 0) + bb - ibb + (sf or 0) + (hbp or 0))
        out["woba"] = _div(num, den, MIN_PA)
    else:
        out["woba"] = None

    out["iso"] = _div((tb - h) if (tb is not None and h is not None) else None,
                      ab, MIN_PA)
    obp, slg = _f(row, "obp"), _f(row, "slg")
    out["ops"] = (obp + slg) if (obp is not None and slg is not None) else None

    # ---- Plate discipline (outcome-based) ---------------------------------
    out["k_pct"] = _div(so, pa, MIN_PA)
    out["bb_pct"] = _div(bb, pa, MIN_PA)
    out["ubb_pct"] = _div((bb - ibb) if bb is not None else None, pa, MIN_PA)
    out["bb_per_k"] = _div(bb, so, 0)
    if out["k_pct"] is not None and out["bb_pct"] is not None:
        out["bb_minus_k_pct"] = out["bb_pct"] - out["k_pct"]
    else:
        out["bb_minus_k_pct"] = None
    out["hbp_pct"] = _div(hbp, pa, MIN_PA)

    # ---- Plate discipline (pitch-level) -----------------------------------
    # The genuinely new layer: swing/miss behaviour, not just its outcomes.
    pitches = _f(row, "pitches_seen")
    swings = _f(row, "total_swings")
    misses = _f(row, "swings_and_misses")

    out["swing_pct"] = _div(swings, pitches, MIN_PITCHES)
    out["whiff_pct"] = _div(misses, swings, 20)          # misses per swing
    out["swstr_pct"] = _div(misses, pitches, MIN_PITCHES)  # misses per pitch
    out["contact_pct"] = (
        1.0 - out["whiff_pct"] if out["whiff_pct"] is not None else None)
    out["pitches_per_pa"] = _div(pitches, pa, MIN_PA)

    # A hitter who strikes out a lot *without* whiffing is taking called
    # strikes — a different (and more fixable) problem than bat-to-ball.
    if out["k_pct"] is not None and out["swstr_pct"] is not None:
        out["k_minus_swstr"] = out["k_pct"] - out["swstr_pct"]
    else:
        out["k_minus_swstr"] = None

    # ---- Batted-ball profile ----------------------------------------------
    bip = _f(row, "balls_in_play")
    gb = _sum(row, "ground_outs", "ground_hits")
    fb = _sum(row, "fly_outs", "fly_hits")
    ld = _sum(row, "line_outs", "line_hits")
    pu = _sum(row, "pop_outs", "pop_hits")

    out["gb_pct"] = _div(gb, bip, MIN_BIP)
    out["fb_pct"] = _div(fb, bip, MIN_BIP)
    out["ld_pct"] = _div(ld, bip, MIN_BIP)
    out["pu_pct"] = _div(pu, bip, MIN_BIP)
    out["gb_fb_ratio"] = _div(gb, fb, MIN_FB)
    out["hr_per_fb"] = _div(hr, fb, MIN_FB)
    # Air balls that are neither pop-ups nor grounders: the power-contact base.
    out["air_pct"] = (
        (out["fb_pct"] + out["ld_pct"])
        if (out["fb_pct"] is not None and out["ld_pct"] is not None) else None)

    out["babip"] = _div(
        (h - hr) if (h is not None and hr is not None) else None,
        (bip - hr) if (bip is not None and hr is not None) else None,
        MIN_BIP)

    # ---- Baserunning & situational ----------------------------------------
    sb, cs = _f(row, "stolen_bases"), _f(row, "caught_stealing")
    out["sb_attempts"] = _sum(row, "stolen_bases", "caught_stealing")
    out["sb_success_pct"] = _div(sb, out["sb_attempts"], 4)
    out["sb_rate"] = _div(sb, pa, MIN_PA)
    out["gidp_pct"] = _div(_f(row, "gidp"), _f(row, "gidp_opp"), 10)

    xbh = _sum(row, "doubles", "triples", "home_runs")
    out["xbh_pct"] = _div(xbh, pa, MIN_PA)
    out["hr_pct"] = _div(hr, pa, MIN_PA)

    # Drop the swing metrics where the league's pitch log is known bad.
    if not pitch_data_reliable(row):
        _suppress_pitch_metrics(out, _PITCH_DERIVED_HIT)

    return out


# ============================================================================
# PITCHER METRICS
# ============================================================================

MIN_BF = 30
MIN_IP = 10.0

# FIP constant: normalizes FIP to the league ERA. Level- and era-specific in
# principle; `fip_constant()` uses the league context when one is supplied.
FIP_CONSTANT_DEFAULT = 3.10


def pitcher_advanced(row: dict, lg_hr_per_fb: Optional[float] = None,
                     fip_constant: Optional[float] = None) -> dict:
    """All self-contained advanced pitching metrics for one season_stats row.

    `lg_hr_per_fb` enables xFIP (HR normalized to a league rate); without it
    xFIP is None. `fip_constant` overrides the default league-ERA anchor.
    """
    out: dict[str, Optional[float]] = {}

    ip = _f(row, "ip")
    bf = _f(row, "p_batters_faced")
    so = _f(row, "p_so")
    bb = _f(row, "p_bb")
    ibb = _f(row, "p_ibb") or 0.0
    hbp = _f(row, "p_hbp")
    hr = _f(row, "p_hr")
    h = _f(row, "p_hits")
    er = _f(row, "p_earned_runs")

    # ---- Rate stats off batters faced (better than per-9) ------------------
    out["k_pct"] = _div(so, bf, MIN_BF)
    out["bb_pct"] = _div(bb, bf, MIN_BF)
    out["ubb_pct"] = _div((bb - ibb) if bb is not None else None, bf, MIN_BF)
    out["hbp_pct"] = _div(hbp, bf, MIN_BF)
    if out["k_pct"] is not None and out["bb_pct"] is not None:
        out["k_minus_bb_pct"] = out["k_pct"] - out["bb_pct"]
    else:
        out["k_minus_bb_pct"] = None
    out["k_bb_ratio"] = _div(so, bb, 0)
    out["hr_pct"] = _div(hr, bf, MIN_BF)

    out["k9"] = _div(so, ip, MIN_IP)
    out["bb9"] = _div(bb, ip, MIN_IP)
    out["hr9"] = _div(hr, ip, MIN_IP)
    out["h9"] = _div(h, ip, MIN_IP)
    if out["k9"] is not None:
        out["k9"] *= 9.0
    if out["bb9"] is not None:
        out["bb9"] *= 9.0
    if out["hr9"] is not None:
        out["hr9"] *= 9.0
    if out["h9"] is not None:
        out["h9"] *= 9.0

    # ---- Pitch-level ------------------------------------------------------
    pitches = _f(row, "p_pitches")
    strikes = _f(row, "p_strikes")
    swings = _f(row, "p_total_swings")
    misses = _f(row, "p_swings_and_misses")

    out["strike_pct"] = _div(strikes, pitches, MIN_PITCHES)
    out["swing_pct"] = _div(swings, pitches, MIN_PITCHES)
    out["whiff_pct"] = _div(misses, swings, 20)
    out["swstr_pct"] = _div(misses, pitches, MIN_PITCHES)
    out["contact_pct"] = (
        1.0 - out["whiff_pct"] if out["whiff_pct"] is not None else None)
    out["pitches_per_bf"] = _div(pitches, bf, MIN_BF)
    out["pitches_per_ip"] = _div(pitches, ip, MIN_IP)

    # ---- Batted-ball profile allowed --------------------------------------
    bip = _f(row, "p_balls_in_play")
    gb = _sum(row, "p_ground_outs", "p_ground_hits")
    fb = _sum(row, "p_fly_outs", "p_fly_hits")
    ld = _sum(row, "p_line_outs", "p_line_hits")
    pu = _sum(row, "p_pop_outs", "p_pop_hits")

    out["gb_pct"] = _div(gb, bip, MIN_BIP)
    out["fb_pct"] = _div(fb, bip, MIN_BIP)
    out["ld_pct"] = _div(ld, bip, MIN_BIP)
    out["pu_pct"] = _div(pu, bip, MIN_BIP)
    out["gb_fb_ratio"] = _div(gb, fb, MIN_FB)
    out["hr_per_fb"] = _div(hr, fb, MIN_FB)
    out["babip_against"] = _div(
        (h - hr) if (h is not None and hr is not None) else None,
        (bip - hr) if (bip is not None and hr is not None) else None,
        MIN_BIP)

    # ---- ERA estimators ---------------------------------------------------
    c = fip_constant if fip_constant is not None else FIP_CONSTANT_DEFAULT

    # True FIP — includes HBP, unlike the k9/bb9/hr9 reconstruction this
    # replaces. Uses raw counts, so no per-9 rounding error.
    if None not in (hr, bb, so) and ip and ip >= MIN_IP:
        out["fip"] = (13.0 * hr + 3.0 * (bb + (hbp or 0)) - 2.0 * so) / ip + c
    else:
        out["fip"] = None

    # xFIP — HR replaced by the pitcher's fly balls at a league HR/FB rate.
    # Strips out the homer luck that makes small-sample FIP jumpy.
    if (lg_hr_per_fb is not None and fb is not None and None not in (bb, so)
            and ip and ip >= MIN_IP):
        x_hr = fb * lg_hr_per_fb
        out["xfip"] = (13.0 * x_hr + 3.0 * (bb + (hbp or 0)) - 2.0 * so) / ip + c
    else:
        out["xfip"] = None

    # SIERA-style estimator from the rate components. Not the licensed SIERA
    # formula — a transparent regression-shaped stand-in that rewards the same
    # things (Ks, avoiding walks, grounders) with diminishing returns.
    if None not in (out["k_pct"], out["bb_pct"]) and out["gb_pct"] is not None:
        k, b, g = out["k_pct"], out["bb_pct"], out["gb_pct"]
        out["siera_proxy"] = (
            6.15 - 18.0 * k + 12.0 * b - 2.4 * g + 10.0 * k * k - 6.0 * k * g)
    else:
        out["siera_proxy"] = None

    out["era"] = _div(er, ip, MIN_IP)
    if out["era"] is not None:
        out["era"] *= 9.0

    if None not in (h, bb, hbp, so, er, hr) and ip:
        # LOB% — share of baserunners stranded. High = lucky or high-leverage
        # escape artistry; regresses hard.
        reached = h + bb + (hbp or 0)
        denom = reached - 1.4 * hr
        out["lob_pct"] = _div(reached - er, denom, 1)
    else:
        out["lob_pct"] = None

    out["whip"] = _div(
        (h + bb) if (h is not None and bb is not None) else None, ip, MIN_IP)

    # ---- Role -------------------------------------------------------------
    g, gs = _f(row, "p_games"), _f(row, "p_games_started")
    out["start_pct"] = _div(gs, g, 0)
    out["ip_per_game"] = _div(ip, g, 0)
    out["is_starter"] = (
        1.0 if (out["start_pct"] is not None and out["start_pct"] >= 0.5)
        else (0.0 if out["start_pct"] is not None else None))

    # Drop the swing/strike metrics where the league's pitch log is known bad.
    if not pitch_data_reliable(row):
        _suppress_pitch_metrics(out, _PITCH_DERIVED_PIT)

    return out


# ============================================================================
# LEAGUE CONTEXT — (season, level) baselines for the relative metrics
# ============================================================================

def league_context(rows: list[dict]) -> dict:
    """Aggregate a set of season_stats rows (one season, one level) into the
    league baselines the relative metrics need.

    Pass every row for that season/level — the aggregate is a true league
    total, not an average of player rates, so low-PA players don't distort it.
    """
    tot: dict[str, float] = {}

    def add(key: str, val: Optional[float]) -> None:
        if val is not None:
            tot[key] = tot.get(key, 0.0) + val

    for r in rows:
        for k in ("pa", "ab", "hits", "doubles", "triples", "home_runs",
                  "bb", "ibb", "hbp", "sf", "so", "runs", "balls_in_play",
                  "ground_outs", "ground_hits", "fly_outs", "fly_hits",
                  "line_outs", "line_hits", "pop_outs", "pop_hits",
                  "pitches_seen", "total_swings", "swings_and_misses"):
            add(k, _f(r, k))
        for k in ("ip", "p_batters_faced", "p_earned_runs", "p_hr", "p_bb",
                  "p_so", "p_hbp", "p_hits", "p_balls_in_play",
                  "p_ground_outs", "p_ground_hits", "p_fly_outs", "p_fly_hits",
                  "p_pitches", "p_strikes", "p_total_swings",
                  "p_swings_and_misses"):
            add(k, _f(r, k))

    ctx: dict[str, Optional[float]] = {
        "n_rows": float(len(rows)),
        "pa": tot.get("pa"),
        "ip": tot.get("ip"),
    }

    # League wOBA and the runs-per-PA anchor that wRC+ is built on.
    lg = hitter_advanced({**tot, "season_year": rows[0].get("season_year")
                          if rows else None})
    ctx["lg_woba"] = lg.get("woba")
    ctx["lg_r_per_pa"] = _div(tot.get("runs"), tot.get("pa"), 100)
    ctx["lg_k_pct"] = lg.get("k_pct")
    ctx["lg_bb_pct"] = lg.get("bb_pct")
    ctx["lg_gb_pct"] = lg.get("gb_pct")
    ctx["lg_babip"] = lg.get("babip")
    ctx["lg_swstr_pct"] = lg.get("swstr_pct")

    # wOBA scale: ties the wOBA spread to the OBP scale for the era.
    ctx["woba_scale"] = woba_weights(
        rows[0].get("season_year") if rows else None)["scale"]

    # League HR/FB — the input xFIP needs.
    lg_fb = (tot.get("fly_outs", 0.0) + tot.get("fly_hits", 0.0)) or None
    ctx["lg_hr_per_fb"] = _div(tot.get("home_runs"), lg_fb, 50)

    # Pitching side.
    ctx["lg_era"] = _div(tot.get("p_earned_runs"), tot.get("ip"), 100)
    if ctx["lg_era"] is not None:
        ctx["lg_era"] *= 9.0
    p_fb = (tot.get("p_fly_outs", 0.0) + tot.get("p_fly_hits", 0.0)) or None
    ctx["lg_p_hr_per_fb"] = _div(tot.get("p_hr"), p_fb, 50)

    # FIP constant that makes league FIP equal league ERA.
    if (ctx["lg_era"] is not None and tot.get("ip")
            and None not in (tot.get("p_hr"), tot.get("p_bb"), tot.get("p_so"))):
        raw = (13.0 * tot["p_hr"] + 3.0 * (tot["p_bb"] + tot.get("p_hbp", 0.0))
               - 2.0 * tot["p_so"]) / tot["ip"]
        ctx["fip_constant"] = ctx["lg_era"] - raw
    else:
        ctx["fip_constant"] = FIP_CONSTANT_DEFAULT

    return ctx


# ============================================================================
# BATTED-BALL DIRECTION & CONTACT QUALITY
#
# Derived from a `batted_ball_profile` row (see data/sources/spray.py), which
# is scraped from game-feed hitData and therefore exists at every level.
# ============================================================================

MIN_BATTED_BALLS = 25


def spray_advanced(row: dict) -> dict:
    """Pull / centre / oppo rates and contact quality for one profile row."""
    out: dict[str, Optional[float]] = {}
    n = _f(row, "batted_balls")

    out["pull_pct"] = _div(_f(row, "pull_n"), n, MIN_BATTED_BALLS)
    out["center_pct"] = _div(_f(row, "center_n"), n, MIN_BATTED_BALLS)
    out["oppo_pct"] = _div(_f(row, "oppo_n"), n, MIN_BATTED_BALLS)

    # Pull-minus-oppo: one signed number for approach. Strongly positive is a
    # pull-heavy hitter, whose power tends to translate but whose batting
    # average is more shift- and defense-sensitive.
    if out["pull_pct"] is not None and out["oppo_pct"] is not None:
        out["pull_minus_oppo"] = out["pull_pct"] - out["oppo_pct"]
    else:
        out["pull_minus_oppo"] = None

    out["spray_angle_mean"] = _f(row, "spray_angle_mean")
    # Spread of the spray distribution — a proxy for all-fields ability that
    # does not depend on where the middle of that distribution sits.
    out["spray_angle_sd"] = _f(row, "spray_angle_sd")

    # Contact quality from the stringer's `hardness` grade.
    #
    # TREAT WITH SUSPICION. Measured across levels and eras, the grade is
    # overwhelmingly "medium" and barely discriminates:
    #
    #     MLB 2025   medium 83.5%   hard 14.3%   soft  2.2%
    #     AAA 2025   medium 83.4%   hard  9.7%   soft  6.9%
    #     AA  2025   medium 91.8%   hard  6.3%   soft  1.8%
    #     A   2025   medium 84.1%   hard  8.6%   soft  7.4%
    #     MLB 2015   medium 85.8%   hard  6.7%   soft  7.5%
    #     AA  2015   medium 97.6%   hard  1.3%   soft  1.1%
    #
    # Only 2-16% of batted balls get a non-medium tag, and the rate swings
    # with level and season far more than hitter populations plausibly do —
    # AA moved from 97.6% medium in 2015 to 91.8% in 2025. That is scorer
    # behaviour drifting, not contact quality. These are emitted because they
    # are free, but they are NOT a substitute for exit velocity and must not
    # enter the panel without per-level-season normalization; even normalized,
    # the discriminating power is thin.
    graded = _sum(row, "hard_n", "medium_n", "soft_n")
    out["hard_pct"] = _div(_f(row, "hard_n"), graded, MIN_BATTED_BALLS)
    out["soft_pct"] = _div(_f(row, "soft_n"), graded, MIN_BATTED_BALLS)
    if out["hard_pct"] is not None and out["soft_pct"] is not None:
        out["hard_minus_soft"] = out["hard_pct"] - out["soft_pct"]
    else:
        out["hard_minus_soft"] = None

    # Play-level trajectory. Redundant with the season-stat grid where both
    # exist, but it is the fallback where the season grid is missing.
    traj = _sum(row, "gb_n", "fb_n", "ld_n", "pu_n")
    out["gb_pct_pbp"] = _div(_f(row, "gb_n"), traj, MIN_BATTED_BALLS)
    out["fb_pct_pbp"] = _div(_f(row, "fb_n"), traj, MIN_BATTED_BALLS)
    out["ld_pct_pbp"] = _div(_f(row, "ld_n"), traj, MIN_BATTED_BALLS)
    out["pu_pct_pbp"] = _div(_f(row, "pu_n"), traj, MIN_BATTED_BALLS)

    return out


def wrc_plus(woba: Optional[float], ctx: dict,
             park_factor: Optional[float] = None) -> Optional[float]:
    """wRC+ — offense indexed to the league at that level (100 = average).

    This is the single most useful hitting number for cross-level comparison:
    a .350 wOBA in the Cal League and a .350 wOBA in the Florida State League
    are not the same accomplishment, and wRC+ says so.
    """
    lg_woba = ctx.get("lg_woba")
    scale = ctx.get("woba_scale")
    lg_rpa = ctx.get("lg_r_per_pa")
    if None in (woba, lg_woba, scale, lg_rpa) or not lg_rpa:
        return None
    wraa_per_pa = (woba - lg_woba) / scale
    val = (wraa_per_pa + lg_rpa)
    if park_factor:
        val /= park_factor
    return 100.0 * val / lg_rpa


def index_minus(value: Optional[float], lg_value: Optional[float],
                park_factor: Optional[float] = None) -> Optional[float]:
    """ERA-/FIP- style index where 100 is league average and lower is better."""
    if value is None or not lg_value:
        return None
    v = value / park_factor if park_factor else value
    return 100.0 * v / lg_value


def index_plus(value: Optional[float], lg_value: Optional[float]) -> Optional[float]:
    """K%+/BB%+ style index where 100 is league average and higher is more."""
    if value is None or not lg_value:
        return None
    return 100.0 * value / lg_value
