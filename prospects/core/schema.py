"""
prospects/core/schema.py
====================

Data structures for the prospect classifier system.

Key design decisions:
- Event-based outputs, not survival curve through levels. Card value steps at
  discrete career events (made top-100, MLB debut, All-Star, MVP), not
  continuously with WAR.
- Minimal required fields. Most fields are Optional because data availability
  varies wildly across sources and eras.
- Single Prospect type covers both historical (training) and current (inference)
  players. The presence of an Outcome record is what distinguishes them.
- All numeric uncertainty handled via StochasticValue (point estimate + stdev).
  Used sparingly for stats with small sample sizes or subjective grades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Optional


# ============================================================================
# CAREER EVENTS — the outputs of our classifier
# ============================================================================

class CareerEvent(IntEnum):
    """
    Discrete events that move card prices.

    Each event is a binary the player either triggered or didn't.
    Probability at each event is what the classifier outputs.

    Ordered roughly by difficulty (and card price impact).
    """
    TOP_100_PROSPECT = 1      # Ever ranked in MLB Pipeline or BA top 100
    TOP_25_PROSPECT = 2       # Ever ranked top 25 (sharper signal)
    MLB_DEBUT = 3             # Any MLB game
    ESTABLISHED_MLB = 4       # 500+ career PA or 200+ career IP
    ALL_STAR_ONCE = 5         # 1+ All-Star selection
    ALL_STAR_THREE_PLUS = 6   # 3+ All-Star selections
    MAJOR_AWARD = 7           # MVP, Cy Young, or Rookie of the Year
    HOF_TRAJECTORY = 8        # 50+ career WAR or HOF inducted

    @classmethod
    def all_events(cls) -> list["CareerEvent"]:
        return sorted(cls, key=int)


# ============================================================================
# STOCHASTIC VALUES — for inputs with measurement uncertainty
# ============================================================================

@dataclass
class StochasticValue:
    """A measurement with associated uncertainty.

    Used for stats with small sample sizes (e.g., 50-PA stretch in AA) and for
    subjective grades (tool grades, rankings). Point estimates with full
    confidence use stdev=0.
    """
    value: float
    stdev: float = 0.0
    n_observations: Optional[int] = None  # sample size if applicable

    def __post_init__(self):
        if self.stdev < 0:
            raise ValueError(f"stdev must be non-negative, got {self.stdev}")

    @classmethod
    def point(cls, value: float) -> "StochasticValue":
        return cls(value=value, stdev=0.0)


# ============================================================================
# SEASON STATS — one row per player-season-level
# ============================================================================

@dataclass
class SeasonStats:
    """
    One player's performance in one season at one level.

    A player who played at AA and AAA in 2024 has two SeasonStats rows.
    Most fields are Optional because hitters don't have pitching stats and
    vice versa, and not all sources provide every metric.
    """
    player_id: str
    season_year: int
    level: str                # "AAA", "AA", "A+", "A", "A-", "RK", "MLB", "NCAA-D1"
    org: Optional[str] = None
    age_during_season: Optional[float] = None

    # Hitter
    pa: int = 0
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    woba: Optional[float] = None
    iso: Optional[float] = None
    k_pct: Optional[float] = None
    bb_pct: Optional[float] = None
    babip: Optional[float] = None
    home_runs: Optional[int] = None
    stolen_bases: Optional[int] = None

    # Pitcher
    ip: float = 0.0
    era: Optional[float] = None
    fip: Optional[float] = None
    whip: Optional[float] = None
    k9: Optional[float] = None
    bb9: Optional[float] = None
    hr9: Optional[float] = None
    velo_avg: Optional[float] = None

    # ---- Hitter raw counting stats -------------------------------------
    # Stored raw (not as rates) so any rate — including era-specific wOBA
    # weights — can be recomputed later without a re-pull. All available
    # from the MLB Stats API stitch endpoint for MLB *and* every MiLB level
    # back to 2005.
    ab: Optional[int] = None
    hits: Optional[int] = None
    doubles: Optional[int] = None
    triples: Optional[int] = None
    bb: Optional[int] = None
    ibb: Optional[int] = None
    hbp: Optional[int] = None
    sf: Optional[int] = None
    sac_bunts: Optional[int] = None
    so: Optional[int] = None
    total_bases: Optional[int] = None
    runs: Optional[int] = None
    rbi: Optional[int] = None
    caught_stealing: Optional[int] = None
    gidp: Optional[int] = None
    gidp_opp: Optional[int] = None
    left_on_base: Optional[int] = None
    reached_on_error: Optional[int] = None

    # Hitter batted-ball profile (out + hit components by trajectory)
    ground_outs: Optional[int] = None
    air_outs: Optional[int] = None
    fly_outs: Optional[int] = None
    line_outs: Optional[int] = None
    pop_outs: Optional[int] = None
    ground_hits: Optional[int] = None
    fly_hits: Optional[int] = None
    line_hits: Optional[int] = None
    pop_hits: Optional[int] = None
    balls_in_play: Optional[int] = None

    # Hitter plate discipline (pitch-level aggregates)
    pitches_seen: Optional[int] = None
    total_swings: Optional[int] = None
    swings_and_misses: Optional[int] = None

    # ---- Pitcher raw counting stats ------------------------------------
    p_batters_faced: Optional[int] = None
    p_ab: Optional[int] = None
    p_hits: Optional[int] = None
    p_doubles: Optional[int] = None
    p_triples: Optional[int] = None
    p_hr: Optional[int] = None
    p_bb: Optional[int] = None
    p_ibb: Optional[int] = None
    p_hbp: Optional[int] = None
    p_so: Optional[int] = None
    p_sf: Optional[int] = None
    p_sac_bunts: Optional[int] = None
    p_earned_runs: Optional[int] = None
    p_runs: Optional[int] = None
    p_total_bases: Optional[int] = None
    p_gidp: Optional[int] = None
    p_gidp_opp: Optional[int] = None
    p_balks: Optional[int] = None
    p_wild_pitches: Optional[int] = None
    p_pickoffs: Optional[int] = None
    p_outs: Optional[int] = None

    # Pitcher batted-ball profile
    p_ground_outs: Optional[int] = None
    p_air_outs: Optional[int] = None
    p_fly_outs: Optional[int] = None
    p_line_outs: Optional[int] = None
    p_pop_outs: Optional[int] = None
    p_ground_hits: Optional[int] = None
    p_fly_hits: Optional[int] = None
    p_line_hits: Optional[int] = None
    p_pop_hits: Optional[int] = None
    p_balls_in_play: Optional[int] = None

    # Pitcher pitch-level / discipline induced
    p_pitches: Optional[int] = None
    p_strikes: Optional[int] = None
    p_total_swings: Optional[int] = None
    p_swings_and_misses: Optional[int] = None

    # Pitcher role / usage
    p_games: Optional[int] = None
    p_games_started: Optional[int] = None
    p_complete_games: Optional[int] = None
    p_saves: Optional[int] = None
    p_holds: Optional[int] = None
    p_babip_allowed: Optional[float] = None
    p_avg_against: Optional[float] = None
    p_obp_against: Optional[float] = None
    p_slg_against: Optional[float] = None

    # Defense/positional context
    primary_position: Optional[str] = None

    # ---- Within-cohort percentile ranks (level, season_year) ----
    # Derived, not pulled: written by data/backfills/percentile_backfill.py
    # after every pull. Stored as columns rather than computed in the feature
    # layer because nine separate call sites do `SELECT * FROM season_stats`
    # and group by player; attaching in memory means attaching in all nine,
    # and a single miss silently feeds the model all-MISSING columns — or,
    # worse, trains with them and scores without.
    #
    # These go stale if the cohort gains rows without a recompute, which the
    # in-progress season does on every pull. The backfill is wired into
    # refresh.py directly after `pull` for that reason.
    pct_woba: Optional[float] = None
    pct_iso: Optional[float] = None
    pct_k_pct: Optional[float] = None
    pct_bb_pct: Optional[float] = None
    pct_avg: Optional[float] = None
    pct_obp: Optional[float] = None
    pct_slg: Optional[float] = None
    pct_hr_per_pa: Optional[float] = None
    pct_sb_per_pa: Optional[float] = None
    pct_swstr_pct: Optional[float] = None
    pct_contact_pct: Optional[float] = None
    pct_pitches_per_pa: Optional[float] = None
    pct_gb_pct: Optional[float] = None
    pct_fb_pct: Optional[float] = None
    pct_ld_pct: Optional[float] = None
    pct_hr_per_fb: Optional[float] = None
    pct_babip: Optional[float] = None
    pct_era: Optional[float] = None
    pct_k9: Optional[float] = None
    pct_bb9: Optional[float] = None
    pct_fip: Optional[float] = None
    pct_whip: Optional[float] = None
    pct_hr9: Optional[float] = None
    pct_p_swstr_pct: Optional[float] = None
    pct_p_contact_pct: Optional[float] = None
    pct_p_strike_pct: Optional[float] = None
    pct_p_pitches_per_bf: Optional[float] = None
    pct_p_gb_pct: Optional[float] = None
    pct_p_fb_pct: Optional[float] = None
    pct_p_ld_pct: Optional[float] = None
    pct_p_k_bb_ratio: Optional[float] = None
    pct_p_babip_against: Optional[float] = None
    pct_p_siera_proxy: Optional[float] = None


# ============================================================================
# PLATOON SPLITS — one row per player-season-level-side
# ============================================================================

@dataclass
class PlatoonSplit:
    """A player's line against one handedness of opposing pitcher (or batter,
    for pitcher rows). `side` is "L" or "R" (the *opponent's* handedness).

    Available for MLB and every MiLB level via the statsapi statSplits
    endpoint with sitCodes=vl,vr.
    """
    player_id: str
    season_year: int
    level: str
    side: str                 # "L" or "R"
    is_pitcher: bool = False

    pa: int = 0
    ab: Optional[int] = None
    hits: Optional[int] = None
    doubles: Optional[int] = None
    triples: Optional[int] = None
    home_runs: Optional[int] = None
    bb: Optional[int] = None
    ibb: Optional[int] = None
    hbp: Optional[int] = None
    sf: Optional[int] = None
    so: Optional[int] = None
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    ops: Optional[float] = None
    babip: Optional[float] = None
    ground_outs: Optional[int] = None
    air_outs: Optional[int] = None


# ============================================================================
# FIELDING — one row per player-season-level-position
# ============================================================================

@dataclass
class FieldingStats:
    """Defensive workload and rate stats at one position.

    The MLB Stats API exposes no advanced defensive metric (no DRS/UZR/OAA)
    for MiLB, so this captures workload (innings by position), reliability
    (errors/chances) and range (range factor). Position *scarcity* — how
    premium the spots a player can hold are — is the real signal here.
    """
    player_id: str
    season_year: int
    level: str
    position: str             # "C", "SS", "CF", ...

    games: Optional[int] = None
    games_started: Optional[int] = None
    innings: Optional[float] = None
    chances: Optional[int] = None
    putouts: Optional[int] = None
    assists: Optional[int] = None
    errors: Optional[int] = None
    throwing_errors: Optional[int] = None
    double_plays: Optional[int] = None
    fielding_pct: Optional[float] = None
    range_factor_per9: Optional[float] = None


# ============================================================================
# PARK FACTORS — one row per team-season-level
# ============================================================================

@dataclass
class ParkFactor:
    """Multiplicative run/HR environment for one team's home park.

    Computed from the team's own home-vs-road splits (the standard
    "halved" park factor), regressed toward 1.0 by sample size. 1.00 is
    neutral; 1.15 means the park inflates the stat by 15%.
    """
    team_id: str
    season_year: int
    level: str
    org: Optional[str] = None

    pf_runs: Optional[float] = None
    pf_hr: Optional[float] = None
    pf_hits: Optional[float] = None
    pf_doubles: Optional[float] = None
    pf_triples: Optional[float] = None
    pf_so: Optional[float] = None
    pf_bb: Optional[float] = None
    home_games: Optional[int] = None
    road_games: Optional[int] = None


# ============================================================================
# STATCAST — MLB only, 2015+ (batted-ball tracking and pitch characteristics)
# ============================================================================

@dataclass
class StatcastBatting:
    """Statcast batted-ball and plate-discipline profile. MLB only."""
    player_id: str
    season_year: int

    batted_balls: Optional[int] = None
    avg_exit_velo: Optional[float] = None
    max_exit_velo: Optional[float] = None
    avg_launch_angle: Optional[float] = None
    barrel_pct: Optional[float] = None
    hard_hit_pct: Optional[float] = None
    sweet_spot_pct: Optional[float] = None
    xba: Optional[float] = None
    xslg: Optional[float] = None
    xwoba: Optional[float] = None
    xwobacon: Optional[float] = None
    # Plate discipline (zone-aware — the piece season stats cannot give)
    o_swing_pct: Optional[float] = None     # chase rate
    z_swing_pct: Optional[float] = None
    swing_pct: Optional[float] = None
    o_contact_pct: Optional[float] = None
    z_contact_pct: Optional[float] = None
    contact_pct: Optional[float] = None
    zone_pct: Optional[float] = None
    whiff_pct: Optional[float] = None
    # Batted-ball direction
    pull_pct: Optional[float] = None
    center_pct: Optional[float] = None
    oppo_pct: Optional[float] = None
    gb_pct: Optional[float] = None
    fb_pct: Optional[float] = None
    ld_pct: Optional[float] = None
    iffb_pct: Optional[float] = None
    hr_per_fb: Optional[float] = None
    # Value / context
    wrc_plus: Optional[float] = None
    woba_fg: Optional[float] = None
    war: Optional[float] = None
    def_runs: Optional[float] = None
    bsr: Optional[float] = None
    sprint_speed: Optional[float] = None


@dataclass
class StatcastPitching:
    """Statcast pitch characteristics and arsenal. MLB only."""
    player_id: str
    season_year: int

    avg_exit_velo_against: Optional[float] = None
    barrel_pct_against: Optional[float] = None
    hard_hit_pct_against: Optional[float] = None
    xera: Optional[float] = None
    xwoba_against: Optional[float] = None
    # Overall velocity / movement
    fb_velo: Optional[float] = None         # four-seam average velocity
    fb_spin: Optional[float] = None
    avg_velo: Optional[float] = None        # all pitches
    # Discipline induced
    o_swing_pct: Optional[float] = None
    z_contact_pct: Optional[float] = None
    contact_pct: Optional[float] = None
    swstr_pct: Optional[float] = None
    zone_pct: Optional[float] = None
    first_strike_pct: Optional[float] = None
    # Batted ball allowed
    gb_pct: Optional[float] = None
    fb_pct: Optional[float] = None
    ld_pct: Optional[float] = None
    hr_per_fb: Optional[float] = None
    # Value
    fip_fg: Optional[float] = None
    xfip: Optional[float] = None
    siera: Optional[float] = None
    war: Optional[float] = None


@dataclass
class PitchArsenal:
    """One pitch type in a pitcher's arsenal for one season. MLB only."""
    player_id: str
    season_year: int
    pitch_type: str           # "FF", "SL", "CH", ...

    usage_pct: Optional[float] = None
    avg_velo: Optional[float] = None
    avg_spin: Optional[float] = None
    whiff_pct: Optional[float] = None
    put_away_pct: Optional[float] = None
    run_value_per_100: Optional[float] = None
    xwoba_against: Optional[float] = None


@dataclass
class CatcherDefense:
    """Framing / throwing metrics for catchers. MLB only."""
    player_id: str
    season_year: int

    framing_runs: Optional[float] = None
    strike_rate: Optional[float] = None
    called_pitches: Optional[int] = None
    pop_time: Optional[float] = None
    arm_strength: Optional[float] = None
    exchange_time: Optional[float] = None


# ============================================================================
# PROSPECT — the input to the classifier
# ============================================================================

@dataclass
class Pedigree:
    """How the player entered pro baseball."""
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_pick: Optional[int] = None
    signing_bonus_usd: Optional[float] = None
    age_at_signing: Optional[float] = None
    is_international: bool = False
    international_signing_year: Optional[int] = None
    origin: str = ""              # college name or country


@dataclass
class RiskFactors:
    """Negative attributes that depress career probability."""
    tj_history: bool = False
    has_current_injury: bool = False
    current_injury_type: str = ""


@dataclass
class RankingSnapshot:
    """Where a player ranked on a major prospect list at a point in time."""
    as_of: date
    source: str                    # "MLB Pipeline", "Baseball America", "FanGraphs"
    overall_rank: Optional[int]    # None if outside the published list
    org_rank: Optional[int] = None
    list_size: int = 100


@dataclass
class Prospect:
    """
    Complete prospect record.

    Used for both historical players (training data) and current prospects
    (inference). The presence of a matching CareerOutcome record indicates
    a historical/labeled player.
    """
    # Identity
    player_id: str                 # MLBAM ID preferred; fall back to fangraphs_id
    name: str
    is_pitcher: bool
    primary_position: str          # "SS", "C", "RHP", etc

    # Demographics
    birth_date: Optional[date] = None

    # Origin
    pedigree: Pedigree = field(default_factory=Pedigree)

    # Current state (for inference players)
    current_org: Optional[str] = None
    current_level: Optional[str] = None
    highest_level_reached: Optional[str] = None

    # Risk
    risk: RiskFactors = field(default_factory=RiskFactors)

    # Rankings history
    rankings: list[RankingSnapshot] = field(default_factory=list)

    # Metadata
    notes: str = ""
    as_of_date: Optional[date] = None


# ============================================================================
# CAREER OUTCOME — training labels for historical players
# ============================================================================

@dataclass
class CareerOutcome:
    """
    Resolved or near-resolved career outcome for a historical player.

    The `events` dict is the training label — which CareerEvents the player
    triggered. These are derived from underlying career stats by
    outcome_labels.label_career.
    """
    player_id: str
    career_complete: bool          # retired or stable trajectory

    # Underlying stats used to derive event triggers
    career_pa: int = 0
    career_ip: float = 0.0
    career_war: float = 0.0
    # Number of MLB seasons with meaningful usage (>=20 IP or >=100 PA). Used
    # for the role-fair ESTABLISHED_MLB rule so multi-year relievers / utility
    # players count as established even below the 200 IP / 500 PA volume bars.
    n_sustained_mlb_seasons: int = 0
    all_star_selections: int = 0
    mvp_count: int = 0
    cy_young_count: int = 0
    roy_count: int = 0
    is_hof_inducted: bool = False
    is_hof_likely: bool = False    # 50+ WAR
    best_overall_rank: Optional[int] = None  # best prospect ranking ever achieved

    # Career timeline
    pro_debut_year: Optional[int] = None
    mlb_debut_year: Optional[int] = None
    final_mlb_year: Optional[int] = None

    # Derived event flags — set by outcome_labels.label_career
    events: dict[CareerEvent, bool] = field(default_factory=dict)


# ============================================================================
# CLASSIFIER OUTPUT
# ============================================================================

@dataclass
class EventProbability:
    """P(event triggered) with credible interval."""
    event: CareerEvent
    p_mean: float
    p_lo: float                    # 10th percentile (90% CI lower)
    p_hi: float                    # 90th percentile (90% CI upper)

    def __post_init__(self):
        if not 0 <= self.p_mean <= 1:
            raise ValueError(f"p_mean must be in [0, 1], got {self.p_mean}")
        if not 0 <= self.p_lo <= self.p_mean <= self.p_hi <= 1:
            raise ValueError(
                f"Invalid CI: lo={self.p_lo}, mean={self.p_mean}, hi={self.p_hi}"
            )

    def ci_width(self) -> float:
        return self.p_hi - self.p_lo


@dataclass
class ProspectPrediction:
    """
    Complete classifier output for one prospect.

    Contains P(triggered) for each of the 8 career events with credible intervals.
    Confidence is a composite measure (narrower CIs across events = higher).
    """
    player_id: str
    as_of_date: date
    events: dict[CareerEvent, EventProbability] = field(default_factory=dict)
    confidence: float = 0.0        # 0-1, derived from CI widths
    model_version: str = ""
    features_used: int = 0
    features_imputed: int = 0


# ============================================================================
# CARD PRICING — the size model
# ============================================================================

@dataclass
class EventMultiplier:
    """
    Card price multiplier when a player triggers a specific event.

    These multipliers are calibrated empirically from historical card prices
    conditional on career outcomes. The "baseline" is what the card would
    trade at if the player never triggered any event beyond their current state.
    """
    event: CareerEvent
    multiplier_mean: float
    multiplier_stdev: float
    n_observations: int = 0


@dataclass
class CardEV:
    """Expected card value derived from prediction + multipliers."""
    player_id: str
    product: str                   # "2022 Bowman Chrome Draft"
    parallel: str                  # "Green Refractor /99"
    current_market_price: float

    expected_value: float          # E[future price]
    ev_lo: float                   # 10th percentile of EV distribution
    ev_hi: float                   # 90th percentile

    edge: float                    # (EV - current_price) / current_price
    multiple: float                # EV / current_price
