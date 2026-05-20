"""
prospects/schema.py
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

    # Defense/positional context
    primary_position: Optional[str] = None


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
