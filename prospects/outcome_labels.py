"""
prospects/outcome_labels.py
=============================

Converts a player's career stats into binary CareerEvent labels.

These labels are the training targets for the classifier. One player gets
8 binary labels (one per CareerEvent). The classifier learns to predict
each from prospect-stage features.

The thresholds are precise so labeling is deterministic and reproducible.
"""

from __future__ import annotations

from prospects.schema import CareerEvent, CareerOutcome


# ============================================================================
# THRESHOLDS
# ============================================================================

ESTABLISHED_MLB_PA = 500          # ~one full position-player season
ESTABLISHED_MLB_IP = 200          # ~one starter year or 3 RP years

TOP_100_RANK_THRESHOLD = 100      # ever appeared on a major top-100 list
TOP_25_RANK_THRESHOLD = 25        # ever appeared in top 25

ALL_STAR_ONCE = 1
ALL_STAR_MULTIPLE = 3             # 3+ All-Star appearances = sustained star

HOF_TRAJECTORY_WAR = 50.0         # career WAR threshold for HOF-tier


# ============================================================================
# LABELING
# ============================================================================

def label_career(outcome: CareerOutcome) -> CareerOutcome:
    """
    Populate outcome.events with binary triggers for each CareerEvent.

    Mutates and returns the outcome record. Idempotent.

    Note: TOP_100_PROSPECT and TOP_25_PROSPECT require best_overall_rank
    to be set. If unset, those events default to False (we don't know whether
    they were ranked, so we conservatively say no).
    """
    events: dict[CareerEvent, bool] = {}

    # Prospect ranking events
    best_rank = outcome.best_overall_rank
    events[CareerEvent.TOP_100_PROSPECT] = (
        best_rank is not None and best_rank <= TOP_100_RANK_THRESHOLD
    )
    events[CareerEvent.TOP_25_PROSPECT] = (
        best_rank is not None and best_rank <= TOP_25_RANK_THRESHOLD
    )

    # MLB level events
    events[CareerEvent.MLB_DEBUT] = outcome.mlb_debut_year is not None
    events[CareerEvent.ESTABLISHED_MLB] = (
        outcome.career_pa >= ESTABLISHED_MLB_PA
        or outcome.career_ip >= ESTABLISHED_MLB_IP
    )

    # Star events
    events[CareerEvent.ALL_STAR_ONCE] = (
        outcome.all_star_selections >= ALL_STAR_ONCE
    )
    events[CareerEvent.ALL_STAR_THREE_PLUS] = (
        outcome.all_star_selections >= ALL_STAR_MULTIPLE
    )

    # Major awards
    events[CareerEvent.MAJOR_AWARD] = (
        outcome.mvp_count > 0
        or outcome.cy_young_count > 0
        or outcome.roy_count > 0
    )

    # HOF trajectory
    events[CareerEvent.HOF_TRAJECTORY] = (
        outcome.is_hof_inducted
        or outcome.is_hof_likely
        or outcome.career_war >= HOF_TRAJECTORY_WAR
    )

    outcome.events = events
    return outcome


# ============================================================================
# COHORT SUMMARIES — for sanity-checking training data
# ============================================================================

def base_rates(outcomes: list[CareerOutcome]) -> dict[CareerEvent, float]:
    """Fraction of cohort that triggered each event."""
    if not outcomes:
        return {e: 0.0 for e in CareerEvent.all_events()}
    n = len(outcomes)
    return {
        event: sum(1 for o in outcomes if o.events.get(event, False)) / n
        for event in CareerEvent.all_events()
    }


def describe_cohort(outcomes: list[CareerOutcome]) -> str:
    """Pretty-print event base rates for a labeled cohort."""
    rates = base_rates(outcomes)
    lines = [
        f"Cohort size: {len(outcomes):,} players",
        "",
        f"{'Event':<25} {'% triggered':>12}",
        "-" * 40,
    ]
    for event in CareerEvent.all_events():
        pct = rates[event] * 100
        lines.append(f"{event.name:<25} {pct:>11.2f}%")
    return "\n".join(lines)
