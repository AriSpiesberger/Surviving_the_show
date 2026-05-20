"""
prospects/
==========

Probabilistic prospect classifier with event-based outputs.

Pipeline:
    1. Ingest data from pybaseball, armstjc MiLB repo, ncaa_bbStats (free).
    2. Label historical players via outcome_labels.label_career.
    3. Train classifier on features -> P(event triggered) for each CareerEvent.
    4. Apply to current prospects.
    5. Combine probabilities with event-multipliers (size model) -> card EV.
    6. Integrate with scanner.py for live buy signals.

Module structure:
    schema.py            - core data types
    outcome_labels.py    - convert career stats to training labels
    storage.py           - SQLite persistence
    ingestion/           - data acquisition
        pybaseball_loader.py    - MLB career, draft, Lahman
        milb_stats.py           - MiLB stats from armstjc/MLB Stats API
        ncaa_loader.py          - college stats from ncaa_bbStats
        run_bulk_pull.py        - orchestrator for one-time historical pull
    features/            - feature engineering
    classifier/          - model training and inference
    ev/                  - expected value calculator
    ranking/             - ranked buy lists
    tests/               - test suite
"""

from prospects.schema import (
    CardEV,
    CareerEvent,
    CareerOutcome,
    EventMultiplier,
    EventProbability,
    Pedigree,
    Prospect,
    ProspectPrediction,
    RankingSnapshot,
    RiskFactors,
    SeasonStats,
    StochasticValue,
)
from prospects.outcome_labels import (
    base_rates,
    describe_cohort,
    label_career,
)
from prospects.storage import ProspectDB

__all__ = [
    # Schema
    "CareerEvent",
    "StochasticValue",
    "Pedigree",
    "RiskFactors",
    "RankingSnapshot",
    "Prospect",
    "SeasonStats",
    "CareerOutcome",
    "EventProbability",
    "ProspectPrediction",
    "EventMultiplier",
    "CardEV",
    # Labeling
    "label_career",
    "base_rates",
    "describe_cohort",
    # Storage
    "ProspectDB",
]
