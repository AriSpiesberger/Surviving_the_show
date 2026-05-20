"""
Foundation tests. Validates schema, labeling, storage.

Run:
    cd bowman-scanner
    python -m prospects.tests.test_foundation
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prospects.schema import (
    CardEV,
    CareerEvent,
    CareerOutcome,
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


def section(t):
    print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")


def check(label: str, ok: bool, details: str = ""):
    mark = "✓" if ok else "✗"
    extra = f"  -- {details}" if details else ""
    print(f"  {mark} {label}{extra}")
    if not ok:
        raise AssertionError(f"Check failed: {label}")


# ============================================================================
# SCHEMA
# ============================================================================

def test_career_event_enum():
    section("CareerEvent enum")
    events = CareerEvent.all_events()
    check("8 events defined", len(events) == 8)
    check("ordered ascending",
          all(int(events[i]) < int(events[i+1]) for i in range(len(events) - 1)))
    check("TOP_100 is first", events[0] == CareerEvent.TOP_100_PROSPECT)
    check("HOF_TRAJECTORY is last", events[-1] == CareerEvent.HOF_TRAJECTORY)


def test_stochastic_value():
    section("StochasticValue")
    sv = StochasticValue(value=0.380, stdev=0.030, n_observations=489)
    check("constructs with all fields", sv.value == 0.380 and sv.n_observations == 489)
    check("StochasticValue.point", StochasticValue.point(0.5).stdev == 0.0)
    try:
        StochasticValue(value=1, stdev=-0.1)
        check("rejects negative stdev", False)
    except ValueError:
        check("rejects negative stdev", True)


def test_prospect_minimal():
    section("Prospect — minimal construction")
    p = Prospect(
        player_id="renteria_2026",
        name="Francisco Renteria",
        is_pitcher=False,
        primary_position="OF",
        pedigree=Pedigree(
            is_international=True,
            international_signing_year=2026,
            signing_bonus_usd=4_000_000,
            origin="Venezuela",
        ),
    )
    check("builds with only required fields", p.name == "Francisco Renteria")
    check("pedigree captured", p.pedigree.signing_bonus_usd == 4_000_000)
    check("default risk", not p.risk.tj_history)


def test_prospect_full():
    section("Prospect — full record")
    p = Prospect(
        player_id="miller_aidan",
        name="Aidan Miller",
        is_pitcher=False,
        primary_position="SS",
        birth_date=date(2004, 6, 9),
        pedigree=Pedigree(
            draft_year=2023, draft_round=1, draft_pick=27,
            signing_bonus_usd=4_650_000, age_at_signing=19.0,
        ),
        current_org="PHI",
        current_level="AAA",
        highest_level_reached="AAA",
        risk=RiskFactors(has_current_injury=True, current_injury_type="lower_back"),
        rankings=[
            RankingSnapshot(
                as_of=date(2026, 3, 1), source="MLB Pipeline",
                overall_rank=23, org_rank=1, list_size=100,
            ),
        ],
        as_of_date=date(2026, 5, 13),
    )
    check("full record builds", p.current_level == "AAA")
    check("rankings preserved", len(p.rankings) == 1)


# ============================================================================
# OUTCOME LABELING
# ============================================================================

def test_label_busted():
    section("Outcome — busted prospect")
    o = CareerOutcome(
        player_id="bust_001",
        career_complete=True,
        career_pa=0, career_ip=0, career_war=0,
        all_star_selections=0,
        best_overall_rank=None,
        mlb_debut_year=None,
    )
    label_career(o)
    check("did not reach MLB", not o.events[CareerEvent.MLB_DEBUT])
    check("did not establish", not o.events[CareerEvent.ESTABLISHED_MLB])
    check("not All-Star", not o.events[CareerEvent.ALL_STAR_ONCE])
    check("no award", not o.events[CareerEvent.MAJOR_AWARD])
    check("no top-100 (no rank data)", not o.events[CareerEvent.TOP_100_PROSPECT])


def test_label_regular():
    section("Outcome — MLB regular (Bohm-like)")
    o = CareerOutcome(
        player_id="bohm_like",
        career_complete=False,  # active
        career_pa=2200, career_ip=0, career_war=8.5,
        all_star_selections=0,
        best_overall_rank=44,  # was a top-100 prospect
        mlb_debut_year=2020,
    )
    label_career(o)
    check("MLB debut", o.events[CareerEvent.MLB_DEBUT])
    check("established (2200 PA)", o.events[CareerEvent.ESTABLISHED_MLB])
    check("top-100 prospect", o.events[CareerEvent.TOP_100_PROSPECT])
    check("not top-25", not o.events[CareerEvent.TOP_25_PROSPECT])
    check("no All-Star", not o.events[CareerEvent.ALL_STAR_ONCE])
    check("no major award", not o.events[CareerEvent.MAJOR_AWARD])


def test_label_superstar():
    section("Outcome — superstar (Trout-like)")
    o = CareerOutcome(
        player_id="trout_like",
        career_complete=False,
        career_pa=7500, career_ip=0, career_war=85.0,
        all_star_selections=11,
        mvp_count=3,
        best_overall_rank=2,
        mlb_debut_year=2011,
    )
    label_career(o)
    check("MLB debut", o.events[CareerEvent.MLB_DEBUT])
    check("established", o.events[CareerEvent.ESTABLISHED_MLB])
    check("top-25", o.events[CareerEvent.TOP_25_PROSPECT])
    check("All-Star 1+", o.events[CareerEvent.ALL_STAR_ONCE])
    check("All-Star 3+", o.events[CareerEvent.ALL_STAR_THREE_PLUS])
    check("MVP", o.events[CareerEvent.MAJOR_AWARD])
    check("HOF trajectory", o.events[CareerEvent.HOF_TRAJECTORY])


def test_label_pitcher():
    section("Outcome — pitcher (Cy Young winner)")
    o = CareerOutcome(
        player_id="degrom_like",
        career_complete=False,
        career_pa=0, career_ip=1400, career_war=37.0,
        all_star_selections=4,
        cy_young_count=2,
        best_overall_rank=15,
        mlb_debut_year=2014,
    )
    label_career(o)
    check("MLB debut", o.events[CareerEvent.MLB_DEBUT])
    check("established (1400 IP)", o.events[CareerEvent.ESTABLISHED_MLB])
    check("top-25 prospect", o.events[CareerEvent.TOP_25_PROSPECT])
    check("All-Star 3+", o.events[CareerEvent.ALL_STAR_THREE_PLUS])
    check("Major award (Cy Young)", o.events[CareerEvent.MAJOR_AWARD])
    check("not HOF trajectory yet (37 WAR)",
          not o.events[CareerEvent.HOF_TRAJECTORY])


def test_cohort_summary():
    section("Cohort base rates")
    cohort = []
    # 50 busts
    for i in range(50):
        o = CareerOutcome(
            player_id=f"bust_{i}", career_complete=True,
            career_pa=0, career_ip=0, career_war=0,
            best_overall_rank=None, mlb_debut_year=None,
        )
        label_career(o)
        cohort.append(o)
    # 20 regulars
    for i in range(20):
        o = CareerOutcome(
            player_id=f"reg_{i}", career_complete=False,
            career_pa=2000, career_ip=0, career_war=8,
            all_star_selections=0,
            best_overall_rank=60, mlb_debut_year=2020,
        )
        label_career(o)
        cohort.append(o)
    # 5 stars
    for i in range(5):
        o = CareerOutcome(
            player_id=f"star_{i}", career_complete=False,
            career_pa=5000, career_ip=0, career_war=30,
            all_star_selections=4, mvp_count=1,
            best_overall_rank=10, mlb_debut_year=2015,
        )
        label_career(o)
        cohort.append(o)

    rates = base_rates(cohort)
    n = len(cohort)
    check("75 players in cohort", n == 75)
    check(f"MLB debut ≈ 25/75",
          abs(rates[CareerEvent.MLB_DEBUT] - 25/75) < 0.01,
          details=f"{rates[CareerEvent.MLB_DEBUT]:.3f}")
    check(f"Major award ≈ 5/75",
          abs(rates[CareerEvent.MAJOR_AWARD] - 5/75) < 0.01)
    print()
    print(describe_cohort(cohort))


# ============================================================================
# STORAGE
# ============================================================================

def test_storage_prospect_roundtrip():
    section("Storage — prospect roundtrip")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db = ProspectDB(path)
        p = Prospect(
            player_id="escobar_aroon",
            name="Aroon Escobar",
            is_pitcher=False,
            primary_position="2B",
            pedigree=Pedigree(
                is_international=True,
                international_signing_year=2022,
                signing_bonus_usd=300_000,
                age_at_signing=16.5,
            ),
            current_org="PHI",
            current_level="AA",
            highest_level_reached="AA",
            rankings=[
                RankingSnapshot(
                    as_of=date(2026, 3, 1), source="MLB Pipeline",
                    overall_rank=None, org_rank=8,
                ),
            ],
            as_of_date=date(2026, 5, 13),
        )
        db.upsert_prospect(p)
        loaded = db.get_prospect("escobar_aroon")
        check("persisted", loaded is not None)
        check("name preserved", loaded["name"] == "Aroon Escobar")
        check("org preserved", loaded["current_org"] == "PHI")
        check("level preserved", loaded["current_level"] == "AA")
        check("count_prospects works", db.count_prospects() == 1)

        rankings = db.get_rankings("escobar_aroon")
        check("ranking persisted", len(rankings) == 1)
        check("ranking org_rank", rankings[0]["org_rank"] == 8)
    finally:
        os.unlink(path)


def test_storage_season_stats():
    section("Storage — season stats")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db = ProspectDB(path)
        s = SeasonStats(
            player_id="escobar_aroon",
            season_year=2025,
            level="AA",
            org="PHI",
            age_during_season=20.5,
            pa=400,
            avg=0.295, woba=0.370, k_pct=0.18, bb_pct=0.10, iso=0.140,
            home_runs=15,
        )
        db.upsert_season_stats(s)
        loaded = db.get_season_stats("escobar_aroon")
        check("persisted", len(loaded) == 1)
        check("woba preserved", abs(loaded[0]["woba"] - 0.370) < 0.001)
        check("count_season_stats", db.count_season_stats() == 1)

        # Upsert again (idempotent)
        s2 = SeasonStats(
            player_id="escobar_aroon",
            season_year=2025,
            level="AA",
            org="PHI",
            pa=450,  # updated
        )
        db.upsert_season_stats(s2)
        loaded = db.get_season_stats("escobar_aroon")
        check("update preserved", loaded[0]["pa"] == 450)
        check("still single row (idempotent)", len(loaded) == 1)
    finally:
        os.unlink(path)


def test_storage_outcome():
    section("Storage — outcome roundtrip")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db = ProspectDB(path)
        o = CareerOutcome(
            player_id="trout_like",
            career_complete=False,
            career_pa=7500, career_war=85,
            all_star_selections=11,
            mvp_count=3,
            best_overall_rank=2,
            mlb_debut_year=2011,
        )
        label_career(o)
        db.upsert_outcome(o)
        loaded = db.get_outcome("trout_like")
        check("persisted", loaded is not None)
        check("career_war preserved", loaded.career_war == 85)
        check("HOF event preserved",
              loaded.events[CareerEvent.HOF_TRAJECTORY])
        check("MAJOR_AWARD preserved",
              loaded.events[CareerEvent.MAJOR_AWARD])
    finally:
        os.unlink(path)


def test_storage_prediction():
    section("Storage — prediction roundtrip")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db = ProspectDB(path)
        events = {
            CareerEvent.MLB_DEBUT: EventProbability(
                event=CareerEvent.MLB_DEBUT, p_mean=0.50, p_lo=0.35, p_hi=0.65,
            ),
            CareerEvent.ALL_STAR_ONCE: EventProbability(
                event=CareerEvent.ALL_STAR_ONCE, p_mean=0.15, p_lo=0.08, p_hi=0.25,
            ),
        }
        p = ProspectPrediction(
            player_id="test_player",
            as_of_date=date(2026, 5, 13),
            events=events,
            confidence=0.7,
            model_version="v0.1",
            features_used=18,
            features_imputed=4,
        )
        db.insert_prediction(p)
        loaded = db.get_latest_prediction("test_player")
        check("persisted", loaded is not None)
        check("confidence", loaded.confidence == 0.7)
        check("MLB prob preserved",
              abs(loaded.events[CareerEvent.MLB_DEBUT].p_mean - 0.50) < 0.001)
    finally:
        os.unlink(path)


def test_storage_best_rank():
    section("Storage — best_rank derivation")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db = ProspectDB(path)
        for as_of, rank in [
            (date(2024, 1, 1), 75),
            (date(2024, 6, 1), 45),
            (date(2025, 1, 1), 28),  # best
            (date(2025, 6, 1), 35),
        ]:
            db.upsert_ranking(
                "rising_prospect",
                RankingSnapshot(as_of=as_of, source="MLB Pipeline", overall_rank=rank),
            )
        best = db.best_rank("rising_prospect")
        check("best rank = 28", best == 28, details=f"got {best}")
    finally:
        os.unlink(path)


# ============================================================================
# RUNNER
# ============================================================================

def main():
    print("=" * 60)
    print("  PROSPECT CLASSIFIER FOUNDATION TESTS")
    print("  (event-based, v2)")
    print("=" * 60)

    test_career_event_enum()
    test_stochastic_value()
    test_prospect_minimal()
    test_prospect_full()

    test_label_busted()
    test_label_regular()
    test_label_superstar()
    test_label_pitcher()
    test_cohort_summary()

    test_storage_prospect_roundtrip()
    test_storage_season_stats()
    test_storage_outcome()
    test_storage_prediction()
    test_storage_best_rank()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
