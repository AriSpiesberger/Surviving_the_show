"""
Point-in-time 40-man roster features (prospects.features.roster).

Run:
    python -m pytest tests/test_roster_features.py
"""

from __future__ import annotations

import math

import pytest

from prospects.features import roster


@pytest.fixture
def loaded(tmp_path):
    """Load a tiny cache covering 2020-2023, restore module state afterwards."""
    saved = (roster.ROSTER40_NAMES, roster._SEASONS, roster._FIRST, roster._LAST)
    p = tmp_path / "roster40_nov.csv"
    p.write_text(
        "season,date,team_id,mlbam_id,status\n"
        "2020,2020-11-30,110,1,A\n"
        "2021,2021-11-30,110,1,A\n"
        "2021,2021-11-30,111,2,A\n"
        "2023,2023-11-30,110,1,A\n"
        "2023,2023-11-30,112,3,A\n"  # covered seasons span 2020-2023
    )
    roster._load(p)
    yield roster
    roster.ROSTER40_NAMES, roster._SEASONS, roster._FIRST, roster._LAST = saved


def test_off_by_default_adds_nothing():
    # the test environment does not set ROSTER40_ON, so the contract is unchanged
    assert roster.ROSTER40_NAMES == []
    assert roster.roster40_dict(1, 2021) == {}


def test_point_in_time_counts_only_seasons_up_to_as_of(loaded):
    assert loaded.roster40_dict(1, 2020) == {"r40_on_nov": 1.0, "r40_novs": 1.0}
    assert loaded.roster40_dict(1, 2021) == {"r40_on_nov": 1.0, "r40_novs": 2.0}
    assert loaded.roster40_dict(1, 2022) == {"r40_on_nov": 0.0, "r40_novs": 2.0}  # off the roster in 2022
    assert loaded.roster40_dict(1, 2023) == {"r40_on_nov": 1.0, "r40_novs": 3.0}


def test_not_yet_added_player_is_zero_not_missing(loaded):
    assert loaded.roster40_dict(3, 2021) == {"r40_on_nov": 0.0, "r40_novs": 0.0}  # added only in 2023
    assert loaded.roster40_dict(999, 2021) == {"r40_on_nov": 0.0, "r40_novs": 0.0}  # never on one


def test_outside_covered_seasons_is_nan(loaded):
    for yr in (2019, 2024):
        d = loaded.roster40_dict(1, yr)
        assert math.isnan(d["r40_on_nov"]) and math.isnan(d["r40_novs"])


def test_mlbam_id_forms(loaded):
    assert loaded.roster40_dict("2", 2021)["r40_on_nov"] == 1.0
    assert loaded.roster40_dict(2.0, 2021)["r40_on_nov"] == 1.0
    assert loaded.roster40_dict(None, 2021) == {"r40_on_nov": 0.0, "r40_novs": 0.0}
