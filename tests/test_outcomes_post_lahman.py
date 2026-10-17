"""
Outcome labels for seasons after Lahman's last year.

Lahman stops at 2021; outcomes.py now takes later seasons from the MLB Stats
API (season_stats MLB rows, mlb_awards). These pin the merge rule: Lahman is
untouched for the years it covers, and later years only come from the API.

Run:
    python -m pytest tests/test_outcomes_post_lahman.py
"""

from __future__ import annotations

from prospects.data.sources.outcomes import (
    established_year,
    extend_years,
    extend_yearly,
    sustained_seasons,
)

LAST = 2021


def test_lahman_years_are_kept_and_api_fills_only_later_years():
    lahman = [(2020, 50.0), (2021, 300.0)]
    api = {2021: 999.0, 2022: 600.0, 2023: 650.0}  # 2021 overlap must be ignored
    assert extend_yearly(lahman, api, LAST) == [(2020, 50.0), (2021, 300.0), (2022, 600.0), (2023, 650.0)]


def test_player_lahman_never_saw():
    # debuted 2022: no Lahman rows at all
    assert extend_yearly([], {2022: 132.0, 2023: 622.0}, LAST) == [(2022, 132.0), (2023, 622.0)]
    assert established_year([(2022, 132.0), (2023, 622.0)], []) == 2023  # cumulative 500 PA in 2023


def test_event_years_merge():
    assert extend_years([2019, 2021], [2021, 2023, 2024], LAST) == [2019, 2021, 2023, 2024]
    assert extend_years([], [2024, 2025], LAST) == [2024, 2025]


def test_established_unchanged_when_reached_inside_lahman():
    pa = extend_yearly([(2018, 450.0), (2019, 200.0)], {2022: 600.0}, LAST)
    assert established_year(pa, []) == 2019


def test_reliever_reaches_sustained_path_across_the_boundary():
    # 20+ IP in 2021 (Lahman), 2022 and 2023 (API): third meaningful season is 2023
    ip = extend_yearly([(2021, 25.0)], {2022: 30.0, 2023: 40.0}, LAST)
    assert sustained_seasons({}, dict(ip)) == [2021, 2022, 2023]
    assert established_year([], ip) == 2023


def test_no_mlb_time_is_not_established():
    assert established_year([], []) is None
    assert established_year([(2024, 40.0)], [(2024, 5.0)]) is None
