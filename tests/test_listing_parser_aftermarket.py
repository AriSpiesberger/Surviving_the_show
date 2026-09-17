"""Base-tier price comps must be certified pack-pulled Chrome autos only."""
import pytest

from prospects.market.listing_parser import parse_title

REJECT = [
    ("2024 Bowman Draft Jared Thomas 1st Bowman Non-Chrome Auto Signed IP Rockies", "Jared Thomas"),
    ("2025 Bowman Draft Chrome Ethan Conrad #BDC-17 1st Bowman RC IP Auto GTP", "Ethan Conrad"),
    ("2025 Bowman Chrome Marek Houston 1st Auto TTM signed", "Marek Houston"),
    ("2024 Bowman Chrome Kane Kepley 1st Prospect Autograph JSA COA", "Kane Kepley"),
    ("2024 Bowman 1st Roc Riggio signed in person auto", "Roc Riggio"),
    ("2024 Bowman Chrome 1st Auto Hand Signed Joe Whitman", "Joe Whitman"),
]
ACCEPT = [
    ("2024 Bowman Tavian Josenberger Chrome Auto 1st #CPA-TJ Orioles", "Tavian Josenberger"),
    ("Bowman 2026 Bowman Chrome Victor Figueroa 1st Bowman Auto Orioles #CPA-VF", "Victor Figueroa"),
    ("2024 Bowman Chrome 1st Auto Phillip Dipoto CPA-PD", "Phillip Dipoto"),
    ("2025 Bowman Chrome Draft Seaver King 1st Bowman Auto CDA-SK Nationals", "Seaver King"),
    # "ip" inside a word or a lowercase surname must not trip the IP token
    ("2024 Bowman Chrome 1st Auto Tripp Clark CPA-TC", "Tripp Clark"),
    ("2023 Bowman Chrome 1st Bowman Auto Basil Coates CPA-BC", "Basil Coates"),
]


@pytest.mark.parametrize("title,name", REJECT)
def test_aftermarket_and_non_chrome_rejected(title, name):
    assert not parse_title(title, name).accepted


@pytest.mark.parametrize("title,name", ACCEPT)
def test_certified_chrome_autos_accepted(title, name):
    r = parse_title(title, name)
    assert r.accepted and r.denominator == 0
