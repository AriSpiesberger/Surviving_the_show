"""
Title parser for eBay Browse API listings.

Target card: 1st Bowman Chrome Auto, numbered /499 or /99.

Strategy: regex on the listing title.
  - Must contain player name (we trust the search; verify lightly).
  - Must mention Bowman + a 'Chrome' or '1st Bowman' indicator.
  - Must indicate an autograph (auto, autograph, signed).
  - Must contain "/499" or "/99" (and not a more-restrictive denominator).
  - Exclude lots, custom cards, breaks (probabilistic items), refractor/auto-
    less printings.

Returns a normalized result dict so the aggregator can group cleanly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Denominator tiers we track. 0 = BASE (unnumbered base auto refractor),
# the highest-volume tier. /99 and /499 are numbered parallels.
ACCEPTED_DENOMINATORS = [0]

# Any of these in a title is grounds for rejecting the listing as the wrong
# variant (numbered to fewer than 99 = a rarer parallel we don't track here).
TIGHTER_DENOMINATORS = ["75", "50", "25", "10", "5", "1/1", "1 OF 1", "ONE OF ONE"]

# Parallel keywords that imply a colored / atomic / superfractor numbered
# parallel even if the numbering itself isn't visible. Use to reject "Gold
# Refractor" type listings from the BASE tier specifically.
COLORED_PARALLEL_RE = re.compile(
    r"\b(gold|orange|atomic|superfractor|red|blue|purple|aqua|x[- ]?fractor|"
    r"green|black|pink|sepia|fuchsia|magenta|negative)\s*(refractor|auto|ref)?\b",
    re.IGNORECASE,
)

# Auto indicators
AUTO_RE = re.compile(r"\b(auto(?:graph)?|signed|signature)\b", re.IGNORECASE)

# Bowman / 1st-Bowman indicators
BOWMAN_RE = re.compile(r"\bbowman\b", re.IGNORECASE)
ONE_FIRST_RE = re.compile(r"\b(1st|first|1\s?st)\b", re.IGNORECASE)
CHROME_RE = re.compile(r"\bchrome\b", re.IGNORECASE)

# Lot / bulk / break exclusions
EXCLUDE_RE = re.compile(
    r"\b(lot of|case break|hot box|mystery|repack|player break|"
    r"break of|reprint|custom|aceo|sketch|graded\s*[1-9]\.|"
    r"complete set|team set)\b",
    re.IGNORECASE,
)

# Graded slabs — user trades raw only. Match a TPG paired with a grade
# number or "gem mint" only; standalone TPG names are too false-positive-prone
# (e.g., "PSA-worthy", random three-letter player initials).
GRADED_RE = re.compile(
    r"\b("
    # TPG + numeric grade: "PSA 10", "BGS 9.5", "SGC 10", "CGC 9", "CSG 9.5"
    r"(?:psa|bgs|sgc|cgc|csg|hga|gma|isa)\s*(?:gem\s*)?(?:mt|mint)?\s*\d{1,2}(?:\.\d)?"
    # TPG + "gem mint" / "gem mt"
    r"|(?:psa|bgs|sgc|cgc|csg|hga|gma|isa)\s+gem\s*(?:mt|mint)"
    # generic "graded" markers
    r"|gem\s*mint(?:\s*\d)?"             # "Gem Mint", "Gem Mint 10"
    r"|graded\s*(?:card|gem|mint|\d)"    # "graded card", "graded 10"
    r"|slabbed"                          # "slabbed"
    r")\b",
    re.IGNORECASE,
)

# Generic numbering regex: "/NN" or "#/NN" or "NN/NN" (serial-numbered)
NUMBER_RE = re.compile(r"/(\d{1,4})\b")


@dataclass
class ParsedListing:
    raw_title: str
    accepted: bool
    denominator: Optional[int]      # 99 or 499 when accepted
    is_auto: bool
    is_bowman: bool
    is_chrome: bool
    excluded_reason: Optional[str]


def _denominators_in_title(title: str) -> list[int]:
    return [int(d) for d in NUMBER_RE.findall(title)]


def parse_title(title: str, player_full_name: str) -> ParsedListing:
    t = title or ""
    is_auto = bool(AUTO_RE.search(t))
    is_bowman = bool(BOWMAN_RE.search(t))
    is_chrome = bool(CHROME_RE.search(t))

    # Lightweight name presence check (last-name token only — eBay sellers
    # spell names wildly; first-name match is too brittle).
    last = player_full_name.strip().split()[-1] if player_full_name.strip() else ""
    name_ok = last.lower() in t.lower() if last else True

    if EXCLUDE_RE.search(t):
        return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                             "lot/break/custom/graded-flaw")
    if GRADED_RE.search(t):
        return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                             "graded slab (raw-only scope)")
    if not name_ok:
        return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                             "name token missing")
    if not is_auto:
        return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                             "no auto indicator")
    if not is_bowman:
        return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                             "not a bowman card")

    dens = _denominators_in_title(t)
    if not dens:
        # No numbering => BASE tier (unnumbered Bowman Chrome 1st auto).
        # Reject if a colored-parallel keyword appears (a colored parallel
        # without a printed denominator is a misprint or photo artifact;
        # safer to drop these from BASE).
        if COLORED_PARALLEL_RE.search(t):
            return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                                 "colored parallel without numbering")
        # Optional: require "chrome" or "1st" indicator so we don't pick up
        # base flagship Bowman (non-Chrome) cards.
        if not (is_chrome or ONE_FIRST_RE.search(t)):
            return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                                 "no chrome/1st indicator")
        return ParsedListing(t, True, 0, is_auto, is_bowman, is_chrome, None)

    # Any numbered listing is a parallel — reject (base-only scope).
    return ParsedListing(t, False, None, is_auto, is_bowman, is_chrome,
                         f"numbered parallel /{dens[0]} (base-only scope)")
