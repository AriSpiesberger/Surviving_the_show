"""
Construct eBay search queries for a prospect's 1st Bowman Chrome auto.

Card-year mapping:
  - US draftee:  card_year = draft_year + 1
  - IFA:         card_year = first MiLB year (== signing_year + 1 typically)
                 We backfilled prospects.draft_year = first MiLB year for IFAs,
                 so the same +1 logic works in both cases? NO — for IFAs the
                 draft_year column IS already first_milb_year, not signing_year,
                 so their 1st Bowman is the SAME year, not year+1.

Resolution:
  - drafted (is_international == 0):  card_year = draft_year + 1
  - IFA     (is_international == 1):  card_year = draft_year  (== first MiLB)

The query string targets autographs and the specific year.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def _clean_name(name: str) -> str:
    n = _strip_accents(name or "")
    # Collapse initials like "C.J." -> "CJ" so quoted-phrase eBay queries
    # match listings titled "CJ Kayfus" (the dominant form on eBay).
    n = re.sub(r"\b((?:[A-Za-z]\.){2,})", lambda m: m.group(1).replace(".", ""), n)
    # Single trailing initial like "J. Smith" -> "J Smith"
    n = re.sub(r"\b([A-Za-z])\.(?=\s)", r"\1", n)
    # Strip suffixes like Jr., II, III
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\.?$", "", n, flags=re.IGNORECASE).strip()
    return n


@dataclass
class CardSpec:
    player_id: str
    name: str
    name_query: str         # cleaned name for use in the query string
    card_year: int          # year of the 1st Bowman release
    is_international: bool
    queries: list[str]      # query strings to issue, in priority order


def build_card_spec(prospect_row: dict) -> CardSpec | None:
    """Map a prospect row to its first-Bowman card spec + query strings.

    Returns None when we can't determine a card year."""
    is_intl = bool(int(prospect_row.get("is_international") or 0))
    dy = prospect_row.get("draft_year")
    start_year = prospect_row.get("start_year")
    # IFA fallback: many IFAs have draft_year empty in CSV. Use start_year
    # (their first observed MiLB year) which equals their card-issue year.
    if dy is None or dy == "":
        if is_intl and start_year not in (None, ""):
            card_year = int(start_year)
            dy = int(start_year)  # store for record
        else:
            return None
    else:
        dy = int(dy)
        # Drafted: 1st Bowman is the year AFTER the draft.
        # IFA (when both draft_year and start_year are present and equal):
        # we set draft_year = first MiLB year via the IFA backfill, so the
        # card-issue year is THAT year, not +1.
        card_year = dy if is_intl else dy + 1
    name = prospect_row.get("name") or ""
    if not name:
        return None

    clean = _clean_name(name)

    # Query construction. Skip the year — the "1st Bowman" tag is printed on
    # the card itself and surfaces in listing titles regardless of release
    # year, so we don't need to guess (some players' 1st Chrome auto lands
    # the year after their expected debut release).
    queries = [
        f'"{clean}" 1st Bowman Chrome auto',
        f'"{clean}" 1st Bowman auto',
        f'"{clean}" Bowman auto refractor',
        # Wider fallback if the tight query returns nothing
        f'{clean} Bowman auto',
    ]
    return CardSpec(
        player_id=prospect_row.get("player_id", ""),
        name=name,
        name_query=clean,
        card_year=card_year,
        is_international=is_intl,
        queries=queries,
    )
