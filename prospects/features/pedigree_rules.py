"""Rules for which static pedigree fields may reach a model.

Principle (2026-09-17): a field is usable only on rows where its coverage does
not depend on how the player's career turned out. Anything else is leakage —
the data got richer for players who made it, so *presence* encodes the label.

signing_bonus_usd
    drafts 2005-2016: on file for ~9% of players who never debuted vs ~60% of
        those who did (within the same round tier a known bonus means 2-4x the
        debut rate). Sourced by outcome. The MLB draft API carries no bonuses
        before ~2019, so it cannot be backfilled completely.
    drafts 2017+, rounds 11+: 73-99% covered and still higher for debuters
        (2018: 96% vs 85%).
    drafts 2017+, rounds 1-10: 100% covered in every year, identical for
        debuted and never-debuted. This is also exactly where slot values
        exist, so bonus_vs_slot stays meaningful.
    international: 0% covered in every era.
    => usable only for draft_year >= 2017 and draft_round <= 10.

See tools/leak_audit.py for the checks that established this and that should be
re-run after any data refresh.
"""
from __future__ import annotations

BONUS_COMPLETE_FROM_YEAR = 2017
BONUS_COMPLETE_MAX_ROUND = 10


def usable_signing_bonus(prospect: dict):
    """signing_bonus_usd if it comes from the completely-covered block, else None."""
    bonus = prospect.get("signing_bonus_usd")
    dy, rd = prospect.get("draft_year"), prospect.get("draft_round")
    try:
        ok = (dy is not None and rd is not None
              and int(dy) >= BONUS_COMPLETE_FROM_YEAR
              and int(rd) <= BONUS_COMPLETE_MAX_ROUND
              and int(prospect.get("is_international") or 0) == 0)
    except (TypeError, ValueError):
        ok = False
    return bonus if ok else None
