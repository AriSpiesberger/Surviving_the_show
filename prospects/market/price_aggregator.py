"""
Aggregate raw eBay Browse listings into per-player price summaries.

Output schema (per player + denominator group):
    player_id, name, card_year, denominator, n_listings, n_auctions, n_fixed,
    price_min, price_p25, price_median, price_mean, price_p75, price_max,
    top_listing_url, top_listing_title
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Optional

from prospects.market.ebay_client import ListingSummary
from prospects.market.listing_parser import ParsedListing, parse_title


@dataclass
class PriceSummary:
    player_id: str
    name: str
    card_year: int
    denominator: int
    n_listings: int
    n_auctions: int
    n_fixed: int
    price_min: Optional[float]
    price_p25: Optional[float]
    price_median: Optional[float]
    price_mean: Optional[float]
    price_p75: Optional[float]
    price_max: Optional[float]
    top_listing_url: Optional[str]
    top_listing_title: Optional[str]
    # Cheapest fixed-price (buy-now) listing — the actionable floor.
    # None if all qualifying listings are auctions.
    lowest_buynow_price: Optional[float]
    lowest_buynow_url: Optional[str]
    # Convenience flag: did we manage to find ANY accepted listing for this combo
    has_market: bool

    def as_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "card_year": self.card_year,
            "denominator": self.denominator,
            "n_listings": self.n_listings,
            "n_auctions": self.n_auctions,
            "n_fixed": self.n_fixed,
            "price_min": self.price_min,
            "price_p25": self.price_p25,
            "price_median": self.price_median,
            "price_mean": self.price_mean,
            "price_p75": self.price_p75,
            "price_max": self.price_max,
            "top_listing_url": self.top_listing_url,
            "top_listing_title": self.top_listing_title,
            "lowest_buynow_price": self.lowest_buynow_price,
            "lowest_buynow_url": self.lowest_buynow_url,
            "has_market": int(self.has_market),
        }


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(
    player_id: str,
    name: str,
    card_year: int,
    listings: Iterable[ListingSummary],
) -> list[PriceSummary]:
    """Parse listings, group by accepted denominator (/99, /499), and return
    one PriceSummary per (player, denom) combo that has at least one match."""

    by_denom: dict[int, list[tuple[ListingSummary, ParsedListing]]] = {
        0: [], 99: [], 499: [],
    }
    for li in listings:
        parsed = parse_title(li.title or "", name)
        if not parsed.accepted or parsed.denominator not in by_denom:
            continue
        by_denom[parsed.denominator].append((li, parsed))

    out: list[PriceSummary] = []
    for denom in (0, 99, 499):
        bucket = by_denom[denom]
        if not bucket:
            out.append(PriceSummary(
                player_id=player_id, name=name, card_year=card_year,
                denominator=denom, n_listings=0, n_auctions=0, n_fixed=0,
                price_min=None, price_p25=None, price_median=None,
                price_mean=None, price_p75=None, price_max=None,
                top_listing_url=None, top_listing_title=None,
                lowest_buynow_price=None, lowest_buynow_url=None,
                has_market=False,
            ))
            continue
        prices = sorted(
            (li.price_usd for li, _ in bucket if li.price_usd is not None)
        )
        if not prices:
            out.append(PriceSummary(
                player_id=player_id, name=name, card_year=card_year,
                denominator=denom, n_listings=len(bucket), n_auctions=0,
                n_fixed=0, price_min=None, price_p25=None,
                price_median=None, price_mean=None, price_p75=None,
                price_max=None, top_listing_url=None,
                top_listing_title=None, lowest_buynow_price=None,
                lowest_buynow_url=None, has_market=False,
            ))
            continue
        n_auctions = sum(1 for li, _ in bucket if "AUCTION" in (li.listing_type or ""))
        n_fixed = sum(1 for li, _ in bucket if "FIXED_PRICE" in (li.listing_type or ""))
        # "Top" listing = cheapest qualifying — that's the best buy candidate
        cheapest_li = min(
            (li for li, _ in bucket if li.price_usd is not None),
            key=lambda li: li.price_usd or float("inf"),
        )
        # Lowest fixed-price (buy-now) listing — separate from cheapest
        # overall, which may be an auction's current bid.
        buynow_li = None
        for li, _ in bucket:
            if li.price_usd is None:
                continue
            if "FIXED_PRICE" not in (li.listing_type or ""):
                continue
            if buynow_li is None or li.price_usd < (buynow_li.price_usd or float("inf")):
                buynow_li = li
        out.append(PriceSummary(
            player_id=player_id, name=name, card_year=card_year,
            denominator=denom,
            n_listings=len(bucket),
            n_auctions=n_auctions,
            n_fixed=n_fixed,
            price_min=min(prices),
            price_p25=_percentile(prices, 0.25),
            price_median=statistics.median(prices),
            price_mean=statistics.mean(prices),
            price_p75=_percentile(prices, 0.75),
            price_max=max(prices),
            top_listing_url=cheapest_li.item_url,
            top_listing_title=cheapest_li.title,
            lowest_buynow_price=buynow_li.price_usd if buynow_li else None,
            lowest_buynow_url=buynow_li.item_url if buynow_li else None,
            has_market=True,
        ))
    return out
