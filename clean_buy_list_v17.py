"""Reshape v1.17 buy list + fresh prices into a scannable sheet.

Output columns (sorted by buy_rank):
    rank, name, tier, bucket, age, pos, lvl, org,
    top100%, debut%, est_mlb%, star+%, breakout%,
    buy_low, median, n_listings, listing
"""
from __future__ import annotations

import csv

BUY_LIST = "buy_list_v1.17_enriched.csv"
PRICES = "prices_v1.17_FINAL.csv"
OUT = "buy_list_v1.17_CLEAN.csv"


def to_f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def pct(v):
    f = to_f(v)
    return f"{f * 100:.0f}%" if f is not None else ""


def money(v):
    f = to_f(v)
    return f"${f:,.2f}" if f is not None else ""


def tier_label(buy_rank: int) -> str:
    if buy_rank <= 25:
        return "S"
    if buy_rank <= 75:
        return "A"
    if buy_rank <= 150:
        return "B"
    return "C"


prices_by_pid: dict[str, dict] = {}
with open(PRICES, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("denominator") != "0":
            continue
        if r.get("has_market") != "1":
            continue
        prices_by_pid[r["player_id"]] = r

with open(BUY_LIST, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

out_rows = []
for r in rows:
    pid = r["player_id"]
    p = prices_by_pid.get(pid, {})
    buy_rank = int(float(r.get("buy_rank") or 0))
    out_rows.append({
        "rank": buy_rank,
        "name": r.get("name", ""),
        "tier": tier_label(buy_rank),
        "bucket": r.get("bucket", ""),
        "age": r.get("age_at_snap", ""),
        "pos": r.get("primary_position", ""),
        "lvl": r.get("cur_level_2026", ""),
        "org": r.get("current_org", ""),
        "top100%": pct(r.get("p_TOP_100_PROSPECT")),
        "debut%": pct(r.get("p_MLB_DEBUT")),
        "est_mlb%": pct(r.get("p_ESTABLISHED_MLB")),
        "star+%": pct(r.get("p_STAR_PLUS_ELITE")),
        "breakout%": pct(r.get("p_breakout")),
        "buy_low": money(p.get("lowest_buynow_price")),
        "median": money(p.get("price_median")),
        "n_listings": p.get("n_listings", ""),
        "listing": p.get("lowest_buynow_url") or p.get("top_listing_url", ""),
    })

out_rows.sort(key=lambda r: r["rank"])
fields = list(out_rows[0].keys())

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

n_with_market = sum(1 for r in out_rows if r["buy_low"] or r["median"])
print(f"wrote {len(out_rows)} rows ({n_with_market} with market data) -> {OUT}")
