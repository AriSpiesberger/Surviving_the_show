"""Merge eBay base 1st Bowman Chrome auto prices into a buy list as new columns."""
import csv
import sys

buy_list_path, prices_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

price_cols = [
    "card_year", "n_listings", "n_auctions", "n_fixed",
    "price_min", "price_p25", "price_median", "price_mean",
    "price_p75", "price_max", "top_listing_url", "top_listing_title",
    "has_market",
]

prices_by_pid: dict[str, dict] = {}
with open(prices_path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("denominator") != "0":
            continue
        prices_by_pid[r["player_id"]] = {k: r.get(k, "") for k in price_cols}

with open(buy_list_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    in_fields = list(rows[0].keys())

out_fields = in_fields + [f"ebay_{c}" for c in price_cols]

n_merged = 0
for r in rows:
    p = prices_by_pid.get(r["player_id"])
    if p is not None and p.get("has_market") == "1":
        n_merged += 1
    for c in price_cols:
        r[f"ebay_{c}"] = (p or {}).get(c, "")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(rows)

print(f"merged prices for {n_merged}/{len(rows)} -> {out_path}")
