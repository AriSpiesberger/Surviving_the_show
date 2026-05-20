"""
Join the prices CSV from fetch_prospect_prices.py against the model grades
and rank by simple model-vs-market alignment metrics.

The point of this first cut is to surface mispricing candidates:
  - High predicted probability + low current price = potential buy
  - Low predicted probability + high current price = avoid / short candidate

We report two simple ratios:
    prob_per_dollar      = p_ESTABLISHED_MLB / price_median
    composite_per_dollar = composite_score_raw / price_median

Higher = relatively underpriced vs model.

Usage:
    python -m prospects.scripts.compare_model_vs_market \\
        --prices prices_top100.csv \\
        --denominator 99 \\
        --out edge_top100_99.csv
"""
from __future__ import annotations

import argparse
import csv


def _f(x, default=None):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="prices_top100.csv")
    parser.add_argument("--denominator", type=int, default=0,
                        choices=[0, 99, 499],
                        help="0 = base (unnumbered) auto, 99 = /99, 499 = /499")
    parser.add_argument("--out", default="edge_top100_99.csv")
    args = parser.parse_args()

    with open(args.prices, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(r.get("denominator", "0") or 0) == args.denominator
            and int(r.get("has_market", "0") or 0) == 1]
    print(f"Loaded {len(rows):,} prospect-rows with /{args.denominator} market data")

    out_rows = []
    for r in rows:
        price = _f(r.get("price_median"))
        p_est = _f(r.get("p_ESTABLISHED_MLB"))
        comp = _f(r.get("composite_score_raw"))
        if price is None or price <= 0:
            continue
        prob_per_dollar = (p_est or 0) / price * 100  # P points per $100
        comp_per_dollar = (comp or 0) / price
        out_rows.append({
            "name": r.get("name"),
            "player_id": r.get("player_id"),
            "card_year": r.get("card_year"),
            "denominator": args.denominator,
            "cur_level": r.get("cur_level"),
            "is_international": r.get("is_international"),
            "p_MLB_DEBUT": r.get("p_MLB_DEBUT"),
            "p_ESTABLISHED_MLB": r.get("p_ESTABLISHED_MLB"),
            "p_ALL_STAR_ONCE": r.get("p_ALL_STAR_ONCE"),
            "p_ELITE": r.get("p_ELITE"),
            "composite_score_raw": r.get("composite_score_raw"),
            "grade": r.get("grade"),
            "n_listings": r.get("n_listings"),
            "price_min": r.get("price_min"),
            "price_median": price,
            "price_max": r.get("price_max"),
            "prob_per_dollar_x100": round(prob_per_dollar, 4),
            "composite_per_dollar": round(comp_per_dollar, 4),
            "top_listing_url": r.get("top_listing_url"),
            "top_listing_title": r.get("top_listing_title"),
        })
    out_rows.sort(key=lambda r: r["composite_per_dollar"], reverse=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows):,} rows to {args.out}")

    print("\nTop 15 by composite-per-dollar (most underpriced vs model):")
    print(f"{'Player':<28} {'Lvl':<4} {'P(Est)':>7} {'Comp':>5} {'$med':>6} "
          f"{'edge':>6}")
    print("-" * 70)
    for r in out_rows[:15]:
        print(f"{(r['name'] or '')[:28]:<28} {r['cur_level'] or '':<4} "
              f"{_f(r['p_ESTABLISHED_MLB'], 0):>7.3f} "
              f"{_f(r['composite_score_raw'], 0):>5.2f} "
              f"{_f(r['price_median'], 0):>6.2f} "
              f"{r['composite_per_dollar']:>6.3f}")

    print("\nBottom 5 (most overpriced):")
    for r in out_rows[-5:]:
        print(f"{(r['name'] or '')[:28]:<28} {r['cur_level'] or '':<4} "
              f"{_f(r['p_ESTABLISHED_MLB'], 0):>7.3f} "
              f"{_f(r['composite_score_raw'], 0):>5.2f} "
              f"{_f(r['price_median'], 0):>6.2f} "
              f"{r['composite_per_dollar']:>6.3f}")


if __name__ == "__main__":
    main()
