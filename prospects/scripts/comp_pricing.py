"""
Cohort-relative pricing: for each priced prospect, find the K most similar
priced prospects and compute their median price. The "edge" is how far the
target's price is below its peer-median.

Why this works when sold-price data is sparse: we don't need an absolute
price model. We just compare "Carson Taylor priced at $5" to "other AAA
catchers with similar P(Est) priced at $30 median" → 6x gap = strong signal.

Usage:
    python -m prospects.scripts.comp_pricing \\
        --prices prices_top2000.csv \\
        --denominator 0 \\
        --k 8 \\
        --out comp_base.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np


# Feature columns used to define similarity. Each gets z-scored, then we use
# Euclidean distance on the standardized vector.
SIM_FEATURES = [
    "composite_score_raw",
    "p_ESTABLISHED_MLB",
    "p_MLB_DEBUT",
    "draft_round",       # missing for IFAs → imputed
]
# Categorical features used as "hard" similarity filters (must match).
HARD_FILTERS = ["cur_level"]


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="prices_top2000.csv")
    parser.add_argument("--denominator", type=int, default=0,
                        choices=[0, 99, 499])
    parser.add_argument("--k", type=int, default=8,
                        help="Neighbors per prospect")
    parser.add_argument("--out", default="comp_base.csv")
    args = parser.parse_args()

    with open(args.prices, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if int(r.get("denominator", "0") or 0) == args.denominator
                and int(r.get("has_market", "0") or 0) == 1
                and _f(r.get("price_median")) is not None
                and _f(r.get("price_median")) > 0]
    print(f"Loaded {len(rows):,} priced prospects at /{args.denominator}")
    if len(rows) < args.k + 1:
        raise SystemExit(f"Need at least {args.k+1} priced prospects for k={args.k}")

    # Build similarity feature matrix (impute missing with column median)
    feat_cols = SIM_FEATURES
    X = np.full((len(rows), len(feat_cols)), np.nan, dtype=np.float64)
    for j, c in enumerate(feat_cols):
        vals = np.array([_f(r.get(c), np.nan) for r in rows], dtype=np.float64)
        med = np.nanmedian(vals) if np.isfinite(np.nanmedian(vals)) else 0.0
        vals = np.where(np.isnan(vals), med, vals)
        # z-score
        mu, sd = vals.mean(), vals.std() or 1.0
        X[:, j] = (vals - mu) / sd
    levels = np.array([r.get("cur_level", "") for r in rows])
    prices = np.array([_f(r.get("price_median")) for r in rows])

    out_rows = []
    for i, r in enumerate(rows):
        # Hard-filter to same cur_level
        same_lvl = (levels == levels[i])
        same_lvl[i] = False
        if same_lvl.sum() < args.k:
            # fall back to all if not enough same-level peers
            same_lvl = np.ones(len(rows), dtype=bool)
            same_lvl[i] = False
        cands = np.where(same_lvl)[0]
        # Distances
        dists = np.linalg.norm(X[cands] - X[i], axis=1)
        order = np.argsort(dists)[: args.k]
        neighbors = cands[order]
        n_prices = prices[neighbors]
        comp_median = float(np.median(n_prices))
        comp_p25 = float(np.percentile(n_prices, 25))
        comp_p75 = float(np.percentile(n_prices, 75))
        target_price = float(prices[i])
        # Edge: how far below the peer median is the target? Positive = underpriced.
        rel = target_price / comp_median if comp_median > 0 else 1.0
        # Discount in pct: (1 - target/comp_median) * 100
        discount_pct = (1.0 - rel) * 100.0
        # Also z-score of price among peers
        peer_mean = n_prices.mean()
        peer_sd = n_prices.std() or 1.0
        z = (target_price - peer_mean) / peer_sd  # negative = cheaper than peers
        out_rows.append({
            "player_id": r.get("player_id"),
            "name": r.get("name"),
            "cur_level": r.get("cur_level"),
            "draft_year": r.get("card_year") and (
                int(r["card_year"]) - 1 if r.get("is_international") not in ("1", 1) else int(r["card_year"])
            ),
            "is_international": r.get("is_international"),
            "p_MLB_DEBUT": r.get("p_MLB_DEBUT"),
            "p_ESTABLISHED_MLB": r.get("p_ESTABLISHED_MLB"),
            "p_ALL_STAR_ONCE": r.get("p_ALL_STAR_ONCE"),
            "p_ELITE": r.get("p_ELITE"),
            "composite_score_raw": r.get("composite_score_raw"),
            "grade": r.get("grade"),
            "denominator": args.denominator,
            "price_median": target_price,
            "comp_median": round(comp_median, 2),
            "comp_p25": round(comp_p25, 2),
            "comp_p75": round(comp_p75, 2),
            "rel_to_comp": round(rel, 3),
            "discount_pct": round(discount_pct, 1),
            "peer_z": round(z, 3),
            "n_listings": r.get("n_listings"),
            "top_listing_url": r.get("top_listing_url"),
        })

    # Sort by most underpriced (lowest rel_to_comp = biggest discount)
    out_rows.sort(key=lambda r: r["rel_to_comp"])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows):,} rows to {args.out}\n")

    print(f"Top 15 most-underpriced vs peers (/{args.denominator}):")
    print(f"{'Player':<28} {'Lvl':<4} {'P(Est)':>7} {'$':>7} {'peer$':>7} {'disc%':>7}")
    print("-" * 70)
    for r in out_rows[:15]:
        print(f"{(r['name'] or '')[:28]:<28} {r['cur_level'] or '':<4} "
              f"{_f(r['p_ESTABLISHED_MLB'], 0):>7.3f} "
              f"{r['price_median']:>7.2f} {r['comp_median']:>7.2f} "
              f"{r['discount_pct']:>6.1f}%")

    print(f"\nTop 5 most-overpriced vs peers:")
    for r in out_rows[-5:]:
        print(f"{(r['name'] or '')[:28]:<28} {r['cur_level'] or '':<4} "
              f"{_f(r['p_ESTABLISHED_MLB'], 0):>7.3f} "
              f"{r['price_median']:>7.2f} {r['comp_median']:>7.2f} "
              f"{r['discount_pct']:>6.1f}%")


if __name__ == "__main__":
    main()
